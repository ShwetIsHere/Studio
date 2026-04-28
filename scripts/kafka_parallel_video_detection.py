import argparse
import base64
import json
import os
import queue
import socket
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import uuid
from hdfs import InsecureClient
from kafka import KafkaConsumer, KafkaProducer

# Initialize HDFS WebHDFS Client
hdfs_client = InsecureClient('http://localhost:9870', user='root')
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import (
    KafkaError,
    KafkaTimeoutError,
    NoBrokersAvailable,
    NodeNotReadyError,
    TopicAlreadyExistsError,
)
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process videos via Kafka/local transport and detect fire + weapon."
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        required=True,
        help="List of input video paths (use 2 or more for parallel processing)",
    )
    parser.add_argument(
        "--bootstrap-servers",
        type=str,
        default="localhost:9092",
        help="Kafka bootstrap server(s)",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="cctv_video_frames",
        help="Kafka topic used to publish frames",
    )
    parser.add_argument(
        "--transport",
        type=str,
        default="auto",
        choices=["auto", "kafka", "local"],
        help="Frame transport mode: auto (try Kafka then fallback), kafka, or local queue",
    )
    parser.add_argument(
        "--group-id",
        type=str,
        default="fire_weapon_detectors",
        help="Kafka consumer group id",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=3,
        help="Publish 1 frame every N frames from each video",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument(
        "--fire-model",
        type=str,
        default="",
        help="Path to fire model weights (.pt)",
    )
    parser.add_argument(
        "--weapon-model",
        type=str,
        default="",
        help="Path to weapon model weights (.pt)",
    )
    parser.add_argument(
        "--save-annotated",
        action="store_true",
        help="Save annotated detection frames in output folder",
    )
    parser.add_argument(
        "--show",
        dest="show",
        action="store_true",
        default=True,
        help="Show a live preview window while consuming Kafka frames (default: enabled)",
    )
    parser.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="Disable preview window",
    )
    return parser.parse_args()


def find_default_model_paths(project_root: Path) -> tuple[Path, Path]:
    fire_path = project_root / "models" / "runs" / "train" / "fire_yolo11" / "weights" / "fire_best.pt"
    weapon_path = project_root / "models" / "runs" / "train" / "fire_yolo11" / "weights" / "weapon_best.pt"
    return fire_path, weapon_path


def ensure_dirs(project_root: Path) -> tuple[Path, Path, Path]:
    alerts_dir = project_root / "alerts" / "logs"
    frames_dir = project_root / "alerts" / "frames"
    out_dir = project_root / "output" / "kafka_processed"
    alerts_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    return alerts_dir, frames_dir, out_dir


def _parse_bootstrap_servers(bootstrap_servers: str) -> list[tuple[str, int]]:
    parsed: list[tuple[str, int]] = []
    for item in bootstrap_servers.split(","):
        server = item.strip()
        if not server:
            continue
        if ":" in server:
            host, port_str = server.rsplit(":", 1)
            try:
                port = int(port_str)
            except ValueError:
                port = 9092
        else:
            host = server
            port = 9092
        parsed.append((host.strip(), port))
    return parsed


def assert_kafka_reachable(bootstrap_servers: str, timeout: float = 2.0) -> None:
    targets = _parse_bootstrap_servers(bootstrap_servers)
    if not targets:
        raise SystemExit("No valid --bootstrap-servers provided.")

    for host, port in targets:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                print(f"[KAFKA] Reachable broker: {host}:{port}")
                return
        except OSError:
            continue

    target_text = ", ".join(f"{h}:{p}" for h, p in targets)
    raise SystemExit(
        "Kafka broker is not reachable. "
        f"Tried: {target_text}.\n"
        "Start Kafka first, then rerun this script.\n"
        "Tip (Windows): start Zookeeper, then Kafka server, and keep both terminals open."
    )


def ensure_topic_ready(bootstrap_servers: str, topic: str, partitions: int = 6) -> bool:
    try:
        admin = KafkaAdminClient(
            bootstrap_servers=bootstrap_servers,
            client_id="fire_weapon_topic_admin",
            request_timeout_ms=10000,
        )
    except (NoBrokersAvailable, NodeNotReadyError, KafkaError) as exc:
        print(
            "[KAFKA] Warning: topic admin check skipped because broker metadata is not fully ready "
            f"({type(exc).__name__}). Continuing with producer/consumer startup."
        )
        return False

    try:
        existing = set(admin.list_topics())
        if topic not in existing:
            admin.create_topics(
                new_topics=[
                    NewTopic(name=topic, num_partitions=partitions, replication_factor=1)
                ],
                validate_only=False,
            )
            print(f"[KAFKA] Created topic '{topic}' with {partitions} partitions.")
            return True
        else:
            print(f"[KAFKA] Topic '{topic}' is available.")
            return True
    except TopicAlreadyExistsError:
        print(f"[KAFKA] Topic '{topic}' already exists.")
        return True
    except (NodeNotReadyError, KafkaError) as exc:
        print(
            "[KAFKA] Warning: topic metadata call failed "
            f"({type(exc).__name__}). This often indicates advertised.listeners mismatch. "
            "Continuing; ensure your topic exists or auto-create is enabled on broker."
        )
        return False
    finally:
        admin.close()


def producer_worker(
    producer: KafkaProducer,
    video_path: str,
    video_id: str,
    topic: str,
    frame_step: int,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[PRODUCER:{video_id}] Could not open video: {video_path}")
        eos = {"type": "eos", "video_id": video_id}
        producer.send(topic, key=video_id.encode("utf-8"), value=eos)
        producer.flush()
        return

    frame_index = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sent_count = 0
    print(f"[PRODUCER:{video_id}] Started -> {video_path}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % frame_step == 0:
            ok_enc, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok_enc:
                payload = {
                    "type": "frame",
                    "video_id": video_id,
                    "frame_index": frame_index,
                    "video_total_frames": total_frames,
                    "timestamp_ms": int(time.time() * 1000),
                    "frame_b64": base64.b64encode(encoded.tobytes()).decode("ascii"),
                }
                try:
                    producer.send(topic, key=video_id.encode("utf-8"), value=payload).get(timeout=15)
                    sent_count += 1
                except KafkaTimeoutError as exc:
                    print(
                        f"[PRODUCER:{video_id}] Kafka metadata timeout while sending frame {frame_index}. "
                        "Check topic availability and broker advertised.listeners."
                    )
                    break
                except KafkaError as exc:
                    print(f"[PRODUCER:{video_id}] Kafka send error: {exc}")
                    break

        frame_index += 1

    cap.release()
    eos = {"type": "eos", "video_id": video_id}
    try:
        producer.send(topic, key=video_id.encode("utf-8"), value=eos).get(timeout=10)
        producer.flush()
    except KafkaError:
        pass
    print(f"[PRODUCER:{video_id}] Finished. Sent {sent_count} frames.")


def local_producer_worker(
    frame_queue: queue.Queue,
    video_path: str,
    video_id: str,
    frame_step: int,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[LOCAL-PRODUCER:{video_id}] Could not open video: {video_path}")
        frame_queue.put({"type": "eos", "video_id": video_id})
        return

    frame_index = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sent_count = 0
    print(f"[LOCAL-PRODUCER:{video_id}] Started -> {video_path}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % frame_step == 0:
            frame_queue.put(
                {
                    "type": "frame",
                    "video_id": video_id,
                    "frame_index": frame_index,
                    "video_total_frames": total_frames,
                    "timestamp_ms": int(time.time() * 1000),
                    "frame": frame,
                }
            )
            sent_count += 1

        frame_index += 1

    cap.release()
    frame_queue.put({"type": "eos", "video_id": video_id})
    print(f"[LOCAL-PRODUCER:{video_id}] Finished. Sent {sent_count} frames.")


def process_frame(
    frame: np.ndarray,
    video_id: str,
    frame_index: int,
    args: argparse.Namespace,
    model_fire: YOLO,
    model_weapon: YOLO,
    alerts_dir: Path,
    frames_dir: Path,
    out_dir: Path,
    event_counter: dict[str, int],
) -> np.ndarray:
    result_fire = model_fire.predict(source=frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
    result_weapon = model_weapon.predict(source=frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]

    annotated = result_fire.plot()
    annotated = result_weapon.plot(img=annotated)

    detections = []
    for result in (result_fire, result_weapon):
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = str(result.names[cls_id]).lower()
            if cls_name not in ("fire", "weapon"):
                continue
            detections.append((cls_name, conf))

    if detections:
        ts = int(time.time() * 1000)
        frame_name = f"{video_id}_{ts}_{frame_index}.jpg"
        
        # Upload image directly to HDFS via Docker stdin
        hdfs_image_path = f"/cctv/images/{frame_name}"
        ok_enc_full, encoded_full = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok_enc_full:
            try:
                hdfs_client.write(hdfs_image_path, data=encoded_full.tobytes(), overwrite=True)
                final_image_path = f"hdfs://localhost:9000{hdfs_image_path}"
            except Exception as e:
                print(f"Failed to upload image to HDFS: {e}")
                # Fallback path if docker isn't running
                final_image_path = f"hdfs://localhost:9000{hdfs_image_path}"

        for cls_name, conf in detections:
            event_counter[cls_name] = event_counter.get(cls_name, 0) + 1
            # Add a random unique ID to prevent identical filenames for multiple threats in the same frame
            unique_id = uuid.uuid4().hex[:6]
            log_name = f"alert_{video_id}_{ts}_{frame_index}_{cls_name}_{unique_id}.json"
            
            payload = {
                "timestamp": ts,
                "event_type": cls_name,
                "confidence": round(conf, 3),
                "image_path": final_image_path,
                "camera_id": video_id,
                "source": "kafka_video_file" if args.transport in ("kafka", "auto") else "local_video_file",
            }
            
            # Upload the JSON alert log to HDFS directly via WebHDFS (ultra fast)
            hdfs_log_path = f"/cctv/alerts/logs/{log_name}"
            try:
                hdfs_client.write(hdfs_log_path, data=json.dumps(payload).encode("utf-8"), overwrite=True)
            except Exception as e:
                print(f"Failed to upload JSON log to HDFS: {e}")

    # Resize the image significantly to reduce base64 size for fast IPC and fast UI loading
    small_frame = cv2.resize(annotated, (480, 270))
    ok_enc, encoded = cv2.imencode(".jpg", small_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    if ok_enc:
        b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
        print(f"FRAME_B64:{video_id}:{b64}", flush=True)

    if args.save_annotated:
        out_path = out_dir / f"{video_id}_{frame_index}.jpg"
        cv2.imwrite(str(out_path), annotated)

    return annotated


def render_progress_ui(
    frame: np.ndarray,
    title: str,
    video_progress: dict[str, dict[str, int | bool]],
    event_counter: dict[str, int],
) -> np.ndarray:
    panel_width = 430
    h, w = frame.shape[:2]
    panel = np.full((h, panel_width, 3), 24, dtype=np.uint8)

    cv2.putText(panel, title, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
    total_events = sum(event_counter.values())
    fire_count = event_counter.get("fire", 0)
    weapon_count = event_counter.get("weapon", 0)
    cv2.putText(panel, f"Events: {total_events}", (16, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(panel, f"Fire: {fire_count}", (16, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 180, 255), 1)
    cv2.putText(panel, f"Weapon: {weapon_count}", (160, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 255, 120), 1)

    y = 130
    for video_id in sorted(video_progress.keys()):
        meta = video_progress[video_id]
        last_frame = int(meta.get("last_frame", 0))
        total_frames = int(meta.get("total_frames", 0))
        received_frames = int(meta.get("received_frames", 0))
        done = bool(meta.get("done", False))

        if total_frames > 0:
            pct = max(0.0, min(100.0, (last_frame / total_frames) * 100.0))
        else:
            pct = 0.0

        cv2.putText(panel, f"{video_id} [{'DONE' if done else 'RUN'}]", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (240, 240, 240), 1)
        y += 18
        cv2.rectangle(panel, (16, y), (392, y + 16), (70, 70, 70), -1)
        bar_w = int((pct / 100.0) * (392 - 16))
        cv2.rectangle(panel, (16, y), (16 + bar_w, y + 16), (0, 180, 255), -1)
        y += 28
        cv2.putText(
            panel,
            f"{pct:5.1f}%  frame:{last_frame}  received:{received_frames}",
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (200, 200, 200),
            1,
        )
        y += 28

        if y > h - 28:
            break

    return cv2.hconcat([frame, panel])


def run_consumer(
    args: argparse.Namespace,
    model_fire: YOLO,
    model_weapon: YOLO,
    alerts_dir: Path,
    frames_dir: Path,
    out_dir: Path,
    producer_count: int,
    producer_threads: list[threading.Thread] | None = None,
) -> None:
    try:
        consumer = KafkaConsumer(
            args.topic,
            bootstrap_servers=args.bootstrap_servers,
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            group_id=args.group_id,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            consumer_timeout_ms=15000,
        )
    except (NoBrokersAvailable, NodeNotReadyError, KafkaError) as exc:
        raise SystemExit(
            "Kafka consumer could not initialize. Check broker listeners and topic availability."
        ) from exc

    ended_videos: set[str] = set()
    event_counter: dict[str, int] = {}
    video_progress: dict[str, dict[str, int | bool]] = {
        f"video_{i + 1}": {"last_frame": 0, "total_frames": 0, "received_frames": 0, "done": False}
        for i in range(producer_count)
    }
    empty_polls = 0

    print("[CONSUMER] Started detection from Kafka frames...")
    if args.show:
        print("[VIEW] External preview window is enabled. Press 'q' in window to stop.")
    while len(ended_videos) < producer_count:
        received_any = False

        for msg in consumer:
            received_any = True
            data = msg.value
            msg_type = data.get("type")
            video_id = data.get("video_id", "unknown")

            if msg_type == "eos":
                ended_videos.add(video_id)
                if video_id not in video_progress:
                    video_progress[video_id] = {"last_frame": 0, "total_frames": 0, "received_frames": 0, "done": True}
                else:
                    video_progress[video_id]["done"] = True
                print(f"[CONSUMER] EOS received for {video_id} ({len(ended_videos)}/{producer_count})")
                if len(ended_videos) >= producer_count:
                    break
                continue

            if msg_type != "frame":
                continue

            try:
                raw = base64.b64decode(data["frame_b64"])
                np_arr = np.frombuffer(raw, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            except Exception:
                continue

            if frame is None:
                continue

            frame_index = int(data.get("frame_index", -1))
            total_frames = int(data.get("video_total_frames", 0))
            if video_id not in video_progress:
                video_progress[video_id] = {
                    "last_frame": frame_index,
                    "total_frames": total_frames,
                    "received_frames": 1,
                    "done": False,
                }
            else:
                video_progress[video_id]["last_frame"] = frame_index
                if total_frames > 0:
                    video_progress[video_id]["total_frames"] = total_frames
                video_progress[video_id]["received_frames"] = int(video_progress[video_id]["received_frames"]) + 1

            annotated = process_frame(
                frame=frame,
                video_id=video_id,
                frame_index=frame_index,
                args=args,
                model_fire=model_fire,
                model_weapon=model_weapon,
                alerts_dir=alerts_dir,
                frames_dir=frames_dir,
                out_dir=out_dir,
                event_counter=event_counter,
            )

            if args.show:
                view = render_progress_ui(
                    frame=annotated,
                    title="Kafka Parallel Detection",
                    video_progress=video_progress,
                    event_counter=event_counter,
                )
                cv2.imshow("Kafka Multi Video Detection", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    ended_videos = {f"video_{i + 1}" for i in range(producer_count)}
                    break

        if not received_any:
            empty_polls += 1
            print("[CONSUMER] Waiting for frames...")

            if producer_threads and all(not t.is_alive() for t in producer_threads) and empty_polls >= 3:
                print("[CONSUMER] Producers finished but no more Kafka frames arrived. Stopping consumer.")
                break

            if empty_polls >= 30:
                print("[CONSUMER] Kafka idle timeout reached. Stopping consumer.")
                break
        else:
            empty_polls = 0

    consumer.close()
    cv2.destroyAllWindows()

    print("\n[SUMMARY] Detection counts")
    if not event_counter:
        print("No fire/weapon events detected.")
    else:
        for k, v in event_counter.items():
            print(f"- {k}: {v}")


def run_local_pipeline(
    args: argparse.Namespace,
    model_fire: YOLO,
    model_weapon: YOLO,
    alerts_dir: Path,
    frames_dir: Path,
    out_dir: Path,
) -> None:
    frame_queues = [queue.Queue(maxsize=10) for _ in args.videos]

    producers = []
    for i, video_path in enumerate(args.videos):
        video_id = f"video_{i + 1}"
        t = threading.Thread(
            target=local_producer_worker,
            args=(frame_queues[i], video_path, video_id, args.frame_step),
            daemon=True,
        )
        producers.append(t)
        t.start()

    ended_videos: set[str] = set()
    event_counter: dict[str, int] = {}
    video_progress: dict[str, dict[str, int | bool]] = {
        f"video_{i + 1}": {"last_frame": 0, "total_frames": 0, "received_frames": 0, "done": False}
        for i in range(len(producers))
    }

    print("[LOCAL] Started detection from local in-memory queue.")
    if args.show:
        print("[VIEW] External preview window is enabled. Press 'q' in window to stop.")

    while len(ended_videos) < len(producers):
        received_any = False
        for i, q in enumerate(frame_queues):
            video_id = f"video_{i + 1}"
            if video_id in ended_videos:
                continue

            try:
                item = q.get_nowait()
            except queue.Empty:
                continue

            received_any = True
            msg_type = item.get("type")

            if msg_type == "eos":
                ended_videos.add(video_id)
                if video_id in video_progress:
                    video_progress[video_id]["done"] = True
                continue

            if msg_type != "frame":
                continue

            frame = item.get("frame")
            if frame is None:
                continue

            frame_index = int(item.get("frame_index", -1))
            total_frames = int(item.get("video_total_frames", 0))
            if video_id in video_progress:
                video_progress[video_id]["last_frame"] = frame_index
                if total_frames > 0:
                    video_progress[video_id]["total_frames"] = total_frames
                video_progress[video_id]["received_frames"] = int(video_progress[video_id]["received_frames"]) + 1

            annotated = process_frame(
                frame=frame,
                video_id=video_id,
                frame_index=frame_index,
                args=args,
                model_fire=model_fire,
                model_weapon=model_weapon,
                alerts_dir=alerts_dir,
                frames_dir=frames_dir,
                out_dir=out_dir,
                event_counter=event_counter,
            )

            if args.show:
                view = render_progress_ui(
                    frame=annotated,
                    title="Local Parallel Detection",
                    video_progress=video_progress,
                    event_counter=event_counter,
                )
                cv2.imshow("Parallel Video Detection (Local Fallback)", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    ended_videos = {f"video_{idx + 1}" for idx in range(len(producers))}
                    break

        if not received_any:
            if all(not t.is_alive() for t in producers):
                break
            time.sleep(0.01)

    for t in producers:
        t.join()

    cv2.destroyAllWindows()

    print("\n[SUMMARY] Detection counts")
    if not event_counter:
        print("No fire/weapon events detected.")
    else:
        for k, v in event_counter.items():
            print(f"- {k}: {v}")


def main() -> None:
    args = parse_args()

    if len(args.videos) < 1:
        raise SystemExit("Please provide at least 1 video in --videos.")
    if args.transport in ("kafka", "auto") and len(args.videos) < 2:
        raise SystemExit("Please provide at least 2 videos in --videos when using Kafka transport.")

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent if script_dir.name == "scripts" else script_dir

    default_fire, default_weapon = find_default_model_paths(project_root)
    fire_model_path = Path(args.fire_model) if args.fire_model else default_fire
    weapon_model_path = Path(args.weapon_model) if args.weapon_model else default_weapon

    if not fire_model_path.exists():
        raise SystemExit(f"Fire model not found: {fire_model_path}")
    if not weapon_model_path.exists():
        raise SystemExit(f"Weapon model not found: {weapon_model_path}")

    for video in args.videos:
        if not Path(video).exists():
            raise SystemExit(f"Video file not found: {video}")

    alerts_dir, frames_dir, out_dir = ensure_dirs(project_root)

    print(f"Loading fire model: {fire_model_path}")
    model_fire = YOLO(str(fire_model_path))
    print(f"Loading weapon model: {weapon_model_path}")
    model_weapon = YOLO(str(weapon_model_path))

    if args.transport == "local":
        run_local_pipeline(
            args=args,
            model_fire=model_fire,
            model_weapon=model_weapon,
            alerts_dir=alerts_dir,
            frames_dir=frames_dir,
            out_dir=out_dir,
        )
        print("Processing complete.")
        return

    if args.transport in ("auto", "kafka"):
        assert_kafka_reachable(args.bootstrap_servers)
        topic_ready = ensure_topic_ready(args.bootstrap_servers, args.topic)

        if not topic_ready and args.transport == "auto":
            print("[FALLBACK] Kafka metadata not ready. Switching to local queue mode.")
            run_local_pipeline(
                args=args,
                model_fire=model_fire,
                model_weapon=model_weapon,
                alerts_dir=alerts_dir,
                frames_dir=frames_dir,
                out_dir=out_dir,
            )
            print("Processing complete.")
            return

        try:
            producer = KafkaProducer(
                bootstrap_servers=args.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                linger_ms=10,
                request_timeout_ms=20000,
                api_version_auto_timeout_ms=10000,
            )
        except NoBrokersAvailable as exc:
            if args.transport == "kafka":
                raise SystemExit(
                    "Kafka producer could not connect to any broker. "
                    f"Check --bootstrap-servers ({args.bootstrap_servers}) and ensure Kafka is running."
                ) from exc

            print("[FALLBACK] Kafka producer unavailable. Switching to local queue mode.")
            run_local_pipeline(
                args=args,
                model_fire=model_fire,
                model_weapon=model_weapon,
                alerts_dir=alerts_dir,
                frames_dir=frames_dir,
                out_dir=out_dir,
            )
            print("Processing complete.")
            return

        producer_threads = []
        for i, video_path in enumerate(args.videos, start=1):
            video_id = f"video_{i}"
            t = threading.Thread(
                target=producer_worker,
                args=(producer, video_path, video_id, args.topic, args.frame_step),
                daemon=True,
            )
            producer_threads.append(t)
            t.start()

        run_consumer(
            args=args,
            model_fire=model_fire,
            model_weapon=model_weapon,
            alerts_dir=alerts_dir,
            frames_dir=frames_dir,
            out_dir=out_dir,
            producer_count=len(producer_threads),
            producer_threads=producer_threads,
        )

        for t in producer_threads:
            t.join()

        producer.close()
        print("Processing complete.")
        return


if __name__ == "__main__":
    main()
