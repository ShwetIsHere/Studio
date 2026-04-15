import cv2
import argparse
import time
import json
import os
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Fire and Weapon Detection with Real-time Logging")
    parser.add_argument(
        "source",
        type=str,
        help="Path to the input video file",
    )
    parser.add_argument("--output", type=str, default="output_detection.mp4", help="Path to save the processed video")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--no-show", action="store_true", help="Do not show the video window during processing")
    return parser.parse_args()

def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Missing dependency. Install with: pip install ultralytics opencv-python")
        return

    args = parse_args()

    # Directory Setup for Spark Pipeline
    LOG_DIR = "alerts/logs/"
    FRAME_DIR = "alerts/frames/"
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(FRAME_DIR, exist_ok=True)

    # Paths as requested by user
    FIRE_MODEL_PATH = r"runs\train\fire_yolo11\weights\fire_best.pt"
    WEAPON_MODEL_PATH = r"runs\train\fire_yolo11\weights\weapon_best.pt"

    # Load models
    print(f"Loading fire model from: {FIRE_MODEL_PATH}")
    model_fire = YOLO(FIRE_MODEL_PATH)
    print(f"Loading weapon model from: {WEAPON_MODEL_PATH}")
    model_weapon = YOLO(WEAPON_MODEL_PATH)

    # Summary counters
    summary = {}

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print(f"Processing video: {args.source}")
    print(f"Saving alerts to: {LOG_DIR}")
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results_fire = model_fire.predict(source=frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
        results_weapon = model_weapon.predict(source=frame, conf=args.conf, imgsz=args.imgsz, verbose=False)

        # Plot detections
        annotated_frame = results_fire[0].plot()
        annotated_frame = results_weapon[0].plot(img=annotated_frame)

        # Log detections to JSON for Apache Spark
        current_time = int(time.time() * 1000) # Epoch milliseconds
        has_detection = False

        for res in [results_fire[0], results_weapon[0]]:
            if res.boxes is not None and len(res.boxes) > 0:
                has_detection = True
                for box in res.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = res.names[cls_id]

                    # 1. Update project summary
                    summary[cls_name] = summary.get(cls_name, 0) + 1

                    # 2. Create alert log
                    frame_filename = f"frame_{current_time}_{frame_count}.jpg"
                    frame_path = os.path.join(FRAME_DIR, frame_filename)
                    
                    alert_data = {
                        "timestamp": current_time,
                        "event_type": cls_name,
                        "confidence": round(conf, 2),
                        "image_path": os.path.abspath(frame_path),
                        "camera_id": "CAM_01"
                    }

                    # 3. Save JSON log file
                    log_filename = f"alert_{current_time}_{frame_count}_{cls_name}.json"
                    with open(os.path.join(LOG_DIR, log_filename), 'w') as f:
                        json.dump(alert_data, f)
                    
                    # 4. Save the detection frame
                    cv2.imwrite(frame_path, annotated_frame)

        # Write output video
        out.write(annotated_frame)

        frame_count += 1
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            print(f"Progress: {progress:.1f}% | Total detections so far: {sum(summary.values())}")

        if not args.no_show:
            cv2.imshow("CCTV Security System - Processing", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("\nProcessing Complete.")
    print("Detection Summary:")
    for cls, count in summary.items():
        print(f" - {cls}: {count}")
    print(f"\nFinal video saved: {args.output}")
    print(f"All alert logs are in: {LOG_DIR}")

if __name__ == "__main__":
    main()
