from __future__ import annotations

import cgi
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
PIPELINE_SCRIPT = SCRIPTS_DIR / "kafka_parallel_video_detection.py"
ALERTS_DIR = PROJECT_ROOT / "alerts" / "logs"
FRAMES_DIR = PROJECT_ROOT / "alerts" / "frames"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
SKIP_DIRS = {".git", "frontend", "node_modules"}

RUN_STATE_LOCK = threading.Lock()
RUN_STATE: dict[str, object] = {
    "running": False,
    "start_ms": 0,
    "finish_ms": 0,
    "exit_code": None,
    "videos": [],
    "command": [],
    "stdout_tail": [],
    "stderr_tail": [],
}


def _append_tail(key: str, line: str, max_lines: int = 160) -> None:
    with RUN_STATE_LOCK:
        arr = list(RUN_STATE.get(key, []))
        arr.append(line.rstrip("\n"))
        if len(arr) > max_lines:
            arr = arr[-max_lines:]
        RUN_STATE[key] = arr


def _iter_local_videos(limit: int = 40) -> list[str]:
    videos: list[tuple[float, str]] = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            suffix = Path(name).suffix.lower()
            if suffix not in VIDEO_EXTENSIONS:
                continue
            full = Path(root) / name
            try:
                mtime = full.stat().st_mtime
            except OSError:
                continue
            videos.append((mtime, str(full.resolve())))

    videos.sort(key=lambda x: x[0], reverse=True)
    return [path for _, path in videos[:limit]]


def _collect_alerts_since(start_ms: int) -> list[dict]:
    alerts: list[dict] = []
    if start_ms <= 0 or not ALERTS_DIR.exists():
        return alerts

    for path in ALERTS_DIR.glob("alert_*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                item = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        try:
            timestamp = int(item.get("timestamp", 0))
        except (TypeError, ValueError):
            continue

        source = str(item.get("source", ""))
        if timestamp >= start_ms and source == "local_video_file":
            alerts.append(item)

    return alerts


def _collect_latest_frames(start_ms: int, limit: int = 8) -> list[dict[str, object]]:
    if not FRAMES_DIR.exists():
        return []

    candidates: list[tuple[float, Path]] = []
    for path in FRAMES_DIR.glob("*.jpg"):
        try:
            ts = path.stat().st_mtime * 1000.0
        except OSError:
            continue
        if start_ms > 0 and ts < start_ms:
            continue
        candidates.append((ts, path))

    candidates.sort(key=lambda x: x[0], reverse=True)
    frames: list[dict[str, object]] = []
    for ts, path in candidates[:limit]:
        frames.append(
            {
                "name": path.name,
                "timestamp": int(ts),
                "url": f"/api/frames/{path.name}",
            }
        )
    return frames


def _start_parallel_local_job(videos: list[str], cleanup_dir: Path | None = None) -> tuple[bool, str]:
    with RUN_STATE_LOCK:
        if bool(RUN_STATE.get("running", False)):
            return False, "A parallel run is already in progress."

        start_ms = int(time.time() * 1000)
        cmd = [
            sys.executable,
            str(PIPELINE_SCRIPT),
            "--videos",
            *videos,
            "--transport",
            "local",
            "--no-show",
        ]

        RUN_STATE["running"] = True
        RUN_STATE["start_ms"] = start_ms
        RUN_STATE["finish_ms"] = 0
        RUN_STATE["exit_code"] = None
        RUN_STATE["videos"] = videos
        RUN_STATE["command"] = cmd
        RUN_STATE["stdout_tail"] = []
        RUN_STATE["stderr_tail"] = []

    def _runner() -> None:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        def _read_stream(stream, key: str) -> None:
            if stream is None:
                return
            for line in stream:
                line_str = line.strip()
                if line_str.startswith("FRAME_B64:"):
                    parts = line_str.split(":", 2)
                    if len(parts) == 3:
                        vid = parts[1]
                        b64 = parts[2]
                        with RUN_STATE_LOCK:
                            if "live_frames" not in RUN_STATE:
                                RUN_STATE["live_frames"] = {}
                            RUN_STATE["live_frames"][vid] = b64
                else:
                    _append_tail(key, line)
            stream.close()

        t_out = threading.Thread(target=_read_stream, args=(proc.stdout, "stdout_tail"), daemon=True)
        t_err = threading.Thread(target=_read_stream, args=(proc.stderr, "stderr_tail"), daemon=True)
        t_out.start()
        t_err.start()
        exit_code = proc.wait()
        t_out.join(timeout=2)
        t_err.join(timeout=2)

        try:
            with RUN_STATE_LOCK:
                RUN_STATE["running"] = False
                RUN_STATE["finish_ms"] = int(time.time() * 1000)
                RUN_STATE["exit_code"] = exit_code
        finally:
            if cleanup_dir is not None and cleanup_dir.exists():
                try:
                    shutil.rmtree(cleanup_dir, ignore_errors=True)
                except OSError:
                    pass

    threading.Thread(target=_runner, daemon=True).start()
    return True, "Parallel local run started."


def _start_parallel_uploaded_job(uploaded_files: list[cgi.FieldStorage]) -> tuple[bool, str, list[str]]:
    if not uploaded_files:
        return False, "No uploaded videos provided.", []

    run_dir = PROJECT_ROOT / "output" / "uploaded_parallel_runs" / f"run_{int(time.time() * 1000)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    video_paths: list[str] = []
    for idx, field in enumerate(uploaded_files, start=1):
        original_name = Path(field.filename or f"video_{idx}.mp4").name
        suffix = Path(original_name).suffix.lower() or ".mp4"
        if suffix not in VIDEO_EXTENSIONS:
            suffix = ".mp4"

        target = run_dir / f"video_{idx}{suffix}"
        try:
            content = field.file.read()
            with open(target, "wb") as f:
                f.write(content)
        except OSError:
            shutil.rmtree(run_dir, ignore_errors=True)
            return False, f"Failed to save uploaded video: {original_name}", []

        video_paths.append(str(target.resolve()))

    ok, message = _start_parallel_local_job(video_paths, cleanup_dir=run_dir)
    if not ok:
        shutil.rmtree(run_dir, ignore_errors=True)
        return False, message, []

    return True, message, video_paths


class BackendHandler(BaseHTTPRequestHandler):
    def _set_headers(self, status_code: int = 200) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def _write_json(self, payload: dict, status_code: int = 200) -> None:
        self._set_headers(status_code)
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def _write_binary(self, payload: bytes, content_type: str, status_code: int = 200) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._set_headers(204)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        route = parsed.path
        query = parse_qs(parsed.query)

        if route == "/api/health":
            self._write_json({"success": True, "message": "Backend server is running"})
            return

        if route == "/api/local-videos":
            videos = _iter_local_videos(limit=60)
            self._write_json({"success": True, "videos": videos, "count": len(videos)})
            return

        if route == "/api/run-status":
            with RUN_STATE_LOCK:
                state = {
                    "running": bool(RUN_STATE.get("running", False)),
                    "startMs": int(RUN_STATE.get("start_ms", 0) or 0),
                    "finishMs": int(RUN_STATE.get("finish_ms", 0) or 0),
                    "exitCode": RUN_STATE.get("exit_code"),
                    "videos": list(RUN_STATE.get("videos", [])),
                    "stdoutTail": list(RUN_STATE.get("stdout_tail", []))[-20:],
                    "stderrTail": list(RUN_STATE.get("stderr_tail", []))[-20:],
                }
                live_frames = dict(RUN_STATE.get("live_frames", {}))

            alerts = _collect_alerts_since(state["startMs"])
            counts = Counter(str(a.get("event_type", "unknown")).lower() for a in alerts)
            summary = {k: int(v) for k, v in counts.items() if k in ("fire", "weapon")}
            latest_frames = _collect_latest_frames(state["startMs"], limit=8)
            self._write_json(
                {
                    "success": True,
                    "state": state,
                    "summary": summary,
                    "latestFrames": latest_frames,
                    "totalAlerts": int(sum(summary.values())),
                    "liveFrames": live_frames,
                }
            )
            return

        if route == "/api/latest-frames":
            limit = 8
            if "limit" in query:
                try:
                    limit = max(1, min(50, int(query["limit"][0])))
                except (TypeError, ValueError):
                    limit = 8
            with RUN_STATE_LOCK:
                start_ms = int(RUN_STATE.get("start_ms", 0) or 0)
            self._write_json({"success": True, "frames": _collect_latest_frames(start_ms, limit=limit)})
            return

        if route.startswith("/api/frames/"):
            filename = unquote(route.replace("/api/frames/", "", 1)).strip()
            if not filename or "/" in filename or "\\" in filename:
                self._write_json({"success": False, "error": "Invalid frame filename."}, 400)
                return

            frame_path = FRAMES_DIR / filename
            try:
                if not frame_path.exists() or not frame_path.is_file():
                    self._write_json({"success": False, "error": "Frame not found."}, 404)
                    return
                data = frame_path.read_bytes()
            except OSError:
                self._write_json({"success": False, "error": "Could not read frame."}, 500)
                return

            content_type, _ = mimetypes.guess_type(str(frame_path))
            self._write_binary(data, content_type or "image/jpeg")
            return

        self._write_json({"success": False, "error": "Not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/run-uploaded-parallel":
            if not PIPELINE_SCRIPT.exists():
                self._write_json(
                    {"success": False, "error": f"Pipeline script not found: {PIPELINE_SCRIPT}"},
                    500,
                )
                return

            ctype, _ = cgi.parse_header(self.headers.get("Content-Type", ""))
            if ctype != "multipart/form-data":
                self._write_json(
                    {"success": False, "error": "Content-Type must be multipart/form-data"},
                    400,
                )
                return

            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers["Content-Type"]},
            )

            if "videos" not in form:
                self._write_json(
                    {"success": False, "error": "No videos uploaded. Use form-data key 'videos'."},
                    400,
                )
                return

            raw_fields = form["videos"]
            video_fields = raw_fields if isinstance(raw_fields, list) else [raw_fields]
            valid_fields = [v for v in video_fields if getattr(v, "filename", None)]

            if len(valid_fields) != 4:
                self._write_json(
                    {
                        "success": False,
                        "error": "Please upload exactly 4 videos before starting detection.",
                    },
                    400,
                )
                return

            ok, message, video_paths = _start_parallel_uploaded_job(valid_fields)
            code = 200 if ok else 409
            self._write_json(
                {
                    "success": ok,
                    "message": message,
                    "videoCount": len(video_paths),
                    "videos": video_paths,
                },
                code,
            )
            return

        if self.path == "/api/run-local-parallel":
            if not PIPELINE_SCRIPT.exists():
                self._write_json(
                    {"success": False, "error": f"Pipeline script not found: {PIPELINE_SCRIPT}"},
                    500,
                )
                return

            payload: dict = {}
            content_length = int(self.headers.get("Content-Length", "0") or 0)
            if content_length > 0:
                try:
                    body = self.rfile.read(content_length)
                    payload = json.loads(body.decode("utf-8")) if body else {}
                except (UnicodeDecodeError, json.JSONDecodeError):
                    self._write_json({"success": False, "error": "Invalid JSON body."}, 400)
                    return

            videos = payload.get("videos")
            if not isinstance(videos, list) or not videos:
                videos = _iter_local_videos(limit=20)

            normalized: list[str] = []
            for p in videos:
                path = Path(str(p)).expanduser()
                if not path.is_absolute():
                    path = (PROJECT_ROOT / path).resolve()
                if path.exists() and path.is_file():
                    normalized.append(str(path))

            if not normalized:
                self._write_json(
                    {
                        "success": False,
                        "error": "No valid local videos found. Put videos in project folders and retry.",
                    },
                    400,
                )
                return

            ok, msg = _start_parallel_local_job(normalized)
            code = 200 if ok else 409
            self._write_json(
                {
                    "success": ok,
                    "message": msg,
                    "videoCount": len(normalized),
                    "videos": normalized,
                },
                code,
            )
            return

        if self.path != "/api/analyze":
            self._write_json({"success": False, "error": "Not found"}, 404)
            return

        if not PIPELINE_SCRIPT.exists():
            self._write_json(
                {"success": False, "error": f"Pipeline script not found: {PIPELINE_SCRIPT}"},
                500,
            )
            return

        ctype, _ = cgi.parse_header(self.headers.get("Content-Type", ""))
        if ctype != "multipart/form-data":
            self._write_json(
                {"success": False, "error": "Content-Type must be multipart/form-data"},
                400,
            )
            return

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers["Content-Type"]},
        )

        if "file" not in form:
            self._write_json(
                {"success": False, "error": "No file uploaded. Use form-data key 'file'."},
                400,
            )
            return

        upload_field = form["file"]
        filename = Path(upload_field.filename or "uploaded.mp4")
        suffix = filename.suffix or ".mp4"

        start_ms = int(time.time() * 1000)

        with tempfile.TemporaryDirectory(prefix="cctv_upload_") as tmp_dir:
            input_path = Path(tmp_dir) / f"input{suffix}"
            with open(input_path, "wb") as f:
                f.write(upload_field.file.read())

            cmd = [
                sys.executable,
                str(PIPELINE_SCRIPT),
                "--videos",
                str(input_path),
                "--transport",
                "local",
                "--no-show",
            ]

            proc = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
            )

        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            self._write_json(
                {
                    "success": False,
                    "error": stderr or stdout or "Video analysis failed.",
                    "exitCode": proc.returncode,
                },
                500,
            )
            return

        recent_alerts = []
        for item in _collect_alerts_since(start_ms):
            recent_alerts.append(item)

        counts = Counter(str(a.get("event_type", "unknown")).lower() for a in recent_alerts)
        filtered_counts = {k: v for k, v in counts.items() if k in ("fire", "weapon")}

        max_conf = 0.0
        for a in recent_alerts:
            try:
                max_conf = max(max_conf, float(a.get("confidence", 0.0)))
            except (TypeError, ValueError):
                pass

        threats = [k.upper() for k in sorted(filtered_counts.keys())]

        self._write_json(
            {
                "success": True,
                "threats": threats,
                "confidence": round(max_conf * 100.0, 2),
                "timestamp": int(time.time() * 1000),
                "summary": filtered_counts,
                "stdout": proc.stdout,
            }
        )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "3001"))
    server = ThreadingHTTPServer(("0.0.0.0", port), BackendHandler)
    print(f"Backend server listening on http://localhost:{port}")
    server.serve_forever()
