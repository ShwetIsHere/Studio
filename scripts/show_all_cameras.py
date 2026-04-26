import argparse
import cv2
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show laptop webcam and mobile IP Webcam stream together"
    )
    parser.add_argument(
        "--laptop-index",
        type=int,
        default=0,
        help="Laptop camera index (default: 0)",
    )
    parser.add_argument(
        "--mobile-url",
        type=str,
        default="http://10.74.143.72:8080/",
        help="Base URL of IP Webcam app (default: http://10.74.143.72:8080/)",
    )
    parser.add_argument(
        "--mobile-path",
        type=str,
        default="auto",
        choices=["auto", "video", "videofeed", "mjpegfeed", "shot.jpg"],
        help="Mobile stream path. Use auto to try common endpoints.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Display width per camera frame",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Display height per camera frame",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detection",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size",
    )
    return parser.parse_args()


def normalize_url(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    if path == "":
        return f"{base}/"
    return f"{base}/{path}"


def open_mobile_capture(base_url: str, mode: str) -> tuple[cv2.VideoCapture, str]:
    # IP Webcam apps vary by endpoint, so we try several common ones.
    if mode == "auto":
        candidates = ["video", "videofeed", "mjpegfeed", "shot.jpg", ""]
    else:
        candidates = [mode]

    for path in candidates:
        url = normalize_url(base_url, path)
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                return cap, url
            cap.release()

    return cv2.VideoCapture(), ""


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    fire_model_path = project_root / "models" / "runs" / "train" / "fire_yolo11" / "weights" / "fire_best.pt"
    weapon_model_path = project_root / "models" / "runs" / "train" / "fire_yolo11" / "weights" / "weapon_best.pt"

    if not fire_model_path.exists():
        raise SystemExit(f"Fire model not found: {fire_model_path}")
    if not weapon_model_path.exists():
        raise SystemExit(f"Weapon model not found: {weapon_model_path}")

    print(f"Loading fire model: {fire_model_path}")
    fire_model = YOLO(str(fire_model_path))
    print(f"Loading weapon model: {weapon_model_path}")
    weapon_model = YOLO(str(weapon_model_path))

    laptop_cap = cv2.VideoCapture(args.laptop_index)
    if not laptop_cap.isOpened():
        raise SystemExit(
            f"Could not open laptop camera at index {args.laptop_index}. "
            "Try index 1 or close other camera-using apps."
        )

    mobile_cap, active_mobile_url = open_mobile_capture(args.mobile_url, args.mobile_path)
    if not mobile_cap.isOpened():
        laptop_cap.release()
        raise SystemExit(
            "Could not open mobile stream. Check phone and laptop are on same Wi-Fi, "
            f"and verify URL in browser: {args.mobile_url}"
        )

    print(f"Laptop camera opened at index: {args.laptop_index}")
    print(f"Mobile stream opened at: {active_mobile_url}")
    print("Press 'q' to quit")

    while True:
        ok_laptop, frame_laptop = laptop_cap.read()
        ok_mobile, frame_mobile = mobile_cap.read()

        if not ok_laptop:
            print("Laptop camera frame read failed.")
            break
        if not ok_mobile:
            print("Mobile camera frame read failed.")
            break

        # Run fire+weapon detection on each stream and overlay boxes.
        laptop_fire = fire_model.predict(source=frame_laptop, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        laptop_weapon = weapon_model.predict(source=frame_laptop, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        laptop_annotated = laptop_fire.plot()
        laptop_annotated = laptop_weapon.plot(img=laptop_annotated)

        mobile_fire = fire_model.predict(source=frame_mobile, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        mobile_weapon = weapon_model.predict(source=frame_mobile, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        mobile_annotated = mobile_fire.plot()
        mobile_annotated = mobile_weapon.plot(img=mobile_annotated)

        frame_laptop = cv2.resize(laptop_annotated, (args.width, args.height))
        frame_mobile = cv2.resize(mobile_annotated, (args.width, args.height))

        cv2.putText(
            frame_laptop,
            "Laptop Camera",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame_mobile,
            "Mobile Camera",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

        combined = cv2.hconcat([frame_laptop, frame_mobile])
        cv2.imshow("All Cameras - Laptop + Mobile", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    laptop_cap.release()
    mobile_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()