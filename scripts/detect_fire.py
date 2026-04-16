import argparse
from pathlib import Path

MODEL_PATH = r"C:\Users\shwet\Downloads\fire\runs\train\fire_yolo11\weights\best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect fire using trained YOLO best.pt")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input source: image path, video path, folder path, URL, or 0 for webcam",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--save", action="store_true", help="Save detection outputs")
    parser.add_argument("--show", action="store_true", help="Show live detection window for non-webcam sources")
    return parser.parse_args()


def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Missing dependency. Install with: pip install ultralytics") from exc

    model_file = Path(MODEL_PATH)
    if not model_file.exists():
        raise SystemExit(f"Model file not found: {model_file}")

    args = parse_args()

    source = 0 if args.source == "0" else args.source
    is_webcam = source == 0

    model = YOLO(str(model_file))

    # Webcam runs in streaming mode for realtime detection; press 'q' on the window to quit.
    results = model.predict(
        source=source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=0,
        save=args.save,
        show=True if is_webcam else args.show,
        stream=is_webcam,
        verbose=True,
    )

    names = model.names
    fire_count = 0
    other_count = 0
    smoke_count = 0

    for result in results:
        if result.boxes is None:
            continue
        cls_ids = result.boxes.cls.tolist()
        for cls_id in cls_ids:
            cls_name = names[int(cls_id)]
            if cls_name == "fire":
                fire_count += 1
            elif cls_name == "other":
                other_count += 1
            elif cls_name == "smoke":
                smoke_count += 1

    if is_webcam:
        print("Realtime camera session ended.")
    print("Detection summary:")
    print(f"fire: {fire_count}")
    print(f"other: {other_count}")
    print(f"smoke: {smoke_count}")


if __name__ == "__main__":
    main()
