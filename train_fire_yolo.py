import argparse
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fire detection model with GPU on YOLO dataset")
    parser.add_argument(
        "--data",
        type=str,
        default="Fire Detection.v1i.yolov11/data.yaml",
        help="Path to data.yaml",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Base YOLO model checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument(
        "--batch",
        type=float,
        default=0.9,
        help=(
            "Batch size. Use integer (e.g., 16) or fraction (0.0-1.0) for auto-batch VRAM target. "
            "Default 0.9 aims to use about 90% GPU memory."
        ),
    )
    parser.add_argument("--workers", type=int, default=8, help="Data loader workers")
    parser.add_argument("--project", type=str, default="runs/train", help="Output directory")
    parser.add_argument("--name", type=str, default="fire_yolo11", help="Run name")
    return parser.parse_args()


def build_usable_data_yaml(data_yaml: Path) -> Path:
    import yaml

    def resolve_split_path(dataset_root: Path, split_value: str) -> Path:
        candidates = [
            (dataset_root / split_value).resolve(),
            (dataset_root / split_value.lstrip("./")).resolve(),
            (dataset_root / split_value.lstrip("../")).resolve(),
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    dataset_root = data_yaml.parent
    with data_yaml.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    train_rel = data_cfg.get("train", "train/images")
    val_rel = data_cfg.get("val", "valid/images")
    test_rel = data_cfg.get("test", "test/images")

    train_abs = resolve_split_path(dataset_root, train_rel)
    val_abs = resolve_split_path(dataset_root, val_rel)
    test_abs = resolve_split_path(dataset_root, test_rel)

    if not train_abs.exists():
        raise SystemExit(f"Train images folder not found: {train_abs}")

    # Some exports may not include valid/test splits; fallback to train split so training can run.
    if not val_abs.exists():
        print(f"Warning: val path missing ({val_abs}). Using train images for val.")
        val_abs = train_abs
    if not test_abs.exists():
        print(f"Warning: test path missing ({test_abs}). Using train images for test.")
        test_abs = train_abs

    fixed_cfg = {
        "path": str(dataset_root),
        "train": str(train_abs),
        "val": str(val_abs),
        "test": str(test_abs),
        "nc": data_cfg.get("nc", 0),
        "names": data_cfg.get("names", []),
    }

    temp_file = Path(tempfile.gettempdir()) / "fire_data_autofix.yaml"
    with temp_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(fixed_cfg, f, sort_keys=False)

    print(f"Using data config: {temp_file}")
    return temp_file


def main() -> None:
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install with: pip install ultralytics torch torchvision"
        ) from exc

    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not detected. Please run on a machine with NVIDIA GPU + CUDA.")

    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Using GPU: {gpu_name} ({total_vram_gb:.2f} GB VRAM)")

    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        raise SystemExit(f"data.yaml not found: {data_yaml}")

    train_data_yaml = build_usable_data_yaml(data_yaml)

    model = YOLO(args.model)

    # Fractional batch lets Ultralytics auto-select the highest safe batch for target VRAM usage.
    batch_arg = int(args.batch) if args.batch >= 1 else float(args.batch)
    if isinstance(batch_arg, float):
        target_gb = batch_arg * total_vram_gb
        print(f"Auto-batch target: {batch_arg:.0%} VRAM (~{target_gb:.2f} GB)")
    else:
        print(f"Fixed batch size: {batch_arg}")

    results = model.train(
        data=str(train_data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch_arg,
        workers=args.workers,
        device=0,
        amp=True,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    print("Training complete.")
    print(results)


if __name__ == "__main__":
    main()
