from pathlib import Path

import yaml
from ultralytics import YOLO

def main() -> None:
    # Use the already-downloaded local dataset in this folder.
    base_dir = Path(__file__).resolve().parent
    train_img = base_dir / "images" / "train"
    train_lbl = base_dir / "labels" / "train"
    val_img = base_dir / "images" / "val"
    val_lbl = base_dir / "labels" / "val"

    required_paths = [train_img, train_lbl, val_img, val_lbl]
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing dataset folders:\n" + "\n".join(missing)
        )

    # Create/update YOLO data config.
    yaml_path = base_dir / "weapon_data.yaml"
    data_config = {
        "path": str(base_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["Weapon"],
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_config, f, sort_keys=False)

    print("Local dataset detected. Starting training...")

    model = YOLO("yolo11n.pt")
    model.train(
        data=str(yaml_path),
        epochs=40,
        imgsz=640,
        batch=32,
        device=0,
        workers=8,
        plots=True,
        name="cctv_weapon_v2",
    )


if __name__ == "__main__":
    main()

