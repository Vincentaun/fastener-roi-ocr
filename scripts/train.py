"""
Step 3: Train YOLOv8 on the converted dataset.

Why these defaults:
- Tiny model (`yolov8n.pt`) keeps CPU training tractable.
- `imgsz=1280` is large because the boxes (dimension callouts) are very
  small relative to the page; downsampling to the YOLO default of 640
  shrinks them to ~25 px and hurts detection.
- Heavy mosaic/translate/scale augmentation compensates for tiny data.
- Long `patience` because a small dataset trains noisily.

Usage:
    python train.py --data data/screws/data.yaml [--epochs 100] [--imgsz 1280] [--model yolov8n.pt]
"""
from __future__ import annotations
import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser(description="Train YOLOv8 for screw spec ROI detection.")
    ap.add_argument("--data", required=True, type=Path, help="Path to data.yaml")
    ap.add_argument("--model", default="yolov8n.pt",
                    help="Pretrained model: yolov8n.pt (fast), yolov8s.pt, yolov8m.pt …")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="Training image size. Larger helps small-object detection.")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--project", type=Path, default=Path("runs/screw_roi"))
    ap.add_argument("--name", default="exp")
    ap.add_argument("--device", default="cpu", help="'cpu' or '0' for first GPU")
    args = ap.parse_args()

    model = YOLO(args.model)
    print(f"Training {args.model} on {args.data} for {args.epochs} epochs at imgsz={args.imgsz} ...")

    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        # Augmentation tuned for small dataset + small objects
        mosaic=1.0,
        mixup=0.0,
        translate=0.1,
        scale=0.4,
        fliplr=0.0,   # screws/text are NOT horizontally symmetric -> never flip
        flipud=0.0,
        hsv_h=0.0, hsv_s=0.2, hsv_v=0.3,
        degrees=0.0,
        # Training schedule
        patience=50,
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        warmup_epochs=3,
        # Output
        plots=True,
        verbose=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nDone.  Best weights: {best}")


if __name__ == "__main__":
    main()
