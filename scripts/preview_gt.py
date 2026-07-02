"""
Preview helper: visualize CVAT ground-truth boxes on a PDF using the same
rendering code as `infer.py`. Useful for:
  - confirming the rendering pipeline & box geometry are correct
  - showing reviewers what an inference output will look like

Usage:
    python preview_gt.py --cvat_xml annotations.xml --pdf TRAINING.pdf --out preview.pdf [--dpi 200]
"""
from __future__ import annotations
import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from infer import draw_boxes_on_pdf  # reuse the inference renderer
from cvat_to_yolo import parse_cvat_xml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cvat_xml", required=True, type=Path)
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    label_names, images = parse_cvat_xml(args.cvat_xml)
    per_image_results = []
    for ann in images:
        boxes = [(b.label, b.xtl, b.ytl, b.xbr, b.ybr, 1.0) for b in ann.boxes]
        per_image_results.append({
            "name": ann.name, "width": ann.width, "height": ann.height, "boxes": boxes,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    draw_boxes_on_pdf(args.pdf, args.out, per_image_results, label_names, args.dpi)
    n_boxes = sum(len(r["boxes"]) for r in per_image_results)
    print(f"Drew {n_boxes} GT box(es) onto {args.out}")


if __name__ == "__main__":
    main()
