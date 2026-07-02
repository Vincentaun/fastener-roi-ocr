"""
Step 4: Inference -> CVAT XML + annotated PDF.

Given a trained YOLO model and any new PDF, this script:
  1. Renders each page at the same DPI used for training/annotation.
  2. Runs the YOLO model on each page.
  3. Writes a CVAT-for-images XML (matching the format of the input
     `annotations.xml`) so the user can re-import to CVAT for review.
  4. Writes a copy of the PDF with the predicted ROIs drawn on each page.

Usage:
    python infer.py \
        --weights runs/screw_roi/exp/weights/best.pt \
        --pdf NEW.pdf \
        --out_dir outputs/ \
        [--dpi 200] [--conf 0.25] [--imgsz 1280]
"""
from __future__ import annotations
import argparse
import datetime
import shutil
from pathlib import Path
from xml.dom import minidom
import xml.etree.ElementTree as ET

import fitz  # PyMuPDF
from PIL import Image
from ultralytics import YOLO


# Colors used to draw boxes per class label (RGB, 0-1 floats for PyMuPDF).
PALETTE = [
    (0.96, 0.24, 0.24),  # red
    (0.24, 0.55, 0.92),  # blue
    (0.20, 0.70, 0.36),  # green
    (0.95, 0.62, 0.13),  # orange
    (0.62, 0.30, 0.80),  # purple
    (0.20, 0.70, 0.70),  # teal
    (0.85, 0.20, 0.60),  # pink
    (0.45, 0.45, 0.45),  # gray
]


def render_pages(pdf_path: Path, out_dir: Path, dpi: int) -> list[Path]:
    """Render PDF pages to JPEGs at given DPI; return list of paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    doc = fitz.open(pdf_path)
    paths = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat, alpha=False)
        p = out_dir / f"{pdf_path.stem}_page_{i + 1:03d}.jpg"
        pix.save(p)
        paths.append(p)
    doc.close()
    return paths


def build_cvat_xml(label_names: list[str],
                   per_image_results: list[dict],
                   pdf_name: str) -> ET.ElementTree:
    """Build a CVAT-for-images 1.1 XML tree.

    `per_image_results` is a list of dicts:
        {"name": str, "width": int, "height": int,
         "boxes": [(label, xtl, ytl, xbr, ybr, conf), ...]}
    """
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"

    # Minimal but valid <meta> block (CVAT will accept missing optional fields).
    meta = ET.SubElement(root, "meta")
    job = ET.SubElement(meta, "job")
    ET.SubElement(job, "id").text = "0"
    ET.SubElement(job, "size").text = str(len(per_image_results))
    ET.SubElement(job, "mode").text = "annotation"
    ET.SubElement(job, "overlap").text = "0"
    ET.SubElement(job, "start_frame").text = "0"
    ET.SubElement(job, "stop_frame").text = str(max(0, len(per_image_results) - 1))
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f+00:00")
    ET.SubElement(job, "created").text = now
    ET.SubElement(job, "updated").text = now
    labels_el = ET.SubElement(job, "labels")
    for i, lname in enumerate(label_names):
        lab = ET.SubElement(labels_el, "label")
        ET.SubElement(lab, "name").text = lname
        ET.SubElement(lab, "color").text = "#{:02x}{:02x}{:02x}".format(
            int(PALETTE[i % len(PALETTE)][0] * 255),
            int(PALETTE[i % len(PALETTE)][1] * 255),
            int(PALETTE[i % len(PALETTE)][2] * 255),
        )
        ET.SubElement(lab, "type").text = "any"
        ET.SubElement(lab, "attributes")
    ET.SubElement(meta, "dumped").text = now
    ET.SubElement(meta, "source").text = pdf_name

    for idx, info in enumerate(per_image_results):
        img_el = ET.SubElement(root, "image", {
            "id": str(idx),
            "name": info["name"],
            "width": str(info["width"]),
            "height": str(info["height"]),
        })
        for (label, xtl, ytl, xbr, ybr, conf) in info["boxes"]:
            ET.SubElement(img_el, "box", {
                "label": label,
                "source": "auto",
                "occluded": "0",
                "xtl": f"{xtl:.2f}",
                "ytl": f"{ytl:.2f}",
                "xbr": f"{xbr:.2f}",
                "ybr": f"{ybr:.2f}",
                "z_order": "0",
                "confidence": f"{conf:.4f}",
            })
    return ET.ElementTree(root)


def write_pretty_xml(tree: ET.ElementTree, path: Path) -> None:
    rough = ET.tostring(tree.getroot(), encoding="utf-8")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ", encoding="utf-8")
    path.write_bytes(pretty)


def draw_boxes_on_pdf(src_pdf: Path, out_pdf: Path,
                     per_image_results: list[dict],
                     label_names: list[str],
                     dpi: int) -> None:
    """Draw predicted boxes onto a copy of the PDF.

    Box coords are in *image pixels* at `dpi`. PyMuPDF works in PDF points
    (72 dpi), so divide pixels by (dpi/72) to convert.
    """
    label_to_idx = {n: i for i, n in enumerate(label_names)}
    px_to_pt = 72.0 / dpi

    doc = fitz.open(src_pdf)
    for page_idx, info in enumerate(per_image_results):
        if page_idx >= len(doc):
            break
        page = doc[page_idx]
        for (label, xtl, ytl, xbr, ybr, conf) in info["boxes"]:
            rect = fitz.Rect(xtl * px_to_pt, ytl * px_to_pt,
                            xbr * px_to_pt, ybr * px_to_pt)
            color = PALETTE[label_to_idx.get(label, 0) % len(PALETTE)]
            page.draw_rect(rect, color=color, width=1.2)
            # Caption above the box
            cap = f"{label} {conf:.2f}"
            page.insert_text(
                fitz.Point(rect.x0, max(rect.y0 - 2, 8)),
                cap, fontsize=7, color=color,
            )
    doc.save(out_pdf)
    doc.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Run YOLO on a PDF and output CVAT XML + annotated PDF.")
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--dpi", type=int, default=200,
                    help="Must match the DPI used to annotate the training data.")
    ap.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    ap.add_argument("--imgsz", type=int, default=1280, help="Inference image size (must match training)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = args.out_dir / "_pages"
    page_paths = render_pages(args.pdf, pages_dir, args.dpi)
    print(f"Rendered {len(page_paths)} page(s) at {args.dpi} DPI.")

    model = YOLO(str(args.weights))
    label_names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
    print(f"Model labels: {label_names}")

    per_image_results: list[dict] = []
    total_boxes = 0
    for p in page_paths:
        with Image.open(p) as im:
            w, h = im.size
        res = model.predict(source=str(p), imgsz=args.imgsz,
                           conf=args.conf, iou=args.iou,
                           verbose=False, save=False)[0]
        boxes: list[tuple] = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, cls_id in zip(xyxy, confs, clss):
                name = label_names[cls_id] if cls_id < len(label_names) else str(cls_id)
                boxes.append((name, float(x1), float(y1), float(x2), float(y2), float(c)))
        total_boxes += len(boxes)
        per_image_results.append({
            "name": p.name, "width": w, "height": h, "boxes": boxes,
        })
        print(f"  {p.name}: {len(boxes)} box(es)")

    # CVAT XML
    xml_path = args.out_dir / f"{args.pdf.stem}_predictions.xml"
    tree = build_cvat_xml(label_names, per_image_results, args.pdf.name)
    write_pretty_xml(tree, xml_path)
    print(f"Wrote {xml_path}  ({total_boxes} box(es))")

    # Annotated PDF
    pdf_out = args.out_dir / f"{args.pdf.stem}_with_roi.pdf"
    draw_boxes_on_pdf(args.pdf, pdf_out, per_image_results, label_names, args.dpi)
    print(f"Wrote {pdf_out}")

    # Clean up intermediate pages
    shutil.rmtree(pages_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
