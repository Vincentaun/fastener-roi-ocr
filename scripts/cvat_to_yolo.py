"""
Step 2: CVAT XML annotations -> YOLO dataset.

Reads a CVAT-for-images XML (version 1.1) and produces a YOLO-format
dataset directory:

    <out>/
      images/train/*.jpg
      images/val/*.jpg
      labels/train/*.txt
      labels/val/*.txt
      data.yaml

CVAT image entries are matched to the rendered page images by *page order*
(image id 0 -> 1st page image, id 1 -> 2nd page image, ...). The original
CVAT image names contain CVAT-internal indexing, so order-based matching
is the most reliable.

Because the typical labelled set here is very small (e.g. 2 pages, 8 boxes),
by default we put EVERY image in both train and val. This is a deliberate
small-data choice: it lets YOLO compute a validation metric without us
losing any training data. For larger sets pass --val_split 0.2.

Usage:
    python cvat_to_yolo.py \
        --cvat_xml annotations.xml \
        --images_dir rendered_pages/ \
        --out_dir data/screws/ \
        [--val_split 0.0]
"""
from __future__ import annotations
import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple

import yaml


class Box(NamedTuple):
    label: str
    xtl: float
    ytl: float
    xbr: float
    ybr: float


class ImageAnnot(NamedTuple):
    image_id: int
    name: str
    width: int
    height: int
    boxes: list[Box]


def parse_cvat_xml(xml_path: Path) -> tuple[list[str], list[ImageAnnot]]:
    """Return (label_names, list_of_image_annotations) from a CVAT XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Collect labels from <meta><job><labels><label><name>...
    label_names: list[str] = []
    for lab in root.findall(".//meta/job/labels/label/name"):
        if lab.text:
            label_names.append(lab.text.strip())
    if not label_names:
        # Fallback: scan boxes for unique labels (preserve first-seen order)
        seen: list[str] = []
        for box in root.iter("box"):
            lab = box.attrib.get("label", "")
            if lab and lab not in seen:
                seen.append(lab)
        label_names = seen

    images: list[ImageAnnot] = []
    for img in root.iter("image"):
        boxes = [
            Box(
                label=b.attrib["label"],
                xtl=float(b.attrib["xtl"]),
                ytl=float(b.attrib["ytl"]),
                xbr=float(b.attrib["xbr"]),
                ybr=float(b.attrib["ybr"]),
            )
            for b in img.findall("box")
        ]
        images.append(ImageAnnot(
            image_id=int(img.attrib["id"]),
            name=img.attrib["name"],
            width=int(img.attrib["width"]),
            height=int(img.attrib["height"]),
            boxes=boxes,
        ))
    images.sort(key=lambda x: x.image_id)
    return label_names, images


def boxes_to_yolo_lines(boxes: list[Box], img_w: int, img_h: int,
                       label_to_id: dict[str, int]) -> list[str]:
    """Convert CVAT abs boxes -> YOLO normalized format (class cx cy w h)."""
    lines: list[str] = []
    for b in boxes:
        if b.label not in label_to_id:
            continue
        cls = label_to_id[b.label]
        cx = ((b.xtl + b.xbr) / 2.0) / img_w
        cy = ((b.ytl + b.ybr) / 2.0) / img_h
        w = (b.xbr - b.xtl) / img_w
        h = (b.ybr - b.ytl) / img_h
        # Clip to [0,1] just in case of slightly out-of-bounds annotation
        cx, cy = max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy))
        w, h = max(0.0, min(1.0, w)), max(0.0, min(1.0, h))
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert CVAT XML to a YOLO dataset.")
    ap.add_argument("--cvat_xml", required=True, type=Path)
    ap.add_argument("--images_dir", required=True, type=Path,
                    help="Directory containing the rendered page images (sorted alphabetically = page order).")
    ap.add_argument("--out_dir", required=True, type=Path,
                    help="Output dataset directory (will be created).")
    ap.add_argument("--val_split", type=float, default=0.0,
                    help="Fraction of images to put only in val. 0.0 means every image is in BOTH train and val (use when data is tiny).")
    ap.add_argument("--val_indices", type=str, default=None,
                    help="Explicit 0-based image indices for the val/test set, comma-separated "
                         "(e.g. '2,3' = pages 3 & 4 are test, the rest train). Overrides --val_split.")
    args = ap.parse_args()

    label_names, images = parse_cvat_xml(args.cvat_xml)
    if not label_names:
        raise SystemExit("No labels found in the CVAT XML.")
    label_to_id = {name: i for i, name in enumerate(label_names)}
    print(f"Labels: {label_names}")
    print(f"CVAT images with annotations: {len(images)}")

    # Match CVAT entries to actual page image files in --images_dir, by sorted order.
    page_imgs = sorted([p for p in args.images_dir.iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if len(page_imgs) < len(images):
        raise SystemExit(
            f"Found {len(page_imgs)} images in {args.images_dir} but XML references {len(images)} images. "
            f"Make sure the rendered page count and DPI match the annotation.")

    img_train = args.out_dir / "images" / "train"
    img_val = args.out_dir / "images" / "val"
    lab_train = args.out_dir / "labels" / "train"
    lab_val = args.out_dir / "labels" / "val"
    for d in (img_train, img_val, lab_train, lab_val):
        d.mkdir(parents=True, exist_ok=True)

    # Decide which images go to val
    n = len(images)
    explicit_val = None
    if args.val_indices is not None and args.val_indices.strip():
        explicit_val = {int(x) for x in args.val_indices.split(",") if x.strip() != ""}
        bad = {i for i in explicit_val if i < 0 or i >= n}
        if bad:
            raise SystemExit(f"--val_indices {sorted(bad)} out of range (only {n} annotated image(s): 0..{n-1}). "
                             f"Did you annotate the test pages and re-export?")

    if explicit_val is not None:
        val_indices = explicit_val
        mode = "explicit"
    else:
        n_val = max(1, int(round(n * args.val_split))) if args.val_split > 0 else 0
        val_indices = set(range(n - n_val, n)) if n_val > 0 else set()
        mode = "fraction"

    if explicit_val is not None:
        print(f"Split (explicit): train={sorted(set(range(n)) - val_indices)}, val={sorted(val_indices)}")
    else:
        print(f"Split: train={n - len(val_indices)}, val={len(val_indices) if val_indices else n} (val_split={args.val_split})")

    for idx, ann in enumerate(images):
        src_img = page_imgs[idx]
        # Sanity check size; warn if mismatch (annotations won't align)
        from PIL import Image
        with Image.open(src_img) as im:
            if (im.width, im.height) != (ann.width, ann.height):
                print(f"  ⚠ size mismatch on {src_img.name}: image={im.size}, xml=({ann.width}, {ann.height})")
        lines = boxes_to_yolo_lines(ann.boxes, ann.width, ann.height, label_to_id)

        targets: list[tuple[Path, Path]] = []
        if explicit_val is None and args.val_split == 0.0:
            # Mirror everything into both train and val
            targets = [(img_train, lab_train), (img_val, lab_val)]
        else:
            if idx in val_indices:
                targets = [(img_val, lab_val)]
            else:
                targets = [(img_train, lab_train)]

        stem = src_img.stem
        for img_dir, lab_dir in targets:
            shutil.copy2(src_img, img_dir / src_img.name)
            (lab_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        print(f"  [{idx}] {src_img.name}: {len(lines)} box(es)")

    # Write data.yaml
    data_yaml = {
        "path": str(args.out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(label_names)},
    }
    yaml_path = args.out_dir / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)
    print(f"Wrote {yaml_path}")


if __name__ == "__main__":
    main()
