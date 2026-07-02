"""
OCR step: read the text inside each captured ROI and write ONE screw-spec XML
per input file.

Pipeline position:
    infer.py  ->  ROI boxes (CVAT XML)  ->  [ocr_roi.py]  ->  OCR Result/<name>_ocr.xml

Input:
    --input      the drawing, either a PDF (one or many pages) or a single image
                 (PNG/JPG). One XML is produced for the whole input file.
    --cvat_xml   a CVAT-for-images XML holding the ROI boxes (YOLO predictions
                 from infer.py, or hand annotations). Boxes are matched to pages
                 by image id -> page index.

Output XML layout (pages are separated explicitly):
    <document source="..." pages="2" ocr_engine="tesseract">
      <page number="1">
        <screw name="screw 1">
          <height count="4">
            <roi id="1">
              <bbox .../><value>3.02~3.18</value>
              <raw_text>...</raw_text><ocr_confidence>...</ocr_confidence>
            </roi>
          </height>
        </screw>
      </page>
      <page number="2"> ... </page>
    </document>

Each ROI is a rotated (vertical) dimension callout like "3.02~3.18" that also has
a blue annotation dot on it. The reader removes the dot, de-rotates, upscales, and
prefers OCR candidates that match a dimension pattern, then normalizes the range
separator to '~'. raw_text and confidence are kept for human review.

Usage:
    python ocr_roi.py --input 扣件規格辨識範本_緯2.pdf --cvat_xml annotations.xml
    python ocr_roi.py --input page.png --cvat_xml page_boxes.xml --engine paddle
"""
from __future__ import annotations
import argparse
import datetime
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

import numpy as np
from PIL import Image, ImageOps

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# A dimension range "3.02~3.18" / "11.60-12.40", or a single number "9.03".
DIM_RANGE = re.compile(r"\d{1,3}\.\d{1,2}\s*[~\-]\s*\d{1,3}\.\d{1,2}")
NUM_ONE = re.compile(r"\d{1,3}\.\d{1,2}")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ---------------------------- input -> page images --------------------------
def load_pages(path: Path, dpi: int) -> list[Image.Image]:
    """Return a list of RGB page images. PDF -> one per page; image -> single page."""
    if path.suffix.lower() == ".pdf":
        import fitz  # PyMuPDF
        zoom = dpi / 72.0
        doc = fitz.open(path)
        pages = []
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            pages.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
        doc.close()
        return pages
    if path.suffix.lower() in IMAGE_EXTS:
        return [Image.open(path).convert("RGB")]
    raise SystemExit(f"Unsupported input type: {path.suffix} (use PDF or PNG/JPG).")


# ---------------------------- image preprocessing ---------------------------
def _remove_dot(im: Image.Image) -> Image.Image:
    """Whiten the blue/purple CVAT annotation dot so it doesn't corrupt OCR."""
    a = np.array(im.convert("RGB")).astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mask = (b - r > 35) & (b - g > 35) & (b > 80)
    a[mask] = [255, 255, 255]
    return Image.fromarray(a.astype(np.uint8))


def _prep(im: Image.Image, angle: int, scale: int, thresh: bool) -> Image.Image:
    """De-dot -> grayscale -> rotate -> upscale -> (optional threshold) -> white pad."""
    im = _remove_dot(im)
    g = im.convert("L").rotate(angle, expand=True, fillcolor=255)
    g = g.resize((g.width * scale, g.height * scale), Image.LANCZOS)
    if thresh:
        a = np.array(g)
        g = Image.fromarray(np.where(a > 160, 255, 0).astype(np.uint8))
    return ImageOps.expand(g, border=25, fill=255)


def _normalize(text: str) -> str:
    """Pull the dimension out of an OCR string and set the separator to '~'.
    Returns '' when nothing dimension-like is present (flag for review)."""
    compact = text.replace(" ", "")
    m = DIM_RANGE.search(compact) or NUM_ONE.search(compact)
    if not m:
        return ""
    return re.sub(r"(?<=\d)\s*-\s*(?=\d)", "~", m.group(0))


# ------------------------------- OCR engines --------------------------------
class TesseractEngine:
    name = "tesseract"

    def __init__(self):
        import pytesseract
        self._pt = pytesseract

    def read(self, crop: Image.Image) -> tuple[str, str, float]:
        """Return (value, raw_text, confidence 0-1). Sweeps rotation/scale/psm
        and strongly prefers candidates that match a dimension pattern."""
        best = None  # (score, conf, value, raw)
        for angle in (90, 270):                 # ROIs are vertical
            for scale in (4, 6):
                for thresh in (False, True):
                    pre = _prep(crop, angle, scale, thresh)
                    for psm in (7, 11, 6):
                        cfg = f"--psm {psm} -c tessedit_char_whitelist=0123456789.~-"
                        d = self._pt.image_to_data(
                            pre, config=cfg, output_type=self._pt.Output.DICT)
                        words = [(t.strip(), float(c))
                                 for t, c in zip(d["text"], d["conf"])
                                 if t.strip() and str(c) != "-1"]
                        if not words:
                            continue
                        raw = "".join(w for w, _ in words)
                        conf = sum(c for _, c in words) / len(words)
                        value = _normalize(raw)
                        # +200 if a real dimension was parsed -> dominates the pick
                        score = conf + (200 if DIM_RANGE.search(raw.replace(" ", "")) else 0)
                        if best is None or score > best[0]:
                            best = (score, conf, value, raw)
        if best is None:
            return "", "", 0.0
        _, conf, value, raw = best
        return value, raw, round(conf / 100.0, 3)


class PaddleEngine:
    name = "paddle"

    def __init__(self):
        from paddleocr import PaddleOCR
        self._ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

    def read(self, crop: Image.Image) -> tuple[str, str, float]:
        arr = np.array(_remove_dot(crop).convert("RGB"))
        result = self._ocr.ocr(arr, cls=True)
        texts, confs = [], []
        for line in (result or []):
            for _box, (txt, cf) in (line or []):
                texts.append(txt); confs.append(float(cf))
        if not texts:
            return "", "", 0.0
        raw = " ".join(texts)
        return _normalize(raw), raw, round(sum(confs) / len(confs), 3)


def get_engine(name: str):
    if name == "tesseract":
        return TesseractEngine()
    if name == "paddle":
        return PaddleEngine()
    raise SystemExit(f"Unknown engine '{name}'.")


# ------------------------------- XML output ---------------------------------
def build_xml(source: str, engine_name: str, pages: list[dict]) -> ET.ElementTree:
    root = ET.Element("document", {
        "source": source,
        "pages": str(len(pages)),
        "ocr_engine": engine_name,
        "generated": datetime.datetime.now(datetime.timezone.utc)
                         .strftime("%Y-%m-%d %H:%M:%S+00:00"),
    })
    for pg in pages:
        page_el = ET.SubElement(root, "page", {"number": str(pg["number"])})
        screw_el = ET.SubElement(page_el, "screw", {"name": pg["screw_name"]})
        height_el = ET.SubElement(screw_el, "height", {"count": str(len(pg["rois"]))})
        for r in pg["rois"]:
            roi_el = ET.SubElement(height_el, "roi", {"id": str(r["id"])})
            ET.SubElement(roi_el, "bbox", {
                "xtl": f"{r['xtl']:.2f}", "ytl": f"{r['ytl']:.2f}",
                "xbr": f"{r['xbr']:.2f}", "ybr": f"{r['ybr']:.2f}",
            })
            ET.SubElement(roi_el, "value").text = r["value"]
            ET.SubElement(roi_el, "raw_text").text = r["raw"]
            ET.SubElement(roi_el, "ocr_confidence").text = f"{r['ocr_conf']:.3f}"
            if r.get("det_conf") is not None:
                ET.SubElement(roi_el, "detection_confidence").text = f"{r['det_conf']:.3f}"
    return ET.ElementTree(root)


def write_pretty(tree: ET.ElementTree, path: Path) -> None:
    rough = ET.tostring(tree.getroot(), encoding="utf-8")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ", encoding="utf-8")
    path.write_bytes(pretty)


# --------------------------------- main -------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="OCR captured ROIs into one screw-spec XML per input file.")
    ap.add_argument("--input", required=True, type=Path, help="PDF or PNG/JPG drawing.")
    ap.add_argument("--cvat_xml", required=True, type=Path,
                    help="CVAT XML with ROI boxes (YOLO predictions or annotations).")
    ap.add_argument("--out_dir", type=Path, default=Path("OCR Result"),
                    help="Output folder (created if missing). Default: 'OCR Result'.")
    ap.add_argument("--dpi", type=int, default=200, help="Render DPI for PDFs; must match the box DPI.")
    ap.add_argument("--label", default="Height", help="Which ROI label to OCR.")
    ap.add_argument("--engine", default="tesseract", choices=["tesseract", "paddle"])
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    engine = get_engine(args.engine)
    print(f"OCR engine: {engine.name}")

    page_imgs = load_pages(args.input, args.dpi)
    tree = ET.parse(args.cvat_xml)
    img_entries = {int(e.attrib["id"]): e for e in tree.iter("image")}

    pages: list[dict] = []
    for pidx, page_img in enumerate(page_imgs):
        img_el = img_entries.get(pidx)
        boxes = [b for b in img_el.findall("box")
                 if b.attrib.get("label") == args.label] if img_el is not None else []
        rois = []
        for i, b in enumerate(boxes, start=1):
            x1, y1, x2, y2 = (float(b.attrib[k]) for k in ("xtl", "ytl", "xbr", "ybr"))
            crop = page_img.crop((int(x1), int(y1), int(x2), int(y2)))
            value, raw, conf = engine.read(crop)
            det = b.attrib.get("confidence")
            rois.append({
                "id": i, "xtl": x1, "ytl": y1, "xbr": x2, "ybr": y2,
                "value": value, "raw": raw, "ocr_conf": conf,
                "det_conf": float(det) if det is not None else None,
            })
            flag = "" if value else "   (review)"
            print(f"  page {pidx+1} . roi {i}: '{value}'  (raw='{raw}', conf={conf:.2f}){flag}")
        pages.append({
            "number": pidx + 1, "screw_name": f"screw {pidx + 1}", "rois": rois,
        })

    out_path = args.out_dir / f"{args.input.stem}_ocr.xml"
    write_pretty(build_xml(args.input.name, engine.name, pages), out_path)
    total = sum(len(p["rois"]) for p in pages)
    print(f"\nWrote {out_path}  ({len(pages)} page(s), {total} ROI(s))")


if __name__ == "__main__":
    main()
