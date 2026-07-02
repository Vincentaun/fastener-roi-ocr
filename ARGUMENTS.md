# Arguments Reference / 參數說明

Every command-line argument for every script, with its exact type, default, and
what it actually controls. Flags marked **required** must be supplied; the rest
have defaults and can be omitted.

Two rules apply across several scripts:

- **DPI must be identical** everywhere a page is rendered (`pdf_to_images.py`,
  `infer.py`, `ocr_roi.py`) **and** must equal the DPI at which the drawings were
  annotated in CVAT. At 200 DPI an A4 page renders to 1654×2338 px. If the DPI
  differs, the box pixel-coordinates no longer line up with the image.
- **`--imgsz` must match** between `train.py` and `infer.py`. The model expects
  the same working resolution it was trained at.

---

## 1. `pdf_to_images.py` — render a PDF to page images

| Argument | Required | Type | Default | What it does |
|----------|----------|------|---------|--------------|
| `--pdf` | ✅ | path | — | Input PDF to render. |
| `--out_dir` | ✅ | path | — | Folder to write the page JPEGs into (created if missing). One file per page, named `<pdf-stem>_page_001.jpg`, `_page_002.jpg`, … |
| `--dpi` | | int | `200` | Render resolution. Higher = sharper but larger images. **Must equal the CVAT annotation DPI.** 200 is the value this project was annotated at. |

---

## 2. `cvat_to_yolo.py` — convert CVAT boxes to a YOLO dataset

| Argument | Required | Type | Default | What it does |
|----------|----------|------|---------|--------------|
| `--cvat_xml` | ✅ | path | — | CVAT-for-images 1.1 XML holding the labelled boxes. |
| `--images_dir` | ✅ | path | — | Folder of rendered page images (the output of step 1). Images are matched to the XML's `<image>` entries **by sorted filename order** = page order. |
| `--out_dir` | ✅ | path | — | Output dataset folder. Creates `images/train`, `images/val`, `labels/train`, `labels/val`, and `data.yaml`. |
| `--val_split` | | float | `0.0` | Fraction of images placed **only** in validation. `0.0` = every image is mirrored into **both** train and val (correct for a tiny dataset, so no page is wasted). `0.2` = the last 20% of images go to val. |
| `--val_indices` | | str | `None` | Explicit 0-based image indices for the val set, comma-separated, e.g. `"2,3"` = pages 3 & 4 are validation, the rest train. **Overrides `--val_split`.** Errors if an index is ≥ the number of annotated images (i.e. you asked to hold out a page you never annotated). |

**Choosing the split:** with only a couple of annotated pages, use the default
`--val_split 0.0`. For a real held-out test, annotate the test pages too and use
`--val_indices` (deterministic) or `--val_split 0.2` (fractional).

---

## 3. `train.py` — train the YOLOv8 detector

| Argument | Required | Type | Default | What it does |
|----------|----------|------|---------|--------------|
| `--data` | ✅ | path | — | Path to the `data.yaml` produced by step 2. |
| `--model` | | str | `yolov8n.pt` | Starting weights. `yolov8n.pt` = pretrained nano (**recommended**; auto-downloads once). `yolov8s/m/l/x.pt` = larger, slower, stronger. `yolov8n.yaml` = architecture only, **trained from scratch — learns almost nothing on small data; avoid.** |
| `--epochs` | | int | `100` | Number of full passes over the training set. Small datasets need more (100–300); `patience` stops early if it plateaus. |
| `--imgsz` | | int | `1280` | Resolution the model trains at. Deliberately large because the ROIs (dimension callouts) are tiny relative to a full page; 640 shrinks them too much. **Must match `infer.py --imgsz`.** |
| `--batch` | | int | `4` | Images processed per step. Lower it (2, or 1) if you run out of RAM/VRAM; raise it if you have headroom. |
| `--project` | | path | `runs/screw_roi` | Parent folder for run outputs. |
| `--name` | | str | `exp` | Sub-folder name for this run. Trained weights end up at `<project>/<name>/weights/best.pt`. Re-using a name overwrites (`exist_ok=True`). |
| `--device` | | str | `cpu` | Where to train. `cpu` (works everywhere), `0` (first GPU), `0,1` (two GPUs). |

**Not exposed as flags:** the script fixes sensible augmentation internally —
notably `fliplr=0`/`flipud=0` (screws and text are not mirror-symmetric, so
flipping would teach the model wrong shapes), plus mosaic/scale/HSV jitter tuned
for a small dataset. Edit the `model.train(...)` call to change these.

---

## 4. `infer.py` — detect ROIs on a new PDF

| Argument | Required | Type | Default | What it does |
|----------|----------|------|---------|--------------|
| `--weights` | ✅ | path | — | Trained model file, e.g. `runs/screw_roi/exp/weights/best.pt`. |
| `--pdf` | ✅ | path | — | PDF to run detection on. |
| `--out_dir` | ✅ | path | — | Output folder. Writes `<pdf-stem>_predictions.xml` (CVAT format) and `<pdf-stem>_with_roi.pdf` (boxes drawn on each page). |
| `--dpi` | | int | `200` | Render DPI. **Must match training/annotation DPI.** |
| `--conf` | | float | `0.25` | Confidence threshold. Detections scoring below this are dropped. Lower (`0.05`–`0.10`) surfaces weak/uncertain boxes (useful for debugging a small model); raise (`0.4`+) to keep only confident boxes and cut false positives. |
| `--iou` | | float | `0.45` | Non-max-suppression IoU threshold. When two boxes overlap more than this, the lower-scoring one is removed. Raise it to keep more overlapping boxes, lower it to be more aggressive about merging. |
| `--imgsz` | | int | `1280` | Inference resolution. **Should equal the training `--imgsz`.** |

---

## 5. `preview_gt.py` — draw ground-truth boxes onto a PDF (optional)

| Argument | Required | Type | Default | What it does |
|----------|----------|------|---------|--------------|
| `--cvat_xml` | ✅ | path | — | CVAT XML whose boxes you want to visualise. |
| `--pdf` | ✅ | path | — | The PDF those boxes were drawn on. |
| `--out` | ✅ | path | — | Output PDF path (a copy with the boxes drawn). |
| `--dpi` | | int | `200` | Render DPI; must match the annotation DPI. |

Use it to confirm box geometry is correct, or to show reviewers what a correct
result looks like versus the model's predictions.

---

## 6. `ocr_roi.py` — OCR the captured ROIs into a screw-spec XML

| Argument | Required | Type | Default | What it does |
|----------|----------|------|---------|--------------|
| `--input` | ✅ | path | — | The drawing to read: a **PDF** (one or many pages) **or** a single image (`.png/.jpg/.jpeg/.bmp/.tif`). One XML is produced for the whole file. |
| `--cvat_xml` | ✅ | path | — | CVAT XML holding the ROI boxes to read — either the YOLO `*_predictions.xml` from step 4, or hand annotations. Boxes are matched to pages by `<image id>` → page index. |
| `--out_dir` | | path | `OCR Result` | Output folder (created if missing). Writes `<input-stem>_ocr.xml`. |
| `--dpi` | | int | `200` | Render DPI for **PDF** input; must match the DPI at which the boxes were defined. Ignored for image input (the image is read at its own resolution). |
| `--label` | | str | `Height` | Which box label to OCR. Only `<box>` entries whose `label` equals this are read. Change it when you add other ROI types (e.g. `Diameter`). |
| `--engine` | | str (`tesseract` \| `paddle`) | `tesseract` | OCR backend. `tesseract` runs offline with the system Tesseract binary. `paddle` = PaddleOCR (better on rotated engineering text) but downloads its models on first run. |

**Output structure:** `<document>` → one `<page number="N">` per page → `<screw
name="screw N">` → `<height>` → one `<roi>` per box, each carrying `<bbox>`, the
parsed `<value>`, the exact `<raw_text>`, and `<ocr_confidence>`. If the box came
from YOLO it also carries `<detection_confidence>`.

---

## Quick cheat-sheet

```bash
# 1. render training PDF
python scripts/pdf_to_images.py --pdf TRAIN.pdf --out_dir data/train_pages --dpi 200

# 2. CVAT XML -> YOLO dataset
python scripts/cvat_to_yolo.py --cvat_xml annotations.xml --images_dir data/train_pages --out_dir data/screws --val_split 0.0

# 3. train
python scripts/train.py --data data/screws/data.yaml --model yolov8n.pt --epochs 100 --imgsz 1280

# 4. detect on a new PDF
python scripts/infer.py --weights runs/screw_roi/exp/weights/best.pt --pdf NEW.pdf --out_dir outputs --dpi 200 --imgsz 1280 --conf 0.25

# 5. (optional) preview ground-truth boxes
python scripts/preview_gt.py --cvat_xml annotations.xml --pdf TRAIN.pdf --out outputs/preview.pdf --dpi 200

# 6. OCR the ROIs -> screw-spec XML
python scripts/ocr_roi.py --input NEW.pdf --cvat_xml outputs/NEW_predictions.xml --dpi 200
```
