# Screw-Spec ROI YOLO Pipeline / 扣件規格 ROI 偵測流程

Train a YOLOv8 object detector from CVAT ROI annotations, run it on any new PDF
engineering drawing to capture the ROIs, then OCR each ROI into a structured
screw-spec XML.

Outputs, by stage: **(1)** a CVAT-format `predictions.xml`, **(2)** a PDF with the
detected ROIs drawn on each page, and **(3)** a per-input `*_ocr.xml` listing each
screw and the value read from every ROI.

從 CVAT 標註訓練 YOLOv8，對新的 PDF 工程圖偵測 ROI，再對每個 ROI 進行 OCR，輸出結構化的扣件規格 XML。

```
            INPUT                         OUTPUT
   ┌──────────────────┐          ┌─────────────────────────┐
   │  PDF drawing     │  ─────►  │  predictions.xml (CVAT) │
   │  + CVAT XML      │   YOLO   │  drawing_with_roi.pdf   │
   └──────────────────┘          └─────────────────────────┘
```

---

## 1. Project layout / 專案結構

```
screw_roi_pipeline/
├── README.md
├── requirements.txt
├── scripts/
│   ├── pdf_to_images.py    # Step 1  PDF  → page images (fixed DPI)
│   ├── cvat_to_yolo.py     # Step 2  CVAT XML → YOLO dataset + data.yaml
│   ├── train.py            # Step 3  train YOLOv8
│   ├── infer.py            # Step 4  new PDF → CVAT XML + ROI-marked PDF
│   ├── preview_gt.py       # (opt.)  draw the CVAT ground-truth onto a PDF
│   └── ocr_roi.py          # Step 6  OCR the ROIs → screw-spec XML
├── ARGUMENTS.md            # full reference for every script's flags
├── data/                   # created by steps 1–2
│   ├── train_pages/
│   └── screws/
├── runs/                   # created by step 3 (best.pt lives here)
├── outputs/                # created by step 4 (XML + marked PDF)
└── OCR Result/             # created by step 6 (one screw-spec XML per input)
```

> **Full argument reference:** every flag of every script — type, default,
> required or not, and what it actually controls — is documented in
> [`ARGUMENTS.md`](ARGUMENTS.md). The sections below are the walkthrough;
> `ARGUMENTS.md` is the lookup table.

---

## 2. Installation / 安裝

```bash
# (recommended) create an isolated environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# CPU-only:
pip install -r requirements.txt

# GPU (CUDA 12.x) — install the CUDA build of torch FIRST, then the rest:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

`ultralytics` downloads the pretrained weight `yolov8n.pt` automatically on first
training run, so the machine needs internet access **once** (or pre-copy the file
for an offline/air-gapped site — see §6).

---

## 3. Run the pipeline / 執行流程

> All commands are run from the `screw_roi_pipeline/` folder.
> **The DPI must be identical in every step** (here: **200**, the DPI used when the
> drawings were annotated in CVAT → 1654×2338 px pages).

### Step 1 — PDF → page images

```bash
python scripts/pdf_to_images.py \
    --pdf 扣件規格辨識範本_緯2.pdf \
    --out_dir data/train_pages \
    --dpi 200
```

**What happens:** every page is rendered to a JPEG. Console output:

```
Rendering 扣件規格辨識範本_緯2.pdf at 200 DPI -> data/train_pages
  page 1: 1653x2338  ->  扣件規格辨識範本_緯2_page_001.jpg
  page 2: 1653x2338  ->  扣件規格辨識範本_緯2_page_002.jpg
Done. 2 page(s) written.
```

> The 1653 vs 1654 px difference is sub-pixel rounding and is harmless; step 2
> only prints it as a notice.

---

### Step 2 — CVAT XML → YOLO dataset

```bash
python scripts/cvat_to_yolo.py \
    --cvat_xml annotations.xml \
    --images_dir data/train_pages \
    --out_dir data/screws \
    --val_split 0.0
```

**What happens:** the CVAT boxes are converted to YOLO normalized format and a
`data.yaml` is written. `--val_split 0.0` means *every* image goes into both
`train` and `val` (a deliberate choice for tiny datasets). Console output:

```
Labels: ['Height']
CVAT images with annotations: 2
Split: train=2, val=2 (val_split=0.0)
  [0] 扣件規格辨識範本_緯2_page_001.jpg: 4 box(es)
  [1] 扣件規格辨識範本_緯2_page_002.jpg: 4 box(es)
Wrote data/screws/data.yaml
```

**Produced files:**

```
data/screws/
├── data.yaml
├── images/train/*.jpg   images/val/*.jpg
└── labels/train/*.txt   labels/val/*.txt
```

`data.yaml`:

```yaml
path: .../data/screws
train: images/train
val: images/val
names:
  0: Height
```

A label file (`labels/train/..._page_001.txt`) — one line per box,
`class cx cy w h` (all normalized 0–1):

```
0 0.551941 0.408591 0.041983 0.073999
0 0.574335 0.554944 0.036106 0.067126
0 0.615218 0.560701 0.037715 0.064602
0 0.710490 0.562329 0.045937 0.062074
```

---

### Step 3 — Train YOLOv8

```bash
python scripts/train.py \
    --data data/screws/data.yaml \
    --model yolov8n.pt \
    --epochs 100 \
    --imgsz 1280 \
    --batch 4
```

**What happens:** YOLOv8-nano is fine-tuned from pretrained weights. Console
output (abridged — real run):

```
Ultralytics 8.4.72 🚀 Python-3.12 torch-2.12.1 CPU (Intel Xeon @ 2.80GHz)
train: Scanning .../labels/train... 2 images, 0 backgrounds, 0 corrupt
val:   Scanning .../labels/val...   2 images, 0 backgrounds, 0 corrupt
Image sizes 1280 train, 1280 val
Starting training for 100 epochs...
...
100 epochs completed in 0.0xx hours.
Speed: 1.6ms preprocess, 100.6ms inference, 0.7ms postprocess per image
Done.  Best weights: runs/screw_roi/exp/weights/best.pt
```

**Produced file:** the trained model at
`runs/screw_roi/exp/weights/best.pt` (plus `last.pt`, loss curves, and
`train_batch*.jpg` previews in the same folder).

> `--imgsz 1280` is intentionally large: the ROIs (dimension callouts) are tiny
> relative to the full page, so downscaling to the 640 default shrinks them too
> much. Use 640 only for a quick smoke-test.

---

### Step 4 — Inference: new PDF → XML + marked PDF

```bash
python scripts/infer.py \
    --weights .\runs\detect\runs\screw_roi\exp\weights\best.pt \
    --pdf 扣件規格辨識範本.PDF \
    --out_dir outputs \
    --dpi 200 \
    --imgsz 1280 \
    --conf 0.25
```

**What happens:** each page is rendered, the model predicts ROIs, and two files
are written. Console output *(illustrative — with a properly trained `best.pt`)*:

```
Rendered 4 page(s) at 200 DPI.
Model labels: ['Height']
  扣件規格辨識範本_page_001.jpg: 3 box(es)
  扣件規格辨識範本_page_002.jpg: 2 box(es)
  扣件規格辨識範本_page_003.jpg: 4 box(es)
  扣件規格辨識範本_page_004.jpg: 3 box(es)
Wrote outputs/扣件規格辨識範本_predictions.xml  (12 box(es))
Wrote outputs/扣件規格辨識範本_with_roi.pdf
```

> If the model was trained from `yolov8n.yaml` (no pretrained weights) or on too
> little data, this step still runs but reports `0 box(es)` — the format and files
> are produced correctly, there is simply nothing above the confidence threshold.

**Produced files:**

* `outputs/<name>_predictions.xml` — CVAT-for-images 1.1, re-importable into CVAT
  for human review. Each detected box looks like:

  ```xml
  <image id="0" name="..._page_001.jpg" width="1653" height="2338">
    <box label="Height" source="auto" occluded="0"
         xtl="912.4" ytl="954.7" xbr="978.1" ybr="1102.5"
         z_order="0" confidence="0.78"/>
  </image>
  ```

* `outputs/<name>_with_roi.pdf` — the original PDF with each ROI drawn as a
  coloured rectangle and a `label conf` caption above it.

---

### Step 5 (optional) — Preview the ground-truth boxes

Use this to sanity-check geometry, or to show reviewers what a correct result
looks like. It draws the **CVAT annotations** straight onto the PDF using the same
renderer as Step 4.

```bash
python scripts/preview_gt.py \
    --cvat_xml annotations.xml \
    --pdf 扣件規格辨識範本_緯2.pdf \
    --out outputs/緯2_groundtruth_preview.pdf \
    --dpi 200
```

```
Drew 8 GT box(es) onto outputs/緯2_groundtruth_preview.pdf
```

---

### Step 6 — OCR the ROIs into a screw-spec XML

Reads the text inside each captured ROI and writes **one XML per input file**,
with pages separated. Input can be a PDF (any number of pages) or a single image.

```bash
python scripts/ocr_roi.py \
    --input 扣件規格辨識範本_緯2.pdf \
    --cvat_xml annotations.xml \
    --dpi 200 \
    --engine tesseract
```

**What happens:** each `Height` ROI is cropped, the blue annotation dot is
removed, the vertical text is de-rotated, and the value is read and pattern-matched
to a dimension. Console output (real run):

```
OCR engine: tesseract
  page 1 . roi 1: '3.02~3.18'  (raw='3.02~3.18~', conf=0.55)
  page 1 . roi 2: '3.87~4.13'  (raw='3.87~4.13', conf=0.80)
  ...
Wrote OCR Result/扣件規格辨識範本_緯2_ocr.xml  (2 page(s), 8 ROI(s))
```

**Produced file:** `OCR Result/<input-stem>_ocr.xml` —

```xml
<document source="..." pages="2" ocr_engine="tesseract">
  <page number="1">
    <screw name="screw 1">
      <height count="4">
        <roi id="1">
          <bbox xtl="878.19" ytl="868.78" xbr="947.63" ybr="1041.79"/>
          <value>3.02~3.18</value>
          <raw_text>3.02~3.18~</raw_text>
          <ocr_confidence>0.550</ocr_confidence>
        </roi>
        ...
```

`--engine paddle` uses PaddleOCR instead (better on rotated text; downloads its
models on first run). Requires the Tesseract binary for the default engine — see
requirements.txt.

---

## 4. One-shot run / 一鍵執行

```bash
python scripts/pdf_to_images.py --pdf 扣件規格辨識範本_緯2.pdf --out_dir data/train_pages --dpi 200
python scripts/cvat_to_yolo.py  --cvat_xml annotations.xml --images_dir data/train_pages --out_dir data/screws --val_split 0.0
python scripts/train.py         --data data/screws/data.yaml --model yolov8n.pt --epochs 100 --imgsz 1280 --batch 4
python scripts/infer.py         --weights runs/screw_roi/exp/weights/best.pt --pdf 扣件規格辨識範本.PDF --out_dir outputs --dpi 200 --imgsz 1280
python scripts/ocr_roi.py       --input 扣件規格辨識範本.PDF --cvat_xml outputs/扣件規格辨識範本_predictions.xml --dpi 200
```

---

## 5. Key parameters / 重要參數

| Param | Recommended | Why |
|-------|-------------|-----|
| `--dpi` | **200** | Must match the CVAT annotation DPI in *every* step |
| `--imgsz` | **1280** | ROIs are small; larger input keeps them detectable |
| `--model` | `yolov8n.pt` | Pretrained nano. Move to `yolov8s/m.pt` as data grows |
| `--epochs` | 100–300 | Small data trains noisily; `patience` stops early |
| `--conf` | 0.25 | Detection threshold; lower to 0.10–0.15 to see more |
| `--val_split` | 0.0 (tiny data) → 0.2 (real) | 0.0 mirrors every image into train+val |
| flip aug | **off** | screws/text are not symmetric — never flip |

---

## 6. Offline / air-gapped deployment / 離線部署

For a closed-network site, pre-stage these before going offline:

1. `pip download -r requirements.txt -d offline_wheels/` on a connected machine,
   then `pip install --no-index --find-links offline_wheels/ -r requirements.txt`.
2. Copy `yolov8n.pt` next to `train.py` (Ultralytics uses the local file instead
   of downloading).

---

## 7. Important notes / 重要提醒

* **Dataset size.** The shipped `annotations.xml` has only **8 boxes / 2 pages /
  1 label (`Height`)** — proof-of-concept only. For a model that generalises to
  new drawings, annotate **20–50+ pages** and split labels (e.g. `Height`,
  `Diameter`, `Thread`, `TitleBlock`, `Notes`, `Tolerance`).

* **Pretrained weights are required for real results.** Training from
  `yolov8n.yaml` (architecture only, no pretrained weights) runs but learns almost
  nothing on this little data. Always use `--model yolov8n.pt`.
