"""
Step 1: PDF → page images.

Renders each page of a PDF at a fixed DPI so the pixel coordinates in
the CVAT annotations match the rendered image size.

Usage:
    python pdf_to_images.py --pdf INPUT.pdf --out_dir DIR [--dpi 200]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import fitz  # PyMuPDF


def pdf_to_images(pdf_path: Path, out_dir: Path, dpi: int = 200) -> list[Path]:
    """Render every page of `pdf_path` to JPEGs in `out_dir`.

    Returns the list of written image paths (one per page).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    doc = fitz.open(pdf_path)
    written: list[Path] = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_path = out_dir / f"{pdf_path.stem}_page_{i + 1:03d}.jpg"
        pix.save(out_path)
        written.append(out_path)
        print(f"  page {i + 1}: {pix.width}x{pix.height}  ->  {out_path.name}")
    doc.close()
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a PDF's pages to JPEGs.")
    ap.add_argument("--pdf", required=True, type=Path, help="Path to input PDF")
    ap.add_argument("--out_dir", required=True, type=Path, help="Directory for output images")
    ap.add_argument("--dpi", type=int, default=200,
                    help="Render DPI (must match the DPI used when annotating in CVAT). Default 200.")
    args = ap.parse_args()

    print(f"Rendering {args.pdf.name} at {args.dpi} DPI -> {args.out_dir}")
    written = pdf_to_images(args.pdf, args.out_dir, args.dpi)
    print(f"Done. {len(written)} page(s) written.")


if __name__ == "__main__":
    main()
