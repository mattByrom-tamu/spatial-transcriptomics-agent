"""
Self-contained PDF figure/image + text extractor with progress logging.

Requires:
  pip install pymupdf

What it produces:
  <out_root>/<paper_id>/
    images/   -> extracted raster images
    json/     -> one JSON per image
    text/     -> paper.txt (entire paper text)

Notes:
- Captions are best-effort (PDFs vary wildly). We store both a "best" guess and
  multiple candidates plus local surrounding text for downstream processing.
- This extracts embedded raster images. Vector-only figures may not appear as images.
"""

from __future__ import annotations

import json
import re
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import fitz  # PyMuPDF


def extract_paper_assets(
    pdf_path: str | Path,
    out_root: str | Path,
    paper_id: Optional[str] = None,
    image_format: str = "png",
    include_page_markers_in_paper_text: bool = True,
    local_text_window: int = 1,  # 1 => prev/page/next; 2 => prev2..next2
    verbose: bool = True,
    log_every_n_pages: int = 5,
    log_each_image: bool = True,
) -> Dict[str, Any]:
    """
    Extract all embedded raster images from a PDF and write:
      - one text file containing the entire paper text
      - one JSON per extracted image containing:
          * image id + image file path
          * local text around the image page (prev/page/next; configurable)
          * caption best-guess + candidates
    Also creates a paper folder with subfolders images/, json/, text/.

    Parameters
    ----------
    pdf_path : path-like
        PDF to process
    out_root : path-like
        Root output directory
    paper_id : optional str
        Folder name. Defaults to PDF stem (sanitized).
    image_format : str
        "png" (recommended) or "jpg"/"jpeg"
    include_page_markers_in_paper_text : bool
        If True, inserts "--- Page N ---" markers between pages in paper.txt
    local_text_window : int
        Number of pages before/after to include in local text.
    verbose : bool
        If True, prints progress logs.
    log_every_n_pages : int
        Log once every N pages (e.g., 5). Set to 1 for every page.
    log_each_image : bool
        If True, logs after each saved image.

    Returns
    -------
    dict summary with paths and counts
    """

    # -----------------------------
    # Internal helpers (self-contained)
    # -----------------------------
    start_time = time.time()

    def log(msg: str) -> None:
        if verbose:
            elapsed = time.time() - start_time
            print(f"[{elapsed:6.1f}s] {msg}")

    def _safe_stem(name: str, max_len: int = 120) -> str:
        name = re.sub(r"[^\w\-.]+", "_", name.strip())
        return name[:max_len] if len(name) > max_len else name

    def _sha1_short(text: str, n: int = 10) -> str:
        return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:n]

    FIGURE_CAPTION_LINE_RE = re.compile(
        r"(?im)^\s*(fig(?:ure)?|supp\.?\s*fig(?:ure)?|extended\s*data\s*fig(?:ure)?)\s*\.?\s*\d+[a-zA-Z]?\s*[:.\-]\s+.+$"
    )
    FIGURE_MENTION_RE = re.compile(
        r"(?im)(fig(?:ure)?|supp\.?\s*fig(?:ure)?|extended\s*data\s*fig(?:ure)?)\s*\.?\s*\d+[a-zA-Z]?\s*[:.\-]?"
    )

    def _page_text(doc: fitz.Document, pno: int) -> str:
        if pno < 0 or pno >= doc.page_count:
            return ""
        return doc.load_page(pno).get_text("text") or ""

    def _extract_full_text(doc: fitz.Document) -> str:
        pages = []
        for i in range(doc.page_count):
            t = _page_text(doc, i).strip()
            if not t:
                continue
            if include_page_markers_in_paper_text:
                pages.append(f"\n\n--- Page {i+1} ---\n\n{t}")
            else:
                pages.append(t)
        return "".join(pages).strip()

    def _get_local_text(doc: fitz.Document, pno: int) -> Dict[str, Any]:
        # Collect a window of pages around pno
        window = []
        for k in range(pno - local_text_window, pno + local_text_window + 1):
            window.append((k, _page_text(doc, k)))

        prev_pages = [(i, t) for i, t in window if i < pno]
        next_pages = [(i, t) for i, t in window if i > pno]
        cur_text = _page_text(doc, pno)

        combined = "\n\n".join([t for _, t in window if t]).strip()

        return {
            "page_number": pno + 1,
            "window": {
                "before_pages": [{"page_number": i + 1, "text": t} for i, t in prev_pages if t],
                "page": {"page_number": pno + 1, "text": cur_text},
                "after_pages": [{"page_number": i + 1, "text": t} for i, t in next_pages if t],
            },
            "combined_window_text": combined,
        }

    def _caption_candidates_from_text(text: str) -> List[str]:
        # 1) Strong signal: lines that look like "Figure 2: ..."
        cands: List[str] = []
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for ln in lines:
            if FIGURE_CAPTION_LINE_RE.match(ln):
                cands.append(ln)

        # 2) Weaker: snippets around "Figure X"
        if not cands:
            matches = list(FIGURE_MENTION_RE.finditer(text))
            for m in matches[:8]:
                start = max(0, m.start() - 120)
                end = min(len(text), m.end() + 450)
                snippet = " ".join(text[start:end].split())
                if snippet:
                    cands.append(snippet)

        # Dedup
        seen = set()
        out = []
        for c in cands:
            key = c.lower()
            if key not in seen:
                seen.add(key)
                out.append(c)
        return out[:15]

    def _caption_near_image_on_page(page: fitz.Page, img_rect: fitz.Rect) -> Dict[str, Any]:
        """
        Best-effort caption guess from nearby text blocks.
        Heuristic: captions often appear directly under an image.
        """
        blocks = page.get_text("blocks") or []
        # block tuple: (x0, y0, x1, y1, "text", block_no, block_type)
        below: List[Tuple[float, str]] = []

        for b in blocks:
            if len(b) < 5:
                continue
            x0, y0, x1, y1, txt = b[0], b[1], b[2], b[3], b[4]
            if not txt or not txt.strip():
                continue

            br = fitz.Rect(x0, y0, x1, y1)
            vertical_gap = br.y0 - img_rect.y1

            # below within ~220 pts, with horizontal overlap
            if 0 <= vertical_gap <= 220:
                overlap = min(br.x1, img_rect.x1) - max(br.x0, img_rect.x0)
                if overlap > 20:
                    cleaned = " ".join(txt.split())
                    below.append((vertical_gap, cleaned))

        below.sort(key=lambda t: t[0])
        ordered = [t for _, t in below]

        candidates = []
        for t in ordered:
            if FIGURE_MENTION_RE.search(t):
                candidates.append(t)
            elif len(t) >= 30:
                candidates.append(t)

        best = candidates[0] if candidates else (ordered[0] if ordered else "")
        return {
            "caption_best": best,
            "caption_candidates": candidates[:15],
            "nearby_text_blocks": ordered[:20],
        }

    # -----------------------------
    # Main execution
    # -----------------------------
    pdf_path = Path(pdf_path)
    out_root = Path(out_root)

    if paper_id is None:
        paper_id = _safe_stem(pdf_path.stem)

    paper_dir = out_root / paper_id
    images_dir = paper_dir / "images"
    json_dir = paper_dir / "json"
    text_dir = paper_dir / "text"

    images_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    log(f"Opened PDF: {pdf_path.name}")
    doc = fitz.open(str(pdf_path))
    log(f"Pages: {doc.page_count}")
    log(f"Output directory: {paper_dir}")

    # One text doc for the entire paper
    log("Extracting full paper text...")
    full_text = _extract_full_text(doc)
    paper_txt_path = text_dir / "paper.txt"
    paper_txt_path.write_text(full_text, encoding="utf-8")
    log(f"Saved paper text → {paper_txt_path}")

    extracted_records: List[Dict[str, Any]] = []
    global_img_idx = 0

    for pno in range(doc.page_count):
        if verbose and (pno % max(1, log_every_n_pages) == 0):
            log(f"Processing page {pno + 1}/{doc.page_count}")

        page = doc.load_page(pno)

        # Local text context for this page
        local_text = _get_local_text(doc, pno)
        fallback_candidates = _caption_candidates_from_text(local_text["combined_window_text"])

        # Embedded images on this page
        img_list = page.get_images(full=True)  # tuples; first element is xref
        if not img_list:
            continue

        log(f"  Found {len(img_list)} image(s) on page {pno + 1}")

        # For rect-based caption guessing: map xref->rects
        xref_to_rects: Dict[int, List[fitz.Rect]] = {}
        for img in img_list:
            xref = img[0]
            rects = page.get_image_rects(xref)
            if rects:
                xref_to_rects.setdefault(xref, []).extend(rects)

        for img in img_list:
            xref = img[0]
            global_img_idx += 1

            # Extract image bytes/pixmap
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n - pix.alpha >= 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)  # convert CMYK to RGB
            except Exception as e:
                log(f"    Skipping image xref={xref} due to error: {e}")
                continue

            img_name = f"p{pno+1:03d}_img{global_img_idx:04d}.{image_format.lower()}"
            img_path = images_dir / img_name

            # Save
            fmt = image_format.lower()
            if fmt in ("jpg", "jpeg", "png"):
                pix.save(str(img_path))
            else:
                img_path = img_path.with_suffix(".png")
                pix.save(str(img_path))

            if log_each_image:
                log(f"    Saved image → {img_path.name}")

            # Caption best-effort
            rects = xref_to_rects.get(xref, [])
            caption_info = {"caption_best": "", "caption_candidates": [], "nearby_text_blocks": []}
            if rects:
                caption_info = _caption_near_image_on_page(page, rects[0])

            merged_candidates: List[str] = []
            for c in caption_info.get("caption_candidates", []):
                if c and c not in merged_candidates:
                    merged_candidates.append(c)
            for c in fallback_candidates:
                if c and c not in merged_candidates:
                    merged_candidates.append(c)

            caption_best = caption_info.get("caption_best", "") or (merged_candidates[0] if merged_candidates else "")

            # Stable-ish identifier
            image_identifier = f"{paper_id}::page={pno+1}::xref={xref}::idx={global_img_idx}"
            image_id = _sha1_short(image_identifier)

            record: Dict[str, Any] = {
                "paper_id": paper_id,
                "pdf_filename": pdf_path.name,
                "image_identifier": image_identifier,
                "image_id": image_id,
                "page_number": pno + 1,
                "xref": xref,
                "image_filename": img_path.name,
                "image_path": str(img_path),
                "text": local_text,  # includes window + combined
                "caption": {
                    "best": caption_best,
                    "candidates": merged_candidates[:15],
                    "nearby_text_blocks": caption_info.get("nearby_text_blocks", [])[:20],
                    "method_notes": {
                        "near_image_blocks_used": bool(rects),
                        "fallback_candidates_from_local_text": True,
                        "local_text_window_pages_each_side": local_text_window,
                    },
                },
                "image_rects": [
                    {"x0": r.x0, "y0": r.y0, "x1": r.x1, "y1": r.y1} for r in (rects[:5] if rects else [])
                ],
            }

            json_path = json_dir / f"{img_path.stem}.json"
            json_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
            extracted_records.append(record)

    doc.close()

    log(f"Done. Extracted {len(extracted_records)} images total.")
    log(f"Images dir: {images_dir}")
    log(f"JSON dir:   {json_dir}")
    log(f"Paper text: {paper_txt_path}")

    return {
        "paper_id": paper_id,
        "pdf_path": str(pdf_path),
        "output_dir": str(paper_dir),
        "paper_text_path": str(paper_txt_path),
        "num_images": len(extracted_records),
        "images_dir": str(images_dir),
        "json_dir": str(json_dir),
        "notes": [
            "Extracts embedded raster images via PyMuPDF page.get_images().",
            "Stores local text window around each image page for robustness.",
            "Caption is best-effort; JSON includes candidates for downstream selection.",
        ],
    }


# ---- optional CLI-ish usage example ----
if __name__ == "__main__":
    # Edit these lines and run: python extractor_with_progress.py
    summary = extract_paper_assets(
        pdf_path="E:/WangResearch/Papers for GPT/C-SIDE.pdf",
        out_root="output",
        verbose=True,
        log_every_n_pages=5,
        log_each_image=True,
    )
    print(json.dumps(summary, indent=2))
