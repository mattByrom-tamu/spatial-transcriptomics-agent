import os
import re
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import fitz  # PyMuPDF


# -----------------------------
# Data classes
# -----------------------------
@dataclass
class PageContent:
    page_number: int
    text: str


@dataclass
class FigureContent:
    page_number: int
    image_index: int
    image_path: str


@dataclass
class IngestedPaper:
    source_path: str
    num_pages: int
    pages: List[PageContent]
    figures: List[FigureContent]


# -----------------------------
# Helpers
# -----------------------------
CAPTION_RE = re.compile(r"(?is)^\s*(fig\.|figure)\s*\d+")
PANEL_LABEL_RE = re.compile(r"(?i)^\s*\(?\s*([a-h])\s*\)?\s*$")  # a..h (common)


def safe_stem_from_path(pdf_path: str) -> str:
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    stem = re.sub(r"[^A-Za-z0-9_\-]+", "_", stem).strip("_")
    return stem or "paper"


def rect_union(rects: List[fitz.Rect]) -> Optional[fitz.Rect]:
    if not rects:
        return None
    u = fitz.Rect(rects[0])
    for r in rects[1:]:
        u |= r
    return u


def clamp_rect(r: fitz.Rect, page_rect: fitz.Rect) -> fitz.Rect:
    x0 = max(page_rect.x0, min(r.x0, page_rect.x1))
    y0 = max(page_rect.y0, min(r.y0, page_rect.y1))
    x1 = max(page_rect.x0, min(r.x1, page_rect.x1))
    y1 = max(page_rect.y0, min(r.y1, page_rect.y1))
    rr = fitz.Rect(x0, y0, x1, y1)
    # Ensure non-empty
    if rr.x1 <= rr.x0 + 1:
        rr.x1 = rr.x0 + 1
    if rr.y1 <= rr.y0 + 1:
        rr.y1 = rr.y0 + 1
    return rr


def is_low_variance_pix(pix: fitz.Pixmap, std_thresh: float = 2.0) -> bool:
    """
    Simple blank/near-blank detection without numpy:
    - sample bytes and approximate variance
    """
    try:
        samples = pix.samples
        if not samples:
            return True
        # Convert to grayscale-ish by sampling every 3rd/4th byte (cheap)
        step = max(1, len(samples) // 20000)  # cap work
        vals = []
        n = pix.n  # channels
        for i in range(0, len(samples), step * n):
            # Average first 3 channels if present
            if i + 2 < len(samples):
                v = (samples[i] + samples[i + 1] + samples[i + 2]) / 3.0
            else:
                v = samples[i]
            vals.append(v)
            if len(vals) >= 8000:
                break

        if len(vals) < 50:
            return True

        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = var ** 0.5
        return std < std_thresh
    except Exception:
        return False


def render_clip(page: fitz.Page, clip: fitz.Rect, zoom: float = 3.0) -> fitz.Pixmap:
    mat = fitz.Matrix(zoom, zoom)
    return page.get_pixmap(matrix=mat, clip=clip, alpha=False)


def find_caption_blocks(page: fitz.Page) -> List[Tuple[fitz.Rect, str]]:
    """
    Return [(bbox, text)] for blocks whose text begins with Fig./Figure <number>
    """
    blocks = page.get_text("blocks")  # list of tuples
    out: List[Tuple[fitz.Rect, str]] = []
    for b in blocks:
        x0, y0, x1, y1, txt = b[0], b[1], b[2], b[3], b[4]
        if not txt:
            continue
        t = txt.strip()
        if CAPTION_RE.match(t):
            out.append((fitz.Rect(x0, y0, x1, y1), t))
    return out


def expand_bbox_with_panel_labels(page: fitz.Page, bbox: fitz.Rect, pad: float = 8.0, near_px: float = 60.0) -> fitz.Rect:
    """
    Panel labels (a/b/c/...) are often separate text blocks near the figure.
    Expand bbox to include nearby single-letter blocks.
    """
    blocks = page.get_text("blocks")
    expanded = fitz.Rect(bbox)
    for b in blocks:
        x0, y0, x1, y1, txt = b[0], b[1], b[2], b[3], b[4]
        if not txt:
            continue
        t = txt.strip()
        m = PANEL_LABEL_RE.match(t)
        if not m:
            continue

        r = fitz.Rect(x0, y0, x1, y1)

        # If the label block is close to bbox (within near_px) expand.
        close_h = (r.x1 >= bbox.x0 - near_px) and (r.x0 <= bbox.x1 + near_px)
        close_v = (r.y1 >= bbox.y0 - near_px) and (r.y0 <= bbox.y1 + near_px)
        if close_h and close_v:
            expanded |= r

    # small padding
    expanded.x0 -= pad
    expanded.y0 -= pad
    expanded.x1 += pad
    expanded.y1 += pad
    return expanded


def build_visual_bbox(page: fitz.Page, min_draw_area_frac: float = 0.0005) -> Tuple[Optional[fitz.Rect], Dict[str, Any]]:
    """
    Build a bbox from:
      - image rects (xref placements)
      - vector drawing rects (plots often are vector)
    Returns (bbox, debug_info)
    """
    page_rect = page.rect
    page_area = page_rect.get_area()

    rects: List[fitz.Rect] = []

    # Image rects
    img_rects_all: List[fitz.Rect] = []
    for img in page.get_images(full=True):
        xref = img[0]
        rlist = page.get_image_rects(xref)
        for r in rlist:
            img_rects_all.append(fitz.Rect(r))
    rects.extend(img_rects_all)

    # Drawing rects (vector)
    draw_rects: List[fitz.Rect] = []
    try:
        drawings = page.get_drawings()
        for d in drawings:
            r = d.get("rect", None)
            if not r:
                continue
            rr = fitz.Rect(r)
            # ignore tiny rectangles (noise)
            if rr.get_area() < page_area * min_draw_area_frac:
                continue
            draw_rects.append(rr)
    except Exception:
        drawings = []

    rects.extend(draw_rects)

    bbox = rect_union(rects)
    bbox = clamp_rect(bbox, page_rect) if bbox else None

    union_ratio = (bbox.get_area() / page_area) if bbox else 0.0

    debug = {
        "img_rects": len(img_rects_all),
        "drawings_total": len(drawings) if isinstance(drawings, list) else 0,
        "draw_rects_kept": len(draw_rects),
        "union_ratio": union_ratio,
    }
    return bbox, debug


def is_front_matter(page_text: str) -> bool:
    """
    Heuristic: title/cover pages often contain doi/received/accepted/published and
    *do not* contain figure markers.
    """
    t = (page_text or "").lower()
    has_fig = ("fig." in t) or ("figure" in t)
    front_terms = [
        "doi.org",
        "received:",
        "accepted:",
        "published online",
        "nature communications",
        "article",
        "check for updates",
    ]
    has_front = any(term in t for term in front_terms)
    return has_front and (not has_fig)


# -----------------------------
# Main
# -----------------------------
def ingest_paper(
    pdf_path: str,
    figures_output_dir: Optional[str] = None,
    *,
    skip_first_page: bool = True,
    zoom: float = 3.0,
    union_ratio_threshold: float = 0.08,  # figure-like if bbox covers >= 8% of page
    remove_caption: bool = True,
    verbose: bool = True,
) -> IngestedPaper:
    """
    Extract page text and cropped figure images.

    - Skips first page (often title/front matter)
    - Finds "figure-like" pages using union of image rects + vector drawing rects
    - Crops to the visual bbox
    - Expands bbox to include panel labels (a/b/c/...)
    - Optionally removes caption area (Fig./Figure blocks)
    """

    if figures_output_dir is None:
        figures_output_dir = os.path.join(os.getcwd(), "extracted_figures")
    os.makedirs(figures_output_dir, exist_ok=True)

    stem = safe_stem_from_path(pdf_path)
    paper_fig_dir = os.path.join(figures_output_dir, stem)
    os.makedirs(paper_fig_dir, exist_ok=True)

    if verbose:
        print(f"[ingest_paper] PDF: {pdf_path}")
        print(f"[ingest_paper] Figures dir: {paper_fig_dir}")

    doc = fitz.open(pdf_path)

    pages: List[PageContent] = []
    figures: List[FigureContent] = []
    sidecar: List[Dict[str, Any]] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]

        text = page.get_text("text") or ""
        pages.append(PageContent(page_number=page_idx, text=text))

        # 1) Skip first page (requested)
        if skip_first_page and page_idx == 0:
            if verbose:
                print(f"[ingest_paper] Page {page_idx + 1}: skipped (first page)")
            continue

        # 2) Skip front matter pages (extra safety)
        if is_front_matter(text):
            if verbose:
                print(f"[ingest_paper] Page {page_idx + 1}: skipped (front matter)")
            continue

        bbox, dbg = build_visual_bbox(page)
        fig_like = bbox is not None and dbg["union_ratio"] >= union_ratio_threshold

        if verbose:
            print(
                f"[ingest_paper] Page {page_idx + 1}: "
                f"img_rects={dbg['img_rects']}, drawings={dbg['drawings_total']}, "
                f"union_ratio={dbg['union_ratio']:.3f}, fig_like={fig_like}"
            )

        if not fig_like or bbox is None:
            continue

        # 3) Expand bbox to include nearby panel labels (a/b/c/…)
        bbox2 = expand_bbox_with_panel_labels(page, bbox)

        # 4) Remove caption area if present (caption is usually below figure)
        caption_blocks = find_caption_blocks(page)
        caption_used = None
        if remove_caption and caption_blocks:
            # pick the lowest caption block on the page (largest y0)
            caption_blocks_sorted = sorted(caption_blocks, key=lambda x: x[0].y0, reverse=True)
            cap_rect, cap_text = caption_blocks_sorted[0]
            caption_used = cap_text

            # If caption is below the figure bbox, crop figure to end just above caption.
            # Small padding to avoid cutting off bottom axis labels.
            if cap_rect.y0 > bbox2.y0 and cap_rect.y0 < page.rect.y1:
                bbox2.y1 = min(bbox2.y1, cap_rect.y0 - 8.0)

        bbox2 = clamp_rect(bbox2, page.rect)

        # 5) Render cropped figure
        pix = render_clip(page, bbox2, zoom=zoom)

        # 6) Filter out blank/near-blank crops
        if is_low_variance_pix(pix, std_thresh=2.0):
            if verbose:
                print(f"  -> skipped: low-variance/blank crop on page {page_idx + 1}")
            continue

        # 7) Save
        out_name = f"{stem}_p{page_idx + 1:02d}_fig.png"
        out_path = os.path.join(paper_fig_dir, out_name)
        pix.save(out_path)

        figures.append(
            FigureContent(
                page_number=page_idx,
                image_index=0,
                image_path=out_path,
            )
        )

        sidecar.append(
            {
                "paper": stem,
                "pdf_path": os.path.abspath(pdf_path),
                "page_number_0based": page_idx,
                "image_path": out_path,
                "bbox": [bbox2.x0, bbox2.y0, bbox2.x1, bbox2.y1],
                "union_ratio": dbg["union_ratio"],
                "caption_text": caption_used,
            }
        )

        if verbose:
            cap_flag = "yes" if caption_used else "no"
            print(f"  -> saved: {out_name} (caption_found={cap_flag})")

        pix = None

    # Write sidecar metadata
    try:
        meta_path = os.path.join(paper_fig_dir, f"{stem}_figures.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2)
        if verbose:
            print(f"[ingest_paper] Wrote metadata: {meta_path}")
    except Exception as e:
        if verbose:
            print(f"[WARN] Could not write figures metadata JSON: {e}")

    doc.close()

    if verbose:
        print(f"[ingest_paper] Done. Pages={len(pages)} Figures={len(figures)}")

    return IngestedPaper(
        source_path=os.path.abspath(pdf_path),
        num_pages=len(pages),
        pages=pages,
        figures=figures,
    )
