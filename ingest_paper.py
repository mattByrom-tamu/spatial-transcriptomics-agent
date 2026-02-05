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
PANEL_LABEL_RE = re.compile(r"(?i)^\s*\(?\s*([a-h])\s*\)?\s*$")  # a..h


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
    if rr.x1 <= rr.x0 + 1:
        rr.x1 = rr.x0 + 1
    if rr.y1 <= rr.y0 + 1:
        rr.y1 = rr.y0 + 1
    return rr


def is_low_variance_pix(pix: fitz.Pixmap, std_thresh: float = 2.0) -> bool:
    try:
        samples = pix.samples
        if not samples:
            return True
        step = max(1, len(samples) // 20000)
        vals = []
        n = pix.n
        for i in range(0, len(samples), step * n):
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
    blocks = page.get_text("blocks")
    out: List[Tuple[fitz.Rect, str]] = []
    for b in blocks:
        x0, y0, x1, y1, txt = b[0], b[1], b[2], b[3], b[4]
        if not txt:
            continue
        t = txt.strip()
        if CAPTION_RE.match(t):
            out.append((fitz.Rect(x0, y0, x1, y1), t))
    return out


def build_visual_bbox(page: fitz.Page, min_draw_area_frac: float = 0.0005) -> Tuple[Optional[fitz.Rect], Dict[str, Any]]:
    page_rect = page.rect
    page_area = page_rect.get_area()

    rects: List[fitz.Rect] = []

    # raster placements (most reliable "figure" signal)
    img_rects_all: List[fitz.Rect] = []
    for img in page.get_images(full=True):
        xref = img[0]
        for r in page.get_image_rects(xref):
            img_rects_all.append(fitz.Rect(r))
    rects.extend(img_rects_all)

    # vector drawings (can be polluted by outlined text in some PDFs)
    draw_rects: List[fitz.Rect] = []
    try:
        drawings = page.get_drawings()
        for d in drawings:
            r = d.get("rect", None)
            if not r:
                continue
            rr = fitz.Rect(r)
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


def text_coverage_frac(
    page: fitz.Page,
    *,
    min_chars: int = 60,
    ignore_top_frac: float = 0.08,
    ignore_bottom_frac: float = 0.08,
) -> float:
    """
    Fraction of page area covered by "real" text blocks.
    Helps reject pure text pages (including PDFs where text is outlined and inflates drawings).
    """
    pr = page.rect
    page_area = pr.get_area()
    if page_area <= 0:
        return 0.0

    top_y = pr.y0 + pr.height * ignore_top_frac
    bot_y = pr.y1 - pr.height * ignore_bottom_frac

    total = 0.0
    blocks = page.get_text("blocks")
    for b in blocks:
        x0, y0, x1, y1, txt = b[0], b[1], b[2], b[3], b[4]
        if not txt:
            continue
        r = fitz.Rect(x0, y0, x1, y1)

        # ignore headers/footers
        if r.y1 <= top_y or r.y0 >= bot_y:
            continue

        t = txt.strip()
        if len(t) < min_chars:
            continue

        # don't count captions/panel labels as "body text"
        if CAPTION_RE.match(t):
            continue
        if PANEL_LABEL_RE.match(t):
            continue

        total += r.get_area()

    return max(0.0, min(1.0, total / page_area))


# -----------------------------
# Main
# -----------------------------
def ingest_paper(
    pdf_path: str,
    figures_output_dir: Optional[str] = None,
    *,
    skip_first_page: bool = True,
    zoom: float = 3.0,
    union_ratio_threshold: float = 0.03,     # still used as a weak signal
    save_full_page: bool = True,
    # NEW: remove "mostly text" pages
    text_coverage_threshold: float = 0.35,
    verbose: bool = True,
) -> IngestedPaper:

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

        if skip_first_page and page_idx == 0:
            if verbose:
                print(f"[ingest_paper] Page {page_idx + 1}: skipped (first page)")
            continue

        if is_front_matter(text):
            if verbose:
                print(f"[ingest_paper] Page {page_idx + 1}: skipped (front matter)")
            continue

        caption_blocks = find_caption_blocks(page)
        has_caption = len(caption_blocks) > 0
        caption_used = None
        if has_caption:
            cap_rect, cap_text = sorted(caption_blocks, key=lambda x: x[0].y0, reverse=True)[0]
            caption_used = cap_text

        bbox, dbg = build_visual_bbox(page)

        # NEW: body-text rejection
        tfrac = text_coverage_frac(page)

        # High-recall fig-like
        fig_like = (
            has_caption
            or (dbg["img_rects"] > 0)
            or (bbox is not None and dbg["union_ratio"] >= union_ratio_threshold)
        )

        # NEW: if it’s mostly body text and has no raster images, drop it
        if fig_like and dbg["img_rects"] == 0 and tfrac >= text_coverage_threshold:
            fig_like = False

        if verbose:
            print(
                f"[ingest_paper] Page {page_idx + 1}: "
                f"img_rects={dbg['img_rects']}, drawings={dbg['drawings_total']}, "
                f"union_ratio={dbg['union_ratio']:.3f}, caption={has_caption}, "
                f"text_cov={tfrac:.3f}, fig_like={fig_like}"
            )

        if not fig_like:
            continue

        if not save_full_page:
            continue

        pix = render_clip(page, page.rect, zoom=zoom)
        if is_low_variance_pix(pix, std_thresh=2.0):
            if verbose:
                print(f"  -> skipped: low-variance/blank render on page {page_idx + 1}")
            continue

        out_name = f"{stem}_p{page_idx + 1:02d}_content.png"
        out_path = os.path.join(paper_fig_dir, out_name)
        pix.save(out_path)

        figures.append(FigureContent(page_number=page_idx, image_index=0, image_path=out_path))

        sidecar.append(
            {
                "paper": stem,
                "pdf_path": os.path.abspath(pdf_path),
                "page_number_0based": page_idx,
                "image_path": out_path,
                "method": "full_page",
                "union_ratio": dbg["union_ratio"],
                "img_rects": dbg["img_rects"],
                "text_coverage_frac": tfrac,
                "caption_text": caption_used,
            }
        )

        if verbose:
            cap_flag = "yes" if caption_used else "no"
            print(f"  -> saved: {out_name} (caption_found={cap_flag})")

        pix = None

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
        print(f"[ingest_paper] Done. Pages={len(pages)} Figures(saved_pages)={len(figures)}")

    return IngestedPaper(
        source_path=os.path.abspath(pdf_path),
        num_pages=len(pages),
        pages=pages,
        figures=figures,
    )
