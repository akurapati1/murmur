"""Step 2: can this PDF be listened to without losing its meaning?

Looks at a sample of pages and measures how much of the content lives in figures
(big images, dense vector diagrams, captioned figures) rather than in prose.
A few graphs, tables or equations are fine; a book built around illustrations is not.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from .models import AnalysisReport, Verdict

WORDS_PER_HOUR = 9300  # typical audiobook narration pace

_CAPTION_RE = re.compile(
    r"^\s*(Figure|Fig\.|Table|Chart|Diagram|Listing|Exhibit|Plate)\s+\d+([.\-:]\d+)?",
    re.IGNORECASE | re.MULTILINE,
)
_MATH_CHARS = set("∑∫√∞≈≠≤≥±∂∇∈∉⊂⊆∪∩∀∃→⇒×÷∝∴∵")
_CODE_END_RE = re.compile(r"(;|\{|\})\s*$")
_CODE_HINT_RE = re.compile(r"(==|!=|->|=>|\bdef \w+\(|\bfunction\b|\breturn\b.*;|#include|\bimport \w+)")


@dataclass
class PageStats:
    chars: int
    words: int
    image_ratio: float
    drawing_count: int
    drawing_ratio: float
    captions: int
    math_chars: int
    code_lines: int
    lines: int
    landscape: bool

    @property
    def scanned(self) -> bool:
        return self.chars < 25 and self.image_ratio >= 0.6

    @property
    def figure_page(self) -> bool:
        if self.scanned or self.chars == 0:
            return False
        big_image = self.image_ratio >= 0.30
        big_diagram = self.drawing_count >= 60 and self.drawing_ratio >= 0.30
        return big_image or big_diagram


def _is_math(ch: str) -> bool:
    return ch in _MATH_CHARS or "Ͱ" <= ch <= "Ͽ"


def _page_stats(page: "pymupdf.Page") -> PageStats:
    rect = page.rect
    area = max(rect.width * rect.height, 1.0)
    text = page.get_text("text") or ""

    image_area = 0.0
    try:
        for info in page.get_image_info():
            r = pymupdf.Rect(info["bbox"]) & rect
            if r.is_empty:
                continue
            a = r.width * r.height
            if a / area >= 0.02:  # ignore ornaments and icons
                image_area += a
    except Exception:
        pass

    drawing_count, drawing_ratio = 0, 0.0
    try:
        drawings = page.get_drawings()
        drawing_count = len(drawings)
        if drawing_count:
            union = pymupdf.Rect()
            for d in drawings:
                union |= d["rect"]
            union &= rect
            drawing_ratio = 0.0 if union.is_empty else (union.width * union.height) / area
    except Exception:
        pass

    lines = [ln for ln in text.splitlines() if ln.strip()]
    code_lines = sum(1 for ln in lines if _CODE_END_RE.search(ln) or _CODE_HINT_RE.search(ln))
    return PageStats(
        chars=len(text.strip()),
        words=len(text.split()),
        image_ratio=min(image_area / area, 1.0),
        drawing_count=drawing_count,
        drawing_ratio=drawing_ratio,
        captions=len(_CAPTION_RE.findall(text)),
        math_chars=sum(1 for c in text if _is_math(c)),
        code_lines=code_lines,
        lines=len(lines),
        landscape=rect.width > rect.height * 1.3,
    )


def _sample_indices(page_count: int, n: int) -> list[int]:
    if page_count <= n:
        return list(range(page_count))
    step = page_count / n
    return sorted({min(int(i * step + step / 2), page_count - 1) for i in range(n)})


def ocr_available() -> bool:
    import shutil

    return shutil.which("tesseract") is not None


def analyze_pdf(path: Path, sample_pages: int = 40) -> AnalysisReport:
    doc = pymupdf.open(path)
    try:
        page_count = doc.page_count
        indices = _sample_indices(page_count, sample_pages)
        stats = [_page_stats(doc[i]) for i in indices]
    finally:
        doc.close()

    n = len(stats)
    scanned = sum(1 for s in stats if s.scanned)
    text_pages = [s for s in stats if not s.scanned]
    n_text = max(len(text_pages), 1)

    scanned_ratio = scanned / n
    figure_ratio = sum(1 for s in text_pages if s.figure_page) / n_text
    caption_density = sum(s.captions for s in text_pages) / n_text
    total_chars = max(sum(s.chars for s in text_pages), 1)
    math_density = sum(s.math_chars for s in text_pages) / total_chars
    total_lines = max(sum(s.lines for s in text_pages), 1)
    code_ratio = sum(s.code_lines for s in text_pages) / total_lines
    landscape_ratio = sum(1 for s in stats if s.landscape) / n
    words_per_page = statistics.median([s.words for s in text_pages]) if text_pages else 0.0

    visual_score = round(
        min(1.0, figure_ratio / 0.35) * 0.55
        + min(1.0, caption_density / 0.5) * 0.20
        + min(1.0, math_density / 0.04) * 0.10
        + min(1.0, code_ratio / 0.25) * 0.15,
        3,
    )

    avg_words = (sum(s.words for s in text_pages) / n_text) if text_pages else 0
    est_words = int(avg_words * page_count * (1 - scanned_ratio)) if scanned_ratio < 0.6 else 0
    reasons: list[str] = []
    needs_ocr = False

    if scanned_ratio >= 0.6:
        verdict = Verdict.SCANNED
        needs_ocr = True
        reasons.append(
            f"{round(scanned_ratio * 100)}% of the sampled pages are page photos with no "
            "selectable text, so the words have to be recognised with OCR first."
        )
    elif landscape_ratio >= 0.7 and words_per_page < 120:
        verdict = Verdict.NOT_A_BOOK
        reasons.append("The pages are wide and hold little text, which looks like a slide deck.")
    elif words_per_page < 20:
        verdict = Verdict.NOT_A_BOOK
        reasons.append("There is very little text per page, so this doesn't look like a book.")
    elif figure_ratio >= 0.30 or visual_score >= 0.6:
        verdict = Verdict.NEEDS_VISUALS
        reasons.append(
            f"About {round(figure_ratio * 100)}% of the pages are dominated by figures or diagrams, "
            "so the pictures carry much of the meaning."
        )
    elif figure_ratio >= 0.12 or visual_score >= 0.35:
        verdict = Verdict.BORDERLINE
        reasons.append(
            "Some pages rely on figures or diagrams. The audiobook will skip them, "
            "but the prose should still make sense."
        )
    else:
        verdict = Verdict.AUDIOBOOK_FRIENDLY
        reasons.append("Mostly continuous prose, which works well as an audiobook.")

    if caption_density >= 0.5 and verdict != Verdict.NEEDS_VISUALS:
        reasons.append("Many pages have numbered figure or table captions.")
    if math_density >= 0.02:
        reasons.append("Equations are frequent. They are read as plain text, which can sound odd.")
    if code_ratio >= 0.15:
        reasons.append("Source code listings appear throughout and won't narrate well.")
    if scanned_ratio and verdict != Verdict.SCANNED:
        reasons.append("A few pages are images without text and will be skipped.")

    return AnalysisReport(
        verdict=verdict,
        visual_score=visual_score,
        page_count=page_count,
        sampled_pages=n,
        estimated_words=est_words,
        estimated_hours=round(est_words / WORDS_PER_HOUR, 1),
        reasons=reasons,
        stats={
            "figure_page_ratio": round(figure_ratio, 3),
            "caption_density": round(caption_density, 3),
            "math_density": round(math_density, 4),
            "code_ratio": round(code_ratio, 3),
            "scanned_ratio": round(scanned_ratio, 3),
            "median_words_per_page": words_per_page,
        },
        needs_ocr=needs_ocr,
    )
