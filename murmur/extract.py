"""Step 3: turn a PDF into clean, chapter-structured narration text."""
from __future__ import annotations

import re
import shutil
import threading
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pymupdf

from .chunker import SCENE_BREAK, is_scene_break
from .models import Book, Chapter

ProgressCb = Callable[[float, str], None]

_PAGE_NUM_RE = re.compile(r"^\s*(page\s*)?(\d{1,4}|[ivxlcdm]{1,7})\s*$", re.IGNORECASE)
_DOT_LEADER_RE = re.compile(r"\.{4,}\s*\d+\s*$|(\s\.){4,}")
_URL_ONLY_RE = re.compile(r"^\s*(https?://|www\.)\S+\s*$", re.IGNORECASE)
_CAPTION_RE = re.compile(
    r"^\s*(Figure|Fig\.|Table|Chart|Diagram|Listing|Exhibit|Plate)\s+\d+([.\-:]\d+)?\b", re.IGNORECASE
)
_HEADING_RE = re.compile(
    r"^(chapter|part|book|section)\s+([0-9]+|[ivxlcdm]+|[a-z]+)\b[\s.:\-–—]*$"
    r"|^(prologue|epilogue|preface|introduction|foreword|afterword|interlude)\s*$"
    r"|^(chapter|part)\s+([0-9]+|[ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten)\b.{0,60}$",
    re.IGNORECASE,
)
_BARE_NUMBER_HEADING_RE = re.compile(r"^(\d{1,3}|[IVXLC]{1,6})\.?$")
_SKIP_TITLES = {
    "index", "bibliography", "references", "endnotes", "notes", "copyright", "contents",
    "table of contents", "about the author", "about the authors", "also by", "colophon",
    "credits", "permissions", "list of figures", "list of tables", "further reading",
}
_TERMINAL_RE = re.compile(r"[.!?:;\"'”’)\]…]$")


class Cancelled(Exception):
    """Raised when a job is cancelled mid-flight."""


class OCRUnavailable(RuntimeError):
    pass


def _check(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled()


def _norm_margin(text: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", text.strip().lower()))


def _clean_block_text(text: str) -> str:
    text = text.replace("­", "")  # soft hyphen
    text = re.sub(r"(\w)-\s*\n\s*([a-z])", r"\1\2", text)  # re-join hyphenated words
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"[ \t ]+", " ", text)
    return text.strip()


def _page_blocks(page: "pymupdf.Page", ocr: bool) -> list[tuple[float, float, float, float, str]]:
    textpage = None
    if ocr:
        try:
            textpage = page.get_textpage_ocr(dpi=200, full=True)
        except Exception as exc:  # tesseract or language data missing
            raise OCRUnavailable(f"OCR failed: {exc}") from exc
    raw = page.get_text("blocks", sort=True, textpage=textpage) if textpage else page.get_text("blocks", sort=True)
    out = []
    for b in raw:
        if b[6] != 0:  # skip image blocks
            continue
        out.append((b[0], b[1], b[2], b[3], b[4]))
    return out


def _merge_paragraphs(paras: list[str]) -> list[str]:
    merged: list[str] = []
    for p in paras:
        if merged and not is_scene_break(merged[-1]) and not _TERMINAL_RE.search(merged[-1]) and p[:1].islower():
            merged[-1] = f"{merged[-1]} {p}"
        else:
            merged.append(p)
    return merged


def _is_noise(p: str, skip_captions: bool) -> bool:
    if is_scene_break(p):
        return False  # kept: the narrator pauses here
    if len(p) < 2 or not re.search(r"[A-Za-z]", p):
        return True
    if _DOT_LEADER_RE.search(p) or _URL_ONLY_RE.match(p):
        return True
    if skip_captions and _CAPTION_RE.match(p) and len(p) < 300:
        return True
    return False


def _default_title(stem: str, meta_title: str) -> str:
    if meta_title:
        return meta_title
    return re.sub(r"[_\-]+", " ", stem).strip().title() or "Untitled"


def _split_by_words(paragraphs: list[str], target_words: int = 7000) -> list[Chapter]:
    chapters: list[Chapter] = []
    current: list[str] = []
    count = 0
    for p in paragraphs:
        current.append(p)
        count += len(p.split())
        if count >= target_words:
            chapters.append(Chapter(f"Part {len(chapters) + 1}", current))
            current, count = [], 0
    if current:
        if chapters and count < target_words * 0.3:
            chapters[-1].paragraphs.extend(current)
        else:
            chapters.append(Chapter(f"Part {len(chapters) + 1}", current))
    return chapters


def _heading_chapters(paragraphs: list[str]) -> list[Chapter]:
    """Fallback chapter detection when the PDF has no outline."""
    chapters: list[Chapter] = []
    current_title, current = "Opening", []
    i = 0
    while i < len(paragraphs):
        p = paragraphs[i]
        is_heading = len(p) <= 80 and (
            _HEADING_RE.match(p.strip()) is not None
            or (
                _BARE_NUMBER_HEADING_RE.match(p.strip()) is not None
                and i + 1 < len(paragraphs)
                and len(paragraphs[i + 1]) <= 60
                and not _TERMINAL_RE.search(paragraphs[i + 1])
            )
        )
        if is_heading:
            if current:
                chapters.append(Chapter(current_title, current))
            title = p.strip().rstrip(".:–—- ")
            if re.fullmatch(r"\d{1,3}|[IVXLC]{1,6}", title):
                title = f"Chapter {title}"
            if (
                i + 1 < len(paragraphs)
                and len(paragraphs[i + 1]) <= 60
                and not _TERMINAL_RE.search(paragraphs[i + 1])
                and not _HEADING_RE.match(paragraphs[i + 1].strip())
            ):
                title = f"{title}: {paragraphs[i + 1].strip()}"
                i += 1
            current_title, current = title, []
        else:
            current.append(p)
        i += 1
    if current:
        chapters.append(Chapter(current_title, current))
    return chapters


def _toc_entries(doc: "pymupdf.Document") -> list[tuple[str, int]]:
    toc = [t for t in doc.get_toc(simple=True) if 1 <= t[2] <= doc.page_count]
    if not toc:
        return []
    by_level: dict[int, list[tuple[str, int]]] = {}
    for lvl, title, page in toc:
        by_level.setdefault(lvl, []).append((title.strip(), page - 1))
    level = min(by_level)
    if len(by_level[level]) < 4 and (level + 1) in by_level and len(by_level[level + 1]) > len(by_level[level]):
        level += 1
    entries = sorted(by_level[level], key=lambda e: e[1])
    deduped: list[tuple[str, int]] = []
    for title, page in entries:
        if deduped and deduped[-1][1] == page:
            continue
        deduped.append((title, page))
    return deduped


def _strip_leading_title(paragraphs: list[str], title: str) -> list[str]:
    if paragraphs and paragraphs[0].strip().lower().rstrip(".:") == title.strip().lower().rstrip(".:"):
        return paragraphs[1:]
    return paragraphs


def extract_book(
    path: Path,
    *,
    ocr: bool = False,
    skip_back_matter: bool = True,
    skip_captions: bool = True,
    title_hint: str | None = None,
    progress: ProgressCb | None = None,
    cancel: threading.Event | None = None,
) -> Book:
    path = Path(path)
    if ocr and shutil.which("tesseract") is None:
        raise OCRUnavailable("This PDF needs OCR, but Tesseract isn't installed on this machine.")

    doc = pymupdf.open(path)
    try:
        meta = doc.metadata or {}
        title = _default_title(title_hint or path.stem, (meta.get("title") or "").strip())
        author = (meta.get("author") or "").strip()
        n = doc.page_count

        # Pass 1: raw blocks per page, plus margin text for header/footer detection.
        pages: list[list[tuple[bool, str]]] = []  # (in_margin, text)
        margin_counts: Counter[str] = Counter()
        for i in range(n):
            _check(cancel)
            page = doc[i]
            h = page.rect.height
            blocks = []
            if ocr and len((page.get_text("text") or "").strip()) < 25:
                raw = _page_blocks(page, ocr=True)
            else:
                raw = _page_blocks(page, ocr=False)
            seen_here: set[str] = set()
            for x0, y0, x1, y1, text in raw:
                clean = _clean_block_text(text)
                if not clean:
                    continue
                in_margin = y1 <= h * 0.09 or y0 >= h * 0.91
                blocks.append((in_margin, clean))
                if in_margin:
                    key = _norm_margin(clean)
                    if key not in seen_here:
                        margin_counts[key] += 1
                        seen_here.add(key)
            pages.append(blocks)
            if progress and (i % 10 == 0 or i == n - 1):
                progress((i + 1) / n * 0.8, f"Reading page {i + 1} of {n}")

        threshold = max(3, int(n * 0.2))
        repeated = {k for k, c in margin_counts.items() if c >= threshold}

        # Pass 2: drop margins that repeat, page numbers and noise; build paragraphs per page.
        page_paras: list[list[str]] = []
        for blocks in pages:
            paras = []
            for in_margin, text in blocks:
                if in_margin and (_norm_margin(text) in repeated or _PAGE_NUM_RE.match(text)):
                    continue
                if _is_noise(text, skip_captions):
                    continue
                paras.append(SCENE_BREAK if is_scene_break(text) else text)
            page_paras.append(paras)

        # Chapters
        chapters: list[Chapter] = []
        entries = _toc_entries(doc)
        if len(entries) >= 3:
            first_page = entries[0][1]
            front = _merge_paragraphs([p for pg in page_paras[:first_page] for p in pg])
            if sum(len(p.split()) for p in front) >= 600:
                chapters.append(Chapter("Opening", front))
            for idx, (ctitle, start) in enumerate(entries):
                end = entries[idx + 1][1] if idx + 1 < len(entries) else n
                paras = _merge_paragraphs([p for pg in page_paras[start:end] for p in pg])
                chapters.append(Chapter(ctitle, _strip_leading_title(paras, ctitle)))
        else:
            flat = _merge_paragraphs([p for pg in page_paras for p in pg])
            found = _heading_chapters(flat)
            if len(found) >= 2:
                chapters = [c for c in found if c.paragraphs or c.title != "Opening"]
            else:
                chapters = _split_by_words(flat)

        chapters = [c for c in chapters if c.word_count > 0]
        total = sum(c.word_count for c in chapters)
        if skip_back_matter and total:
            kept = [c for c in chapters if c.title.strip().lower().rstrip(".:") not in _SKIP_TITLES]
            if sum(c.word_count for c in kept) >= total * 0.6:
                chapters = kept
        if progress:
            progress(1.0, f"Found {len(chapters)} chapters")
        return Book(title=title, author=author, chapters=chapters)
    finally:
        doc.close()
