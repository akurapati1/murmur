"""Generators for small synthetic PDFs used throughout the tests."""
from __future__ import annotations

from pathlib import Path

import pymupdf

SENTENCE = "The old lighthouse keeper walked along the shore each morning, counting the gulls and listening to the tide."
BODY = " ".join([SENTENCE] * 14)


def _image() -> "pymupdf.Pixmap":
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 300, 200), False)
    pix.set_rect(pix.irect, (40, 90, 200))
    return pix


def make_novel(path: Path, chapters: int = 4, pages_per_chapter: int = 3, toc: bool = True,
               back_matter: bool = False, figure_pages: int = 0) -> Path:
    """A prose book with running headers, page numbers, chapter headings and (optionally) an outline."""
    doc = pymupdf.open()
    entries = []
    pageno = 0
    figures_left = figure_pages
    for c in range(1, chapters + 1):
        for p in range(pages_per_chapter):
            page = doc.new_page(width=420, height=640)
            page.insert_text((60, 30), "THE LIGHTHOUSE KEEPER", fontsize=9)
            page.insert_text((200, 620), str(pageno + 1), fontsize=9)
            y = 70
            if p == 0:
                page.insert_text((60, y), f"Chapter {c}", fontsize=18)
                y += 40
                entries.append([1, f"Chapter {c}", pageno + 1])
            if figures_left > 0 and p == 1:
                page.insert_image(pymupdf.Rect(40, y, 380, y + 280), pixmap=_image(), keep_proportion=False)
                page.insert_textbox(pymupdf.Rect(50, y + 290, 370, 600), "A short note under the picture.", fontsize=11)
                figures_left -= 1
            else:
                page.insert_textbox(pymupdf.Rect(50, y, 370, 590), BODY, fontsize=11)
            pageno += 1
    if back_matter:
        page = doc.new_page(width=420, height=640)
        page.insert_text((60, 70), "Index", fontsize=18)
        page.insert_textbox(pymupdf.Rect(50, 100, 370, 300), "apples 12, 40 pears 13 plums 90 " * 4, fontsize=11)
        entries.append([1, "Index", pageno + 1])
    if toc:
        doc.set_toc(entries)
    doc.set_metadata({"title": "The Lighthouse Keeper", "author": "A. Writer"})
    doc.save(path)
    doc.close()
    return path


def make_technical(path: Path, pages: int = 12) -> Path:
    """Every page is dominated by a large figure with a numbered caption."""
    doc = pymupdf.open()
    for i in range(pages):
        page = doc.new_page(width=420, height=640)
        page.insert_text((60, 50), f"Chapter {i + 1}", fontsize=16)
        page.insert_image(pymupdf.Rect(40, 80, 380, 330), pixmap=_image(), keep_proportion=False)
        page.insert_textbox(
            pymupdf.Rect(50, 335, 370, 360),
            f"Figure {i + 1}. Block diagram of the system showing data flow.",
            fontsize=10,
        )
        page.insert_textbox(pymupdf.Rect(50, 370, 370, 600), " ".join([SENTENCE] * 6), fontsize=10)
    doc.save(path)
    doc.close()
    return path


def make_scanned(path: Path, pages: int = 6) -> Path:
    """Pages that are only photos: no selectable text."""
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page(width=420, height=640)
        page.insert_image(pymupdf.Rect(0, 0, 420, 640), pixmap=_image(), keep_proportion=False)
    doc.save(path)
    doc.close()
    return path


def make_slides(path: Path, pages: int = 10) -> Path:
    doc = pymupdf.open()
    for i in range(pages):
        page = doc.new_page(width=800, height=450)
        page.insert_text((60, 120), f"Quarterly review {i + 1}", fontsize=32)
        page.insert_text((60, 200), "Revenue is up and costs are down.", fontsize=20)
    doc.save(path)
    doc.close()
    return path


def make_hyphenated(path: Path) -> Path:
    doc = pymupdf.open()
    page = doc.new_page(width=420, height=640)
    lines = [
        "The keeper was an extra-",
        "ordinary person who loved the sea and every",
        "gull that ever landed on the rail.",
    ]
    for i, line in enumerate(lines):
        page.insert_text((50, 100 + i * 14), line, fontsize=11)
    page.insert_textbox(pymupdf.Rect(50, 200, 370, 500), BODY * 3, fontsize=11)
    doc.save(path)
    doc.close()
    return path


def make_scanned_text(path: Path, pages: int = 4) -> Path:
    """A book whose pages are photographs of text: no selectable text layer, like a real scan."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        src = make_novel(Path(tmp) / "src.pdf", chapters=1, pages_per_chapter=pages, toc=False, back_matter=False)
        source = pymupdf.open(src)
        doc = pymupdf.open()
        for page in source:
            pix = page.get_pixmap(dpi=170)
            out = doc.new_page(width=page.rect.width, height=page.rect.height)
            out.insert_image(out.rect, pixmap=pix, keep_proportion=False)
        source.close()
        doc.save(path)
        doc.close()
    return path
