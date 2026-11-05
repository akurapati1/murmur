"""Step 1: is the uploaded file really a readable PDF?"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf

PDF_MAGIC = b"%PDF-"


@dataclass
class ValidationResult:
    ok: bool
    code: str = ""
    reason: str = ""
    page_count: int = 0
    title: str = ""
    author: str = ""


def _head(path: Path, n: int = 1024) -> bytes:
    with open(path, "rb") as f:
        return f.read(n)


def sniff_pdf(path: Path) -> bool:
    """Content-based check (never trusts the extension or MIME type)."""
    return PDF_MAGIC in _head(path)


def describe_non_pdf(path: Path) -> str:
    head = _head(path, 16)
    if head.startswith(b"PK\x03\x04"):
        what = "a ZIP-based file (such as an EPUB or DOCX)"
    elif head.startswith(b"\x89PNG") or head.startswith(b"\xff\xd8"):
        what = "an image"
    else:
        what = "not a PDF"
    return f"This file is {what}. Murmur reads PDF books, so please upload a .pdf file."


def validate_pdf(path: Path) -> ValidationResult:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return ValidationResult(False, "empty", "The file is empty.")
    if not sniff_pdf(path):
        return ValidationResult(False, "not_pdf", describe_non_pdf(path))
    try:
        doc = pymupdf.open(path)
    except Exception:
        return ValidationResult(
            False, "corrupt", "This PDF is damaged and could not be opened. Try re-exporting it."
        )
    try:
        if doc.needs_pass and not doc.authenticate(""):
            return ValidationResult(
                False, "encrypted",
                "This PDF is password-protected. Remove the password and upload it again.",
            )
        if doc.page_count == 0:
            return ValidationResult(False, "empty", "This PDF has no pages.")
        meta = doc.metadata or {}
        return ValidationResult(
            True, page_count=doc.page_count,
            title=(meta.get("title") or "").strip(),
            author=(meta.get("author") or "").strip(),
        )
    finally:
        doc.close()
