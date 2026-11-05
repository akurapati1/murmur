"""Prepare text for speech: normalise, split into sentences, pack into engine-sized chunks."""
from __future__ import annotations

import re

from .models import Chapter, Chunk

SENTENCE_PAUSE = 0.15
PARAGRAPH_PAUSE = 0.5
HEADING_PAUSE = 0.9

_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "no",
    "fig", "inc", "ltd", "co", "mt", "gen", "col", "capt", "lt", "sgt", "rev", "hon",
}
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_FOOTNOTE_RE = re.compile(r"\[\d{1,3}\]|\[\w{1,2}\]")
_ZERO_WIDTH_RE = re.compile(r"[​-‏⁠﻿]")
_BOUNDARY_RE = re.compile(r"[.!?]+[\"'”’)\]]*\s+")


def normalize_for_speech(text: str) -> str:
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _URL_RE.sub("link", text)
    text = _FOOTNOTE_RE.sub("", text)
    text = text.replace("&", " and ").replace("…", "...")
    text = re.sub(r"\s*[—–]\s*", ", ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    start = 0
    for m in _BOUNDARY_RE.finditer(text):
        end = m.end()
        before = text[start:m.start() + 1]
        word_match = re.search(r"([A-Za-z][A-Za-z.]*)\.$", before)
        word = word_match.group(1) if word_match else ""
        token = word.rstrip(".").lower()
        nxt = text[end:end + 1]
        if token in _ABBREVIATIONS:
            continue
        if len(word) == 1 and word.isupper():
            continue  # initials such as "J. K. Rowling"
        if nxt and not (nxt.isupper() or nxt.isdigit() or nxt in "\"'“‘(["):
            continue
        sentences.append(text[start:end].strip())
        start = end
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def _split_long(sentence: str, limit: int) -> list[str]:
    if len(sentence) <= limit:
        return [sentence]
    parts: list[str] = []
    current = ""
    for piece in re.split(r"(?<=[,;:])\s+", sentence):
        if current and len(current) + 1 + len(piece) > limit:
            parts.append(current)
            current = piece
        else:
            current = f"{current} {piece}".strip()
    if current:
        parts.append(current)
    final: list[str] = []
    for part in parts:
        while len(part) > limit:
            cut = part.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            final.append(part[:cut].strip())
            part = part[cut:].strip()
        if part:
            final.append(part)
    return final


def pack_paragraph(paragraph: str, max_chars: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for sentence in split_sentences(normalize_for_speech(paragraph)):
        for piece in _split_long(sentence, max_chars):
            if current and len(current) + 1 + len(piece) > max_chars:
                chunks.append(current)
                current = piece
            else:
                current = f"{current} {piece}".strip()
    if current:
        chunks.append(current)
    return chunks


def build_chunks(chapter: Chapter, max_chars: int = 350) -> list[Chunk]:
    """Chapter title first, then each paragraph as one or more chunks with natural pauses."""
    chunks: list[Chunk] = []
    heading = normalize_for_speech(chapter.title)
    if heading:
        chunks.append(Chunk(heading.rstrip(".") + ".", HEADING_PAUSE))
    for paragraph in chapter.paragraphs:
        pieces = pack_paragraph(paragraph, max_chars)
        for i, piece in enumerate(pieces):
            last = i == len(pieces) - 1
            chunks.append(Chunk(piece, PARAGRAPH_PAUSE if last else SENTENCE_PAUSE))
    return chunks
