"""Prosody: how a paragraph should be *performed*, and the exact silence between spoken pieces.

A narrator doesn't read at one speed with one pause length. This module decides, per sentence:

* how long to breathe afterwards (a question, a full stop, an ellipsis, the end of a paragraph, a
  change of speaker, a scene break all differ),
* how fast and how expressively to say it (a whispered one-line paragraph versus a shouted line).

The decisions come from ``ParaNote`` objects. Rules produce them offline (:func:`heuristic_note`);
the AI director can overwrite them paragraph by paragraph. Pauses are inserted as real silence by the
pipeline, after the engine's own leading and trailing silence has been trimmed, so they are exact.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .chunker import _split_long, is_scene_break, normalize_for_speech, split_sentences
from .models import Chapter, Chunk
from .speech import speak_heading, speak_ready

PAUSE_STYLES = {"snappy": 0.8, "natural": 1.0, "relaxed": 1.25}

# Seconds of silence, before the style multiplier.
PAUSE = {
    "sentence": 0.38,
    "question": 0.46,
    "exclaim": 0.42,
    "ellipsis": 0.65,
    "clause": 0.2,  # inside one sentence, at a comma split
    "dialogue_run": 0.3,  # one speaker's consecutive sentences
    "dialogue_turn": 0.55,  # speaker changes
    "paragraph": 0.85,
    "scene_break": 1.9,
    "heading": 1.4,
    "section_heading": 0.9,
}
PAUSE_LEVEL = {"short": 0.65, "normal": 1.0, "long": 1.55, "dramatic": 2.3}
PACE = {"slow": 0.92, "normal": 1.0, "fast": 1.08}
INTENSITY = {"calm": 0.3, "normal": 0.5, "intense": 0.75}
KINDS = {"narration", "dialogue", "heading", "scene_break", "epigraph", "letter", "verse"}

_QUOTE_OPEN = "\"“‘'"
_TRAILING_CLOSERS = "\"'”’)]"


@dataclass
class ParaNote:
    kind: str = "narration"
    pace: str = "normal"
    pause: str = "normal"
    intensity: str = "normal"
    skip: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ParaNote":
        """Build from untrusted data (an AI answer): unknown values fall back to the default."""
        base = cls()

        def pick(key: str, allowed) -> str:
            v = str(d.get(key, getattr(base, key))).strip().lower()
            return v if v in allowed else getattr(base, key)

        return cls(pick("kind", KINDS), pick("pace", PACE), pick("pause", PAUSE_LEVEL),
                   pick("intensity", INTENSITY), bool(d.get("skip", False)))


def style_scale(style: str | None) -> float:
    return PAUSE_STYLES.get((style or "natural").lower(), 1.0)


def heuristic_note(paragraph: str) -> ParaNote:
    """Rule-based direction for one paragraph, good enough to sound clearly better than a flat read."""
    p = paragraph.strip()
    if is_scene_break(p):
        return ParaNote(kind="scene_break")
    words = p.split()
    letters = [c for c in p if c.isalpha()]
    if letters and 0 < len(words) <= 8 and p.upper() == p and not p.endswith((".", "!", "?", "…")):
        return ParaNote(kind="heading", pace="slow", pause="long")
    note = ParaNote()
    if p[:1] in _QUOTE_OPEN:
        note.kind = "dialogue"
    if p.count("!") >= 2 or p.rstrip(_TRAILING_CLOSERS).endswith("!") and note.kind == "dialogue":
        note.intensity = "intense"
    if p.rstrip(_TRAILING_CLOSERS).endswith(("...", "…")):
        note.pause = "long"
    elif len(words) <= 4 and note.kind == "narration" and re.search(r"[.!?]$", p):
        note.pause = "long"  # a one-line paragraph is a beat; let it land
    return note


def _end_kind(sentence: str) -> str:
    s = sentence.rstrip().rstrip(_TRAILING_CLOSERS).rstrip()
    if s.endswith(("...", "…")):
        return "ellipsis"
    if s.endswith("?"):
        return "question"
    if s.endswith("!"):
        return "exclaim"
    if s.endswith((",", ";", ":", "-")):
        return "clause"
    return "sentence"


def _starts_quote(sentence: str) -> bool:
    return sentence.lstrip()[:1] in _QUOTE_OPEN


def build_chunks(
    chapter: Chapter,
    max_chars: int = 350,
    *,
    notes: list[ParaNote] | None = None,
    style: str = "natural",
    pronunciations: dict[str, str] | None = None,
) -> list[Chunk]:
    """Chapter heading, then every sentence of every paragraph with its own pause, pace and energy."""
    scale = style_scale(style)
    notes = notes if notes is not None else [heuristic_note(p) for p in chapter.paragraphs]
    chunks: list[Chunk] = []

    def add(text: str, pause: float, speed: float = 1.0, intensity: float = 0.5) -> None:
        if re.search(r"\w", text):
            chunks.append(Chunk(text, round(pause * scale, 3), round(speed, 3), round(intensity, 3)))

    def stretch_last(seconds: float) -> None:
        if chunks:
            chunks[-1].pause_after = max(chunks[-1].pause_after, round(seconds * scale, 3))

    title = speak_heading(normalize_for_speech(chapter.title)) if chapter.announce else ""
    if title:
        add(title.rstrip(".:") + ".", PAUSE["heading"], speed=0.95, intensity=0.4)

    for pi, (paragraph, note) in enumerate(zip(chapter.paragraphs, notes)):
        if note.skip:
            continue
        if note.kind == "scene_break":
            stretch_last(PAUSE["scene_break"])
            continue
        if note.kind == "heading":
            stretch_last(PAUSE["section_heading"] + 0.3)
            add(speak_heading(normalize_for_speech(paragraph.title())).rstrip(".:") + ".",
                PAUSE["section_heading"], speed=0.95, intensity=0.4)
            continue
        next_dialogue = pi + 1 < len(notes) and notes[pi + 1].kind == "dialogue" and not notes[pi + 1].skip
        speed = PACE[note.pace] * (0.95 if note.kind in ("epigraph", "verse") else 1.0)
        base_intensity = INTENSITY[note.intensity]
        sentences = split_sentences(normalize_for_speech(paragraph))
        for si, sentence in enumerate(sentences):
            last_sentence = si == len(sentences) - 1
            kind = _end_kind(sentence)
            if last_sentence:
                pause = PAUSE["dialogue_turn"] if note.kind == "dialogue" and next_dialogue else PAUSE["paragraph"]
                pause = max(pause * PAUSE_LEVEL[note.pause], PAUSE[kind] if kind != "clause" else 0.0)
            elif note.kind == "dialogue" and _starts_quote(sentences[si + 1]) and kind in ("sentence", "exclaim"):
                pause = PAUSE["dialogue_run"]
            else:
                pause = PAUSE[kind] * (PAUSE_LEVEL[note.pause] if note.pause == "dramatic" else 1.0)
            intensity = min(1.0, base_intensity + (0.12 if kind == "exclaim" else 0.0))
            pieces = _split_long(sentence, max_chars)
            for part_i, piece in enumerate(pieces):
                text = speak_ready(piece, pronunciations)
                add(text, pause if part_i == len(pieces) - 1 else PAUSE["clause"], speed, intensity)
    return chunks


def describe_pauses(chunks: list[Chunk], limit: int = 12) -> list[str]:
    """Human-readable preview of the first chunks, for ``murmur direct``."""
    return [f"[{c.pause_after:.2f}s  x{c.speed:.2f}  e{c.intensity:.2f}] {c.text[:90]}" for c in chunks[:limit]]
