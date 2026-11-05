"""The narration director: decides *what* is read and *how*, before any voice is involved.

    extracted book ──▶ where does the story start / end? ──▶ per-paragraph direction ──▶ script
                       (rules, then AI double-check)         (rules, then AI per window)

Everything an AI contributes is advisory and validated: a bad, partial or missing answer degrades
to the rule-based result for that part of the book, never to a failed job. AI answers are cached
on disk keyed by their input, so resuming an interrupted job never pays twice.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .ai import AIError, AIProvider, parse_json_object
from .config import Settings
from .extract import Cancelled
from .matter import MAX_TRIM_SHARE, Span, apply_span, classify_title, locate_story
from .models import Book, Chapter, Chunk
from .prosody import ParaNote, build_chunks, heuristic_note

log = logging.getLogger("murmur.director")

PROMPT_VERSION = "v1"
ProgressCb = Callable[[float, str], None]

LOCATE_SYSTEM = """You are the producer of an audiobook. You are given the outline of a book that was extracted from a PDF, \
plus the first and last paragraphs. Decide which part should be narrated.

Narrate from where the actual book begins: a prologue, prelude, or first chapter for fiction; the introduction or \
first chapter for non-fiction. Do NOT narrate: cover or title pages, copyright, publisher, ISBN and printing \
details, dedications, epigraphs, praise and reviews, "also by" lists, tables of contents, maps, cast lists, \
acknowledgments, author biographies, excerpts of other books, newsletter or social-media promotions, notes, \
glossaries and indexes. In fiction, also skip a foreword or preface written by someone other than the story's \
narrator; in non-fiction keep the author's own preface and introduction.

Paragraphs are addressed as "chapter:paragraph" (both zero-based). Reply with JSON only:
{"fiction": true, "story_start": {"chapter": 0, "paragraph": 0}, "story_end": {"chapter": 0, "paragraph": 0}, \
"reasons": ["one short sentence each"]}
story_start is the FIRST paragraph to narrate (use a chapter's paragraph 0 when the story begins at its start). \
story_end is the LAST paragraph to narrate (inclusive)."""

ANNOTATE_SYSTEM = """You direct the narration of an audiobook, advising a narrator on how to perform each numbered \
paragraph, like a studio director would for Audible. Reply with JSON only.

Defaults: kind "narration", pace "normal", pause "normal", intensity "normal", skip false. List ONLY paragraphs \
that deviate from the defaults; most paragraphs should not be listed.
- kind: "dialogue" (mostly spoken lines), "epigraph", "letter", "verse", "heading" (a section title inside the \
chapter), "scene_break" (an ornament such as * * *), or "narration".
- pace: "slow" for grief, awe, a reveal, quiet reflection; "fast" for action, panic, excitement.
- pause (the silence AFTER the paragraph): "short" when the thought continues into the next paragraph; "long" for \
a shift in time, place or mood and for weighty moments; "dramatic" for a cliffhanger, a shock, or a line that \
should echo.
- intensity: "calm" for whispers, tenderness, exhaustion; "intense" for shouting, fear, fury, urgency.
- skip: true ONLY for text that is not part of the book's content: leftover page headers, copyright text, \
advertisements, "also by" lists, author bios.
- pronunciations: only words a speech engine would likely mispronounce (character and place names, invented, \
foreign or archaic words). The value is a respelling in plain English letters, hyphenated by syllable, with the \
stressed syllable in capitals, e.g. "Hermione": "her-MY-oh-nee". At most 12 entries; never include common words.

Format: {"paragraphs": [{"i": 3, "pace": "slow", "pause": "long"}], "pronunciations": {"Name": "respelling"}}"""


# ---- results ----------------------------------------------------------------------------------------
@dataclass
class ScriptChapter:
    title: str
    chunks: list[Chunk]

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "chunks": [c.to_dict() for c in self.chunks]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ScriptChapter":
        return cls(d["title"], [Chunk.from_dict(c) for c in d["chunks"]])


@dataclass
class Script:
    """Everything the narrator will say, with performance directions, plus a report for the UI."""

    chapters: list[ScriptChapter]
    report: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"chapters": [c.to_dict() for c in self.chapters], "report": self.report}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Script":
        return cls([ScriptChapter.from_dict(c) for c in d["chapters"]], d.get("report", {}))

    @property
    def total_chars(self) -> int:
        return sum(len(c.text) for ch in self.chapters for c in ch.chunks)


@dataclass
class DirectorOptions:
    use_ai: bool = True
    start_at_story: bool = True
    keep_intro: bool | None = None
    pause_style: str = "natural"

    @classmethod
    def from_job(cls, options: dict[str, Any], settings: Settings) -> "DirectorOptions":
        ai = options.get("ai")
        return cls(
            use_ai=(ai not in ("off", False, "false", "0")),
            start_at_story=bool(options.get("start_at_story", settings.start_at_story)),
            keep_intro=options.get("keep_intro"),
            pause_style=options.get("pause_style") or settings.pause_style,
        )


class _Cache:
    """Tiny on-disk memo for AI answers, so a restarted job doesn't pay for the same call twice."""

    def __init__(self, path: Path | None):
        self.path, self._lock = path, threading.Lock()
        self._data: dict[str, Any] = {}
        if path and path.exists():
            try:
                self._data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def get(self, key: str) -> Any | None:
        with self._lock:
            return self._data.get(key)

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            if self.path:
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text(json.dumps(self._data))
                os.replace(tmp, self.path)


def _key(*parts: str) -> str:
    return hashlib.sha1("\x1f".join((PROMPT_VERSION, *parts)).encode()).hexdigest()


# ---- locating the story -----------------------------------------------------------------------------
def _snip(text: str, n: int = 150) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def _locate_prompt(chapters: list[Chapter]) -> str:
    n = len(chapters)
    shown = list(range(n)) if n <= 120 else list(range(60)) + list(range(n - 60, n))
    outline = "\n".join(f"{i}. {_snip(chapters[i].title, 70)} ({chapters[i].word_count} words)" for i in shown)

    def view(coords: list[tuple[int, int]]) -> str:
        lines: list[str] = []
        current = None
        for ci, pi in coords:
            if ci != current:
                lines.append(f"-- chapter {ci}: {_snip(chapters[ci].title, 60)} --")
                current = ci
            lines.append(f"{ci}:{pi} | {_snip(chapters[ci].paragraphs[pi])}")
        return "\n".join(lines)

    every = [(ci, pi) for ci, ch in enumerate(chapters) for pi in range(len(ch.paragraphs))]
    first, back = view(every[:100]), view(every[-60:])
    return f"OUTLINE\n{outline}\n\nBEGINNING OF THE BOOK\n{first}\n\nEND OF THE BOOK\n{back}"


def _coord(obj: Any, chapters: list[Chapter]) -> tuple[int, int] | None:
    try:
        c, p = int(obj["chapter"]), int(obj["paragraph"])
    except (KeyError, TypeError, ValueError):
        return None
    if not 0 <= c < len(chapters):
        return None
    return c, min(max(p, 0), max(len(chapters[c].paragraphs) - 1, 0))


def _words_before(chapters: list[Chapter], at: tuple[int, int]) -> int:
    c, p = at
    return sum(ch.word_count for ch in chapters[:c]) + sum(len(x.split()) for x in chapters[c].paragraphs[:p])


def _span_from_ai(chapters: list[Chapter], reply: dict[str, Any], fallback: Span) -> tuple[Span | None, list[str]]:
    """Validate the AI's answer; None means it can't be trusted and the rules win."""
    start, end = _coord(reply.get("story_start"), chapters), _coord(reply.get("story_end"), chapters)
    if start is None or end is None or end < start:
        return None, []
    total = max(sum(c.word_count for c in chapters), 1)
    front_words = _words_before(chapters, start)
    tail_words = total - _words_before(chapters, (end[0], min(end[1] + 1, len(chapters[end[0]].paragraphs))))
    if front_words > total * 0.35 or tail_words > total * 0.35 or (front_words + tail_words) > total * MAX_TRIM_SHARE:
        log.warning("ignoring implausible AI story span %s-%s", start, end)
        return None, []
    span = Span(start=start, end=end, fiction=bool(reply.get("fiction", fallback.fiction)))
    sc, sp = start
    for i in range(sc):
        span.removed.append({"where": "front", "title": chapters[i].title, "words": chapters[i].word_count,
                             "reason": "not part of the story"})
    if sp:
        span.removed.append({"where": "front", "title": f"{chapters[sc].title} (opening pages)",
                             "words": sum(len(x.split()) for x in chapters[sc].paragraphs[:sp]),
                             "reason": "not part of the story"})
    ec, ep = end
    if ep + 1 < len(chapters[ec].paragraphs):
        span.removed.append({"where": "back", "title": f"{chapters[ec].title} (closing pages)",
                             "words": sum(len(x.split()) for x in chapters[ec].paragraphs[ep + 1:]),
                             "reason": "not part of the story"})
    for i in range(ec + 1, len(chapters)):
        span.removed.append({"where": "back", "title": chapters[i].title, "words": chapters[i].word_count,
                             "reason": "not part of the story"})
    reasons = [str(r)[:200] for r in (reply.get("reasons") or []) if isinstance(r, str)][:5]
    return span, reasons


# ---- annotating paragraphs --------------------------------------------------------------------------
@dataclass
class _Window:
    chapter: int
    first: int
    paragraphs: list[str]
    title: str


def _windows(chapters: list[Chapter], max_chars: int, max_paras: int = 30) -> list[_Window]:
    out: list[_Window] = []
    for ci, ch in enumerate(chapters):
        cur: list[str] = []
        size, first = 0, 0
        for pi, p in enumerate(ch.paragraphs):
            if cur and (size + len(p) > max_chars or len(cur) >= max_paras):
                out.append(_Window(ci, first, cur, ch.title))
                cur, size, first = [], 0, pi
            cur.append(p)
            size += len(p)
        if cur:
            out.append(_Window(ci, first, cur, ch.title))
    return out


def _window_prompt(w: _Window) -> str:
    lines = []
    for i, p in enumerate(w.paragraphs):
        if len(p) > 1600:  # a monster paragraph: the model only needs its flavour
            p = p[:1000] + " […] " + p[-500:]
        lines.append(f"[{i}] {p}")
    return f"Chapter: {w.title}\n\n" + "\n".join(lines)


_RESPELL_RE = re.compile(r"^[A-Za-z' \-]{2,60}$")


def _clean_pronunciations(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for word, respelling in list(raw.items())[:12]:
        if isinstance(word, str) and isinstance(respelling, str) and 2 <= len(word) <= 40 \
                and _RESPELL_RE.match(respelling.strip()) and word.strip().lower() != respelling.strip().lower():
            out[word.strip()] = respelling.strip()
    return out


def _parse_window(reply_text: str, count: int) -> tuple[dict[int, ParaNote], dict[str, str]]:
    obj = parse_json_object(reply_text)
    notes: dict[int, ParaNote] = {}
    for item in obj.get("paragraphs") or []:
        if not isinstance(item, dict):
            continue
        try:
            i = int(item.get("i"))
        except (TypeError, ValueError):
            continue
        if 0 <= i < count:
            notes[i] = ParaNote.from_dict(item)
    if len(notes) > count * 0.9 and count >= 6:
        log.warning("AI listed nearly every paragraph; treating the window as unreliable")
        raise AIError("Implausible annotation.")
    skipped = sum(1 for n in notes.values() if n.skip)
    if skipped > max(2, count * 0.3):
        for n in notes.values():
            n.skip = False  # an AI that wants to drop a third of a window is wrong, not the book
    return notes, _clean_pronunciations(obj.get("pronunciations"))


# ---- orchestration ----------------------------------------------------------------------------------
def direct_book(
    book: Book,
    settings: Settings,
    options: DirectorOptions | None = None,
    *,
    provider: AIProvider | None = None,
    cache_path: Path | None = None,
    progress: ProgressCb | None = None,
    cancel: threading.Event | None = None,
) -> Script:
    options = options or DirectorOptions()
    provider = provider if options.use_ai else None
    cache = _Cache(cache_path)
    notes_log: list[str] = []
    stats = {"ai_calls": 0, "ai_cached": 0, "ai_failures": 0}

    def check() -> None:
        if cancel is not None and cancel.is_set():
            raise Cancelled()

    def say(frac: float, msg: str) -> None:
        if progress:
            progress(frac, msg)

    chapters = book.chapters

    # 1. Where does the story start and end?
    say(0.0, "Finding where the story begins")
    rules = locate_story(chapters, keep_intro=options.keep_intro)
    span = rules
    located_by = "rules"
    reasons: list[str] = []
    if options.start_at_story and provider is not None and chapters:
        check()
        prompt = _locate_prompt(chapters)
        key = _key("locate", provider.model, prompt)
        cached = cache.get(key)
        try:
            if cached is None:
                stats["ai_calls"] += 1
                cached = parse_json_object(provider.complete(LOCATE_SYSTEM, prompt, max_tokens=800))
                cache.put(key, cached)
            else:
                stats["ai_cached"] += 1
            ai_span, reasons = _span_from_ai(chapters, cached, rules)
            if ai_span is not None:
                span, located_by = ai_span, "ai"
            else:
                notes_log.append("The AI's choice of start and end looked wrong, so rule-based detection was used.")
        except AIError as exc:
            stats["ai_failures"] += 1
            notes_log.append(f"AI story detection unavailable ({exc}); used rule-based detection.")
    narrated = apply_span(chapters, span) if options.start_at_story else [
        Chapter(c.title, list(c.paragraphs), c.announce and c.title.strip().lower() != "opening") for c in chapters]
    if options.start_at_story:
        notes_log.extend(span.notes)
    if not narrated:  # never produce an empty book because of an over-eager cut
        narrated = [Chapter(c.title, list(c.paragraphs), c.announce) for c in chapters]
        span = Span(fiction=span.fiction)
        notes_log.append("Nothing was left after trimming, so the whole book is narrated.")
    say(0.1, "Planning the narration")

    # 2. How should each paragraph be performed? Rules first, then the AI overrides per window.
    notes: list[list[ParaNote]] = [[heuristic_note(p) for p in c.paragraphs] for c in narrated]
    pronunciations: dict[str, str] = {}
    if provider is not None:
        windows = _windows(narrated, settings.ai_window_chars)
        state = {"failures": 0, "done": 0, "dead": False}
        lock = threading.Lock()

        def run(w: _Window):
            if state["dead"]:
                return w, None, {}
            prompt = _window_prompt(w)
            key = _key("annotate", provider.model, prompt)
            cached = cache.get(key)
            try:
                if cached is not None:
                    with lock:
                        stats["ai_cached"] += 1
                    got, pron = _parse_window(json.dumps(cached), len(w.paragraphs))
                    return w, got, pron
                with lock:
                    stats["ai_calls"] += 1
                text = provider.complete(ANNOTATE_SYSTEM, prompt, max_tokens=2000)
                got, pron = _parse_window(text, len(w.paragraphs))
                cache.put(key, parse_json_object(text))
                return w, got, pron
            except AIError as exc:
                with lock:
                    stats["ai_failures"] += 1
                    state["failures"] += 1
                    if state["failures"] >= 4 and state["done"] == 0:
                        state["dead"] = True  # bad key or server down: stop paying for failures
                        notes_log.append(f"The AI kept failing ({exc}); the rest used rule-based direction.")
                return w, None, {}

        with ThreadPoolExecutor(max_workers=settings.ai_concurrency, thread_name_prefix="murmur-ai") as pool:
            futures = [pool.submit(run, w) for w in windows]
            try:
                for n_done, fut in enumerate(as_completed(futures), start=1):
                    check()
                    w, got, pron = fut.result()
                    if got is not None:
                        with lock:
                            state["done"] += 1
                        for i, note in got.items():
                            base = notes[w.chapter][w.first + i]
                            # Keep the rules' scene-break / heading calls; the AI refines everything else.
                            if base.kind in ("scene_break", "heading") and note.kind == "narration":
                                note.kind = base.kind
                            notes[w.chapter][w.first + i] = note
                        for word, respelling in pron.items():
                            pronunciations.setdefault(word, respelling)
                    say(0.1 + 0.9 * n_done / max(len(windows), 1),
                        f"Directing the narration ({n_done} of {len(windows)})")
            except BaseException:
                for f in futures:
                    f.cancel()
                raise
    check()

    # 3. Build the sentence-level script.
    script_chapters = [
        ScriptChapter(
            c.title,
            build_chunks(c, settings.chunk_chars, notes=n, style=options.pause_style, pronunciations=pronunciations),
        )
        for c, n in zip(narrated, notes)
    ]
    script_chapters = [c for c in script_chapters if c.chunks]
    report = _report(chapters, narrated, span, located_by, provider, reasons, stats, notes_log, options,
                     pronunciations)
    say(1.0, "Narration planned")
    return Script(script_chapters, report)


def _report(original: list[Chapter], narrated: list[Chapter], span: Span, located_by: str,
            provider: AIProvider | None, reasons: list[str], stats: dict[str, int], notes_log: list[str],
            options: DirectorOptions, pronunciations: dict[str, str]) -> dict[str, Any]:
    first = narrated[0] if narrated else None
    preview = _snip(first.paragraphs[0], 140) if first and first.paragraphs else ""
    removed = [r for r in span.removed if r["words"] > 0]
    return {
        "provider": provider.name if provider else "rules",
        "model": provider.model if provider else "",
        "story_found_by": located_by,
        "fiction": span.fiction,
        "start_at_story": options.start_at_story,
        "story_starts_with": {"chapter": first.title if first else "", "text": preview},
        "removed": removed,
        "removed_words": sum(r["words"] for r in removed),
        "narrated_words": sum(c.word_count for c in narrated),
        "chapters": len(narrated),
        "reasons": reasons,
        "pronunciations": pronunciations,
        "pause_style": options.pause_style,
        "ai_calls": stats["ai_calls"],
        "ai_cached": stats["ai_cached"],
        "ai_failures": stats["ai_failures"],
        "notes": notes_log,
    }


__all__ = ["Script", "ScriptChapter", "DirectorOptions", "direct_book", "classify_title"]
