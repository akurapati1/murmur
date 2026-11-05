"""Find where the *story* starts and ends, and cut the publishing furniture around it.

Books wrap their content in material nobody wants read aloud: cover and title pages, copyright
and ISBN blocks, dedications, "praise for", tables of contents, acknowledgments, "also by",
author bios and newsletter pitches. This module finds the narrated span with rules alone.
:mod:`murmur.director` can then let an AI double-check it.

Coordinates are ``(chapter_index, paragraph_index)`` into the list of chapters from extraction.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import Chapter

# ---- chapter titles ------------------------------------------------------------------------------
_MATTER_TITLE_RE = re.compile(
    r"^(?:cover(?: page)?|title page|half[- ]?title(?: page)?|frontispiece|copyright(?: page| notice)?|"
    r"dedication|epigraph|contents|table of contents|praise(?: for.*)?|reviews?|"
    r"also by.*|books by.*|other (?:books|titles|works).*|by the same author|more (?:books )?by.*|"
    r"about the (?:author|authors|book|publisher|type)|about (?:the )?author|author bio|biography|"
    r"acknowledg(?:e)?ments?|credits|colophon|permissions|list of (?:figures|illustrations|tables|maps|plates)|"
    r"maps?|dramatis personae|cast of characters|characters|family tree|"
    r"front matter|back matter|title|"
    r"also available|newsletter|sign up.*|join (?:my|our|the).*|reading group.*|book club.*|discussion questions|"
    r"(?:a )?note (?:on|about|from|to) (?:the )?(?:author|publisher|text|reader|readers|this edition)|"
    r"author'?s note|translator'?s note|editor'?s note|publisher'?s note|"
    r"excerpt(?: from.*)?|preview(?: of.*)?|sneak peek.*|bonus (?:chapter|material|content)|"
    r"thank you for reading.*|the end|index|bibliography|references|endnotes|notes|glossary|"
    r"further reading|appendix.*|sources|select bibliography|copyright and permissions)$",
    re.IGNORECASE,
)
# Foreword / preface / introduction: part of a non-fiction book, noise in a novel.
_OPTIONAL_TITLE_RE = re.compile(r"^(?:foreword|preface|introduction|prefatory note|afterword|postscript|"
                                r"introduction to the .*edition|preface to the .*edition)$", re.IGNORECASE)
# Chapters that are unmistakably where the story begins.
_STORY_TITLE_RE = re.compile(
    r"^(?:prologue|prelude|prolog|overture|pr[eé]cis|chapter\s+(?:1|one|i)\b.*|part\s+(?:1|one|i)\b.*|"
    r"book\s+(?:1|one|i)\b.*|1|one|i|1\.|i\.|one:.*|1:.*|section\s+(?:1|one)\b.*|"
    r"chapter\s+\w+.*)$",
    re.IGNORECASE,
)

# ---- paragraphs ----------------------------------------------------------------------------------
_BOILERPLATE = [re.compile(p, re.IGNORECASE) for p in (
    r"all rights reserved", r"©|\(c\)\s*\d{4}|\bcopyright\b", r"\bisbn(?:-1[03])?\b|\b97[89][-\s]?\d", r"library of congress",
    r"cataloging[- ]in[- ]publication|cataloguing", r"\bfirst (?:edition|published|printing|us edition|paperback)\b",
    r"\bprinted (?:in|by)\b|\bprinting history\b|\b\d+\s*[-–]\s*\d+\s*[-–]\s*\d+\s*[-–]\s*\d+\b",
    r"\bpublished (?:by|in|simultaneously|under)\b", r"\bno part of this (?:book|publication|work)\b",
    r"reproduced.{0,40}(?:form|means)|transmitted in any form", r"this is a work of fiction",
    r"any resemblance to actual", r"names, characters.{0,30}incidents", r"\bcover (?:design|art|illustration|photo)",
    r"\bbook design\b|\btypeset|\bjacket design", r"\bwww\.|https?://|\b\w+@\w+\.(?:com|org|net)\b",
    r"\be-?book\b.*\b(?:edition|isbn)\b", r"\bvisit us\b|\bfollow us\b|sign up for (?:our|the|my) newsletter",
    r"\bimprint of\b|\bdivision of\b|\bpenguin\b|\bharpercollins\b|\brandom house\b|\bsimon\s*(?:&|and)\s*schuster\b|"
    r"\bhachette\b|\bmacmillan\b|\bscholastic\b|\bbloomsbury\b|\bvintage books\b|\bdoubleday\b|\bknopf\b",
    r"\bfor (?:information|permission)\b.*\b(?:contact|write)\b", r"\bpublisher'?s? (?:note|information)\b",
    r"\bmanufactured in\b|\bmade in the\b", r"\bpermission\b.{0,40}\b(?:reprint|quote|reproduce)\b",
    r"\bset in\b.{0,30}\btypeface\b", r"\btranslat(?:ed|ion) by\b",
)]
_BACK_SIGNAL = [re.compile(p, re.IGNORECASE) for p in (
    r"^about the authors?\b", r"^acknowledg(?:e)?ments?\b", r"^also by\b", r"^books by\b", r"^other books\b",
    r"^praise for\b", r"^if you (?:enjoyed|liked|loved)\b", r"^thank you for (?:reading|purchasing|buying)\b",
    r"^(?:sign up|subscribe|join)\b.*\bnewsletter\b", r"^(?:an )?excerpt from\b", r"^coming soon\b",
    r"^read on for\b", r"^discussion questions\b", r"^reading group\b", r"^a note on the\b", r"^author'?s note\b",
    r"^connect with\b", r"^follow\b.*\bon (?:twitter|instagram|facebook)\b", r"^copyright\b", r"^keep reading\b",
    r"^(?:the )?end\.?$", r"^about the publisher\b", r"^glossary\b", r"^bibliography\b", r"^index\b",
    r"^notes\b$", r"^endnotes\b", r"^leave a review\b", r"^please (?:consider )?(?:leave|write)\b.*\breview\b",
)]
_PARA_FRONT_HEADING_RE = re.compile(
    r"^(?:contents|table of contents|dedication|epigraph|praise for.*|also by.*|copyright.*|"
    r"acknowledg(?:e)?ments?|title page|half[- ]?title|about the author|foreword|preface)$", re.IGNORECASE)
_STORY_HEADING_RE = re.compile(
    r"^(?:prologue|prelude|prolog|chapter\s+(?:\d+|[ivxlcdm]+|one|two)\b.{0,50}|part\s+(?:\d+|one|i)\b.{0,40}|"
    r"book\s+(?:\d+|one|i)\b.{0,40}|1|one|i)\.?$", re.IGNORECASE)
_ATTRIBUTION_RE = re.compile(r"^[\"“]?.{0,200}[\"”]?\s*[—–\-~]\s*[A-Z][\w.'’ ]{2,40}$|^[—–\-~]\s*[A-Z]")
_SENTENCE_END_RE = re.compile(r"[.!?…][\"'”’)\]]*$")

MAX_TRIM_SHARE = 0.4  # never cut more than this share of a book's words, whatever any rule says
FRONT_SCAN = 90  # paragraphs from the start examined for front matter
BACK_SCAN = 70  # paragraphs from the end examined for back matter


def _norm(title: str) -> str:
    return re.sub(r"\s+", " ", title.strip().strip(".:–—- ")).lower()


def classify_title(title: str) -> str:
    """'matter' | 'optional' | 'story' | 'neutral'."""
    t = _norm(title)
    if not t or t == "opening":
        return "neutral"
    if _MATTER_TITLE_RE.match(t):
        return "matter"
    if _OPTIONAL_TITLE_RE.match(t):
        return "optional"
    if _STORY_TITLE_RE.match(t):
        return "story"
    return "neutral"


def boilerplate_hits(text: str) -> int:
    return sum(1 for rx in _BOILERPLATE if rx.search(text))


def is_boilerplate(text: str) -> bool:
    hits = boilerplate_hits(text)
    if hits >= 2:
        return True
    return hits == 1 and len(text.split()) < 60


def is_back_signal(text: str) -> bool:
    t = text.strip()
    return len(t) < 160 and any(rx.search(t) for rx in _BACK_SIGNAL)


def _is_prose(text: str) -> bool:
    return len(text.split()) >= 14 and bool(_SENTENCE_END_RE.search(text.strip())) and not is_boilerplate(text)


def looks_like_fiction(chapters: list[Chapter]) -> bool:
    """Dialogue-rich text means a novel; that decides whether forewords and introductions are kept."""
    paras = [p for c in chapters if classify_title(c.title) != "matter" for p in c.paragraphs][:600]
    if len(paras) < 8:
        return False
    quoted = sum(1 for p in paras if p[:1] in "\"“‘'")
    return quoted / len(paras) >= 0.06


# ---- result --------------------------------------------------------------------------------------
@dataclass
class Span:
    """The part of the book that gets narrated, plus a record of what was cut and why."""

    start: tuple[int, int] = (0, 0)
    end: tuple[int, int] | None = None  # inclusive last paragraph; None = end of book
    fiction: bool = False
    removed: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _words(paras: list[str]) -> int:
    return sum(len(p.split()) for p in paras)


def _front_cut(paras: list[str]) -> int:
    """Index of the first paragraph worth narrating within ``paras`` (0 if nothing to cut)."""
    window = paras[:FRONT_SCAN]
    last_front = -1
    for i, p in enumerate(window):
        if is_boilerplate(p) or (len(p) < 60 and _PARA_FRONT_HEADING_RE.match(_norm(p))):
            last_front = i
    start = last_front + 1
    # A real story heading further on beats any guess about dedications and epigraphs.
    for i in range(start, len(window)):
        if _STORY_HEADING_RE.match(window[i].strip()) and (last_front >= 0 or i <= 12):
            return i
        if _is_prose(window[i]):
            break
    # Skip title/author lines, dedications and epigraphs that sit between the boilerplate and the prose.
    i = start
    while i < len(window) and i - start < 14:
        p = window[i].strip()
        short_line = len(p.split()) < 14 and not _SENTENCE_END_RE.search(p)
        attribution = bool(_ATTRIBUTION_RE.match(p))
        dedication = bool(re.match(r"^(?:for|to|in memory of|in loving memory of|dedicated to)\b", p, re.I)) \
            and len(p.split()) < 14
        quoted_epigraph = p[:1] in "\"“" and len(p.split()) < 60 and i + 1 < len(window) \
            and bool(_ATTRIBUTION_RE.match(window[i + 1].strip()))
        if _is_prose(p) and not quoted_epigraph:
            break
        if short_line or attribution or dedication or quoted_epigraph or is_boilerplate(p):
            i += 1
            continue
        break
    # If nothing looked like front matter at all, only trim a couple of stray title lines.
    if last_front < 0:
        return i if i <= 3 and i < len(window) and _is_prose(window[i]) and i > 0 else 0
    return i


def _back_cut(paras: list[str]) -> int:
    """Number of paragraphs of ``paras`` to keep (everything after is back matter)."""
    n = len(paras)
    first = max(0, n - BACK_SCAN)
    for i in range(first, n):
        p = paras[i]
        if is_back_signal(p) and i >= int(n * 0.4):
            return i
    keep = n
    while keep > 0 and (is_boilerplate(paras[keep - 1]) or is_back_signal(paras[keep - 1])) and n - keep < 12:
        keep -= 1
    return keep


def locate_story(chapters: list[Chapter], *, keep_intro: bool | None = None) -> Span:
    """Rule-based story span. ``keep_intro`` forces forewords/introductions in or out (default: only for non-fiction)."""
    span = Span()
    if not chapters:
        return span
    fiction = looks_like_fiction(chapters)
    span.fiction = fiction
    keep_optional = (not fiction) if keep_intro is None else keep_intro
    kinds = [classify_title(c.title) for c in chapters]
    n = len(chapters)
    total_words = max(sum(c.word_count for c in chapters), 1)

    # --- start chapter ---
    story_first = next((i for i in range(min(n, 14)) if kinds[i] == "story"), None)
    start_c = 0
    generic_before = all(
        kinds[i] in ("matter", "optional") or _norm(chapters[i].title) == "opening" for i in range(story_first or 0)
    )
    if story_first is not None and (fiction or generic_before):
        start_c = story_first
        if keep_optional:  # a non-fiction book's own foreword/introduction comes before "Chapter 1"
            start_c = next((i for i in range(story_first) if kinds[i] == "optional"), story_first)
    else:
        while start_c < n - 1:
            k = kinds[start_c]
            if k == "matter" or (k == "optional" and not keep_optional):
                start_c += 1
                continue
            break
    for i in range(start_c):
        span.removed.append({"where": "front", "title": chapters[i].title, "words": chapters[i].word_count,
                             "reason": _why(kinds[i], fiction)})

    # --- paragraph trim inside the first narrated chapter ---
    start_p = 0
    first = chapters[start_c]
    if kinds[start_c] in ("neutral", "story", "optional"):
        cut = _front_cut(first.paragraphs)
        if cut:
            start_p = cut
            span.removed.append({"where": "front", "title": f"{first.title} (opening pages)",
                                 "words": _words(first.paragraphs[:cut]),
                                 "reason": "copyright, title page, dedication or epigraph text"})
    # The "Opening" chapter is entirely front matter when the cut swallowed all of its prose.
    if start_p >= len(first.paragraphs) and start_c < n - 1:
        start_c, start_p = start_c + 1, 0

    # --- end chapter ---
    end_c = n - 1
    while end_c > start_c:
        k = kinds[end_c]
        if k == "matter" or (k == "optional" and not keep_optional):
            span.removed.append({"where": "back", "title": chapters[end_c].title,
                                 "words": chapters[end_c].word_count, "reason": _why(k, fiction)})
            end_c -= 1
            continue
        break
    last = chapters[end_c]
    lo = start_p if end_c == start_c else 0
    keep = _back_cut(last.paragraphs[lo:]) + lo
    end_p: int | None = None
    if keep < len(last.paragraphs):
        end_p = max(keep - 1, lo)
        span.removed.append({"where": "back", "title": f"{last.title} (closing pages)",
                             "words": _words(last.paragraphs[keep:]),
                             "reason": "author bio, acknowledgments, excerpts or promotional text"})
    span.start = (start_c, start_p)
    span.end = (end_c, end_p) if end_p is not None else (end_c, len(last.paragraphs) - 1)

    # --- safety: never gut the book ---
    removed_words = sum(r["words"] for r in span.removed)
    if removed_words > total_words * MAX_TRIM_SHARE:
        span.notes.append("The cut looked too large, so only clearly labelled front and back matter was removed.")
        return _conservative(chapters, kinds, fiction, keep_optional)
    return span


def _conservative(chapters: list[Chapter], kinds: list[str], fiction: bool, keep_optional: bool) -> Span:
    span = Span(fiction=fiction)
    n = len(chapters)
    s = 0
    while s < n - 1 and (kinds[s] == "matter" or (kinds[s] == "optional" and not keep_optional)) \
            and kinds[s] != "story":
        span.removed.append({"where": "front", "title": chapters[s].title, "words": chapters[s].word_count,
                             "reason": _why(kinds[s], fiction)})
        s += 1
    e = n - 1
    while e > s and (kinds[e] == "matter" or (kinds[e] == "optional" and not keep_optional)):
        span.removed.append({"where": "back", "title": chapters[e].title, "words": chapters[e].word_count,
                             "reason": _why(kinds[e], fiction)})
        e -= 1
    span.start = (s, 0)
    span.end = (e, len(chapters[e].paragraphs) - 1)
    span.notes.append("The cut looked too large, so only clearly labelled front and back matter was removed.")
    return span


def _why(kind: str, fiction: bool) -> str:
    if kind == "optional":
        return "foreword, preface or introduction (kept only for non-fiction)"
    return "front or back matter (not part of the story)"


def apply_span(chapters: list[Chapter], span: Span) -> list[Chapter]:
    """Return new chapters containing only the narrated part."""
    if not chapters:
        return []
    sc, sp = span.start
    ec, ep = span.end if span.end is not None else (len(chapters) - 1, len(chapters[-1].paragraphs) - 1)
    out: list[Chapter] = []
    for i in range(sc, ec + 1):
        c = chapters[i]
        lo = sp if i == sc else 0
        hi = (ep + 1) if i == ec else len(c.paragraphs)
        paras = c.paragraphs[lo:hi]
        if not paras:
            continue
        # A chapter cut mid-way no longer starts at its heading, so don't announce a generic title.
        announce = c.announce and not (i == sc and lo > 0 and _norm(c.title) == "opening")
        if _norm(c.title) == "opening":
            announce = False
        out.append(Chapter(c.title, list(paras), announce))
    return out
