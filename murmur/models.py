"""Plain data types shared across the pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Verdict(str, Enum):
    AUDIOBOOK_FRIENDLY = "audiobook_friendly"
    BORDERLINE = "borderline"
    NEEDS_VISUALS = "needs_visuals"
    NOT_A_BOOK = "not_a_book"
    SCANNED = "scanned"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    NEEDS_CONFIRMATION = "needs_confirmation"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


ACTIVE_STATUSES = {JobStatus.QUEUED.value, JobStatus.RUNNING.value}
TERMINAL_STATUSES = {
    JobStatus.NEEDS_CONFIRMATION.value,
    JobStatus.DONE.value,
    JobStatus.FAILED.value,
    JobStatus.CANCELLED.value,
    JobStatus.REJECTED.value,
}


@dataclass
class AnalysisReport:
    verdict: Verdict
    visual_score: float
    page_count: int
    sampled_pages: int
    estimated_words: int
    estimated_hours: float
    reasons: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    needs_ocr: bool = False

    @property
    def convertible(self) -> bool:
        return self.verdict in (Verdict.AUDIOBOOK_FRIENDLY, Verdict.BORDERLINE)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        d["convertible"] = self.convertible
        return d


@dataclass
class Chapter:
    title: str
    paragraphs: list[str]
    announce: bool = True  # speak the title aloud before the chapter (False for synthetic titles)

    @property
    def word_count(self) -> int:
        return sum(len(p.split()) for p in self.paragraphs)


@dataclass
class Book:
    title: str
    author: str
    chapters: list[Chapter]

    @property
    def word_count(self) -> int:
        return sum(c.word_count for c in self.chapters)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "author": self.author,
            "chapters": [
                {"title": c.title, "paragraphs": c.paragraphs, "announce": c.announce} for c in self.chapters
            ],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Book":
        return cls(
            title=d.get("title", ""),
            author=d.get("author", ""),
            chapters=[Chapter(c["title"], list(c["paragraphs"]), c.get("announce", True)) for c in d["chapters"]],
        )


@dataclass
class Chunk:
    """One unit of narration: a sentence (or clause) plus exactly how it should be performed."""

    text: str
    pause_after: float = 0.0
    speed: float = 1.0  # relative to the job's base speed
    intensity: float = 0.5  # 0 = very calm, 1 = very expressive (used by engines that support it)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Chunk":
        return cls(d["text"], float(d.get("pause_after", 0.0)), float(d.get("speed", 1.0)),
                   float(d.get("intensity", 0.5)))
