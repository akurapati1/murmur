"""End-to-end job pipeline: validate -> evaluate -> read -> narrate -> encode."""
from __future__ import annotations

import inspect
import json
import logging
import os
import re
import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pymupdf

from .ai import AIError, resolve_provider
from .audio import ChapterTrack, FFmpegError, assemble, float_to_pcm, silence, trim_silence
from .classifier import analyze_pdf, ocr_available
from .config import Settings
from .director import DirectorOptions, Script, direct_book
from .extract import Cancelled, OCRUnavailable, extract_book
from .models import Book, Chunk, JobStatus, Verdict
from .store import JobStore
from .tts import EngineUnavailable, create_engine
from .validation import validate_pdf
from .voices import effective_voice

log = logging.getLogger("murmur.pipeline")

# Share of the progress bar given to each stage.
P_VALIDATE, P_ANALYZE, P_EXTRACT, P_DIRECT, P_NARRATE, P_ENCODE = 0.01, 0.05, 0.12, 0.17, 0.92, 1.0

UpdateCb = Callable[[dict], None]


class _Progress:
    """Throttles database writes so a fast engine doesn't hammer SQLite."""

    def __init__(self, store: JobStore, job_id: str, on_update: UpdateCb | None):
        self.store, self.job_id, self.on_update = store, job_id, on_update
        self._last = 0.0
        self._stage = ""

    def set(self, stage: str, progress: float, message: str, eta: float | None = None, force: bool = False) -> None:
        now = time.monotonic()
        if not force and stage == self._stage and now - self._last < 0.8:
            return
        self._stage, self._last = stage, now
        fields = {"stage": stage, "progress": round(progress, 4), "message": message, "eta_sec": eta}
        self.store.update(self.job_id, **fields)
        if self.on_update:
            self.on_update(fields)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")[:60] or "audiobook"


def _render_cover(pdf: Path, target: Path) -> Path | None:
    try:
        doc = pymupdf.open(pdf)
        try:
            doc[0].get_pixmap(matrix=pymupdf.Matrix(0.8, 0.8)).save(target)
        finally:
            doc.close()
        return target
    except Exception:
        return None


def _accepts_direction(engine) -> bool:
    try:
        return "speed" in inspect.signature(engine.synthesize).parameters
    except (TypeError, ValueError):
        return False


def _synth_chunk(engine, chunk: Chunk, directed: bool = True):
    """Synthesize one chunk, retrying once with simplified text before giving up on it."""
    kwargs = {"speed": chunk.speed, "intensity": chunk.intensity} if directed else {}
    for attempt in (chunk.text, re.sub(r"[^\w\s.,!?'\"-]", " ", chunk.text)):
        if not attempt.strip():
            continue
        try:
            return engine.synthesize(attempt, **kwargs)
        except EngineUnavailable:
            raise
        except Exception as exc:
            log.warning("chunk failed (%s): %.60s", exc, attempt)
    return None


def process_job(
    job_id: str,
    store: JobStore,
    settings: Settings,
    cancel: threading.Event | None = None,
    on_update: UpdateCb | None = None,
) -> None:
    job = store.get(job_id)
    if job is None:
        return
    jobdir = settings.jobs_dir / job_id
    work = jobdir / "work"
    prog = _Progress(store, job_id, on_update)
    try:
        _run(job, store, settings, cancel, prog, jobdir, work)
    except Cancelled:
        if getattr(cancel, "shutdown", False):
            return  # server is stopping; leave the job as-is so it resumes on restart
        shutil.rmtree(work, ignore_errors=True)
        store.update(job_id, status=JobStatus.CANCELLED.value, message="Cancelled.", eta_sec=None)
    except (EngineUnavailable, OCRUnavailable, FFmpegError) as exc:
        store.update(job_id, status=JobStatus.FAILED.value, error=str(exc), message="Setup needed.", eta_sec=None)
    except Exception as exc:  # keep the worker alive whatever happens
        log.exception("job %s failed", job_id)
        store.update(job_id, status=JobStatus.FAILED.value, error=f"Unexpected error: {exc}",
                     message="Something went wrong.", eta_sec=None)


def _check(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled()


def _run(job, store: JobStore, settings: Settings, cancel, prog: _Progress, jobdir: Path, work: Path) -> None:
    job_id = job["id"]
    options = job.get("options") or {}
    pdf = Path(job["input_path"])
    store.update(job_id, status=JobStatus.RUNNING.value, error=None)
    _check(cancel)

    # 1. Is it a PDF?
    prog.set("validate", P_VALIDATE, "Checking the file", force=True)
    check = validate_pdf(pdf)
    if not check.ok:
        store.update(job_id, status=JobStatus.REJECTED.value, error=check.reason, message=check.reason, eta_sec=None)
        return

    # 2. Is it a book that works as audio?
    report = job.get("report")
    if report is None:
        prog.set("analyze", P_VALIDATE, "Looking through the pages", force=True)
        report = analyze_pdf(pdf, settings.sample_pages).to_dict()
        store.update(job_id, report=report)
    verdict = report["verdict"]
    prog.set("analyze", P_ANALYZE, "Evaluated the book", force=True)
    _check(cancel)

    if verdict in (Verdict.NEEDS_VISUALS.value, Verdict.NOT_A_BOOK.value) and not options.get("force"):
        store.update(job_id, status=JobStatus.NEEDS_CONFIRMATION.value, message=report["reasons"][0], eta_sec=None)
        return
    ocr = verdict == Verdict.SCANNED.value
    if ocr and not ocr_available():
        msg = ("This PDF is made of page photos, so it needs OCR. Install Tesseract "
               "(apt install tesseract-ocr / brew install tesseract) and upload it again.")
        store.update(job_id, status=JobStatus.REJECTED.value, error=msg, message=msg, eta_sec=None)
        return

    # 3. Pull out the text, chapter by chapter.
    work.mkdir(parents=True, exist_ok=True)
    book_file = work / "book.json"
    if book_file.exists():
        book = Book.from_dict(json.loads(book_file.read_text()))
    else:
        def on_extract(frac: float, msg: str) -> None:
            prog.set("extract", P_ANALYZE + (P_EXTRACT - P_ANALYZE) * frac, msg)

        book = extract_book(pdf, ocr=ocr, skip_back_matter=settings.skip_back_matter,
                            skip_captions=settings.skip_captions, title_hint=Path(job["filename"]).stem,
                            progress=on_extract, cancel=cancel)
        book_file.write_text(json.dumps(book.to_dict()))
    if book.word_count < 100:
        msg = "I couldn't find readable text in this PDF."
        store.update(job_id, status=JobStatus.REJECTED.value, error=msg, message=msg, eta_sec=None)
        return
    title = check.title or book.title
    store.update(job_id, title=title, author=check.author or book.author)

    # 4. Direct: where the story starts, and how every sentence should be performed.
    plan_file = work / "plan.json"
    if plan_file.exists():
        script = Script.from_dict(json.loads(plan_file.read_text()))
    else:
        def on_direct(frac: float, msg: str) -> None:
            prog.set("direct", P_EXTRACT + (P_DIRECT - P_EXTRACT) * frac, msg)

        prog.set("direct", P_EXTRACT, "Finding where the story begins", force=True)
        try:
            provider = resolve_provider(settings)
        except AIError as exc:
            log.warning("AI director unavailable: %s", exc)
            provider = None
        script = direct_book(book, settings, DirectorOptions.from_job(options, settings), provider=provider,
                             cache_path=work / "ai_cache.json", progress=on_direct, cancel=cancel)
        tmp = plan_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(script.to_dict()))
        os.replace(tmp, plan_file)
    store.update(job_id, direction=script.report)
    if not script.chapters:
        msg = "I couldn't find any story text to read in this PDF."
        store.update(job_id, status=JobStatus.REJECTED.value, error=msg, message=msg, eta_sec=None)
        return

    # 5. Narrate.
    plan = [c.chunks for c in script.chapters]
    chapter_titles = [c.title for c in script.chapters]
    total_chars = max(sum(len(ch.text) for chunks in plan for ch in chunks), 1)
    voice = effective_voice(settings, options.get("voice"))
    engine = create_engine(settings, engine=options.get("engine"), voice=voice, speed=options.get("speed"))
    directed = _accepts_direction(engine)
    audio_dir = work / "audio"
    tracks = [ChapterTrack(audio_dir, i) for i in range(len(plan))]
    done_chars = 0
    session_chars, session_start = 0, time.monotonic()
    skipped = 0
    sample_rate = 0

    for ci, (chapter_title, chunks, track) in enumerate(zip(chapter_titles, plan, tracks)):
        state = track.load()
        chapter_chars = sum(len(c.text) for c in chunks)
        if state["complete"]:
            done_chars += chapter_chars
            sample_rate = state["sample_rate"] or sample_rate
            continue
        f, state = track.open_append()
        start = state["chunks_done"]
        done_chars += sum(len(c.text) for c in chunks[:start])
        try:
            for j in range(start, len(chunks)):
                _check(cancel)
                chunk = chunks[j]
                samples = _synth_chunk(engine, chunk, directed)
                sample_rate = int(engine.sample_rate)
                if samples is None:
                    skipped += 1
                else:
                    # Trim the engine's own padding so the pause is exactly the one the director chose.
                    f.write(float_to_pcm(trim_silence(samples, sample_rate)))
                    f.write(silence(chunk.pause_after, sample_rate))
                track.commit(f, j + 1, sample_rate)
                done_chars += len(chunk.text)
                session_chars += len(chunk.text)
                elapsed = max(time.monotonic() - session_start, 1e-6)
                eta = (total_chars - done_chars) / (session_chars / elapsed) if session_chars else None
                prog.set("narrate", P_EXTRACT + (P_NARRATE - P_EXTRACT) * done_chars / total_chars,
                         f"Reading chapter {ci + 1} of {len(plan)}: {chapter_title[:60]}", eta)
            track.commit(f, len(chunks), sample_rate, complete=True)
        finally:
            f.close()

    # 6. Encode.
    fmt = options.get("format", "m4b")
    out_dir = jobdir / "output"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_path = out_dir / f"{_slug(title)}.{ 'zip' if fmt == 'zip' else fmt }"
    cover = _render_cover(pdf, work / "cover.jpg") if fmt != "zip" else None

    def on_encode(frac: float) -> None:
        prog.set("encode", P_NARRATE + (P_ENCODE - P_NARRATE) * frac, "Packaging your audiobook", None)

    prog.set("encode", P_NARRATE, "Packaging your audiobook", force=True)
    timings = assemble(
        tracks, chapter_titles, out_path, fmt=fmt, sample_rate=sample_rate or 24000,
        bitrate=settings.bitrate, book_title=title, author=check.author or book.author,
        cover=cover, work_dir=work / "encode", progress=on_encode,
    )
    duration = timings[-1]["end"] if timings else 0.0
    note = f" {skipped} passage(s) couldn't be read and were skipped." if skipped else ""
    store.update(
        job_id, status=JobStatus.DONE.value, stage="done", progress=1.0, eta_sec=None,
        message=f"Your audiobook is ready.{note}", output_path=str(out_path), output_format=fmt,
        duration_sec=duration, chapters=timings,
    )
    if not settings.keep_work:
        shutil.rmtree(work, ignore_errors=True)
