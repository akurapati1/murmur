import dataclasses
import threading
from pathlib import Path

import pytest

import murmur.pipeline as pipeline
from murmur.audio import ffmpeg_available
from murmur.classifier import ocr_available
from murmur.jobs import CancelToken
from murmur.pipeline import process_job
from murmur.tts.dummy_engine import DummyEngine
from tests import pdfs
from tests.conftest import new_job

pytestmark = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")


def run(store, settings, job_id, cancel=None):
    process_job(job_id, store, settings, cancel)
    return store.get(job_id)


def test_novel_becomes_an_audiobook(store, settings, novel):
    job = run(store, settings, new_job(store, settings, novel), None)
    assert job["status"] == "done", job
    assert job["progress"] == 1.0
    assert job["title"] == "The Lighthouse Keeper"
    assert job["report"]["verdict"] == "audiobook_friendly"
    assert [c["title"] for c in job["chapters"]] == [f"Chapter {i}" for i in range(1, 5)]
    assert Path(job["output_path"]).stat().st_size > 5000
    assert job["duration_sec"] > 60
    assert not (settings.jobs_dir / job["id"] / "work").exists()  # scratch space cleaned up


def test_technical_book_waits_for_confirmation_then_converts(store, settings, technical):
    job_id = new_job(store, settings, technical)
    job = run(store, settings, job_id)
    assert job["status"] == "needs_confirmation"
    assert job["report"]["verdict"] == "needs_visuals"
    assert job["output_path"] is None

    store.update(job_id, options={**job["options"], "force": True})
    job = run(store, settings, job_id)
    assert job["status"] == "done"


def test_non_pdf_is_rejected(store, settings, tmp_path):
    fake = tmp_path / "x.pdf"
    fake.write_text("hello")
    job = run(store, settings, new_job(store, settings, fake))
    assert job["status"] == "rejected"
    assert "PDF" in job["error"]


def test_scanned_pdf_without_ocr_is_rejected(store, settings, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "ocr_available", lambda: False)
    scan = pdfs.make_scanned(tmp_path / "scan.pdf")
    job = run(store, settings, new_job(store, settings, scan))
    assert job["status"] == "rejected"
    assert "OCR" in job["error"]


def test_missing_engine_gives_a_clear_failure(store, settings, novel):
    no_engine = dataclasses.replace(settings, tts_engine="kokoro")
    job = run(store, no_engine, new_job(store, no_engine, novel, engine=None))
    assert job["status"] == "failed"
    assert "kokoro" in job["error"].lower()


def test_user_cancel(store, settings, novel):
    token = threading.Event()
    token.set()
    job = run(store, settings, new_job(store, settings, novel), token)
    assert job["status"] == "cancelled"


@pytest.mark.parametrize("fmt,ext", [("mp3", ".mp3"), ("zip", ".zip")])
def test_other_formats(store, settings, novel, fmt, ext):
    job = run(store, settings, new_job(store, settings, novel, format=fmt))
    assert job["status"] == "done"
    assert job["output_path"].endswith(ext)


class CountingEngine(DummyEngine):
    """Counts synthesis calls and can pull the plug after N of them."""

    def __init__(self, stop_after=None, token=None):
        super().__init__()
        self.calls = 0
        self.stop_after, self.token = stop_after, token

    def synthesize(self, text):
        self.calls += 1
        if self.stop_after and self.calls == self.stop_after:
            self.token.shutdown = True
            self.token.set()
        return super().synthesize(text)


def test_interrupted_job_resumes_without_redoing_work(store, settings, novel, monkeypatch):
    # Baseline: how many chunks does the whole book take?
    baseline = CountingEngine()
    monkeypatch.setattr(pipeline, "create_engine", lambda *a, **k: baseline)
    run(store, settings, new_job(store, settings, novel))
    total = baseline.calls
    assert total > 20

    # Interrupt part-way through (as a server shutdown would) ...
    token = CancelToken()
    first = CountingEngine(stop_after=15, token=token)
    monkeypatch.setattr(pipeline, "create_engine", lambda *a, **k: first)
    job_id = new_job(store, settings, novel)
    job = run(store, settings, job_id, token)
    assert job["status"] == "running"  # left as-is so it can be resumed
    assert first.calls == 15

    # ... then restart: only the remaining chunks are synthesized.
    second = CountingEngine()
    monkeypatch.setattr(pipeline, "create_engine", lambda *a, **k: second)
    job = run(store, settings, job_id)
    assert job["status"] == "done"
    assert first.calls + second.calls == total


@pytest.mark.skipif(not ocr_available(), reason="Tesseract not installed")
def test_scanned_book_is_read_with_ocr(store, settings, tmp_path):
    scan = pdfs.make_scanned_text(tmp_path / "scan.pdf")
    job = run(store, settings, new_job(store, settings, scan))
    assert job["status"] == "done", job
    assert job["report"]["verdict"] == "scanned"
    assert job["duration_sec"] > 30


def test_title_falls_back_to_the_uploaded_filename(store, settings, tmp_path):
    # This PDF has no title metadata; the audiobook must be named after the file the user chose,
    # not after Murmur's internal storage name.
    src = pdfs.make_technical(tmp_path / "my_great-book.pdf")
    job = run(store, settings, new_job(store, settings, src, force=True))
    assert job["status"] == "done"
    assert job["title"] == "My Great Book"
    assert Path(job["output_path"]).name == "My-Great-Book.m4b"
