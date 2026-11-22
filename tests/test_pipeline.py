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

    def synthesize(self, text, **kwargs):
        self.calls += 1
        if self.stop_after and self.calls == self.stop_after:
            self.token.shutdown = True
            self.token.set()
        return super().synthesize(text, **kwargs)


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


# ---- the AI director in the pipeline -----------------------------------------------------------------
import json  # noqa: E402

from tests.test_director import FakeAI  # noqa: E402


def keep_work(settings):
    return dataclasses.replace(settings, keep_work=True)


@pytest.mark.parametrize("outline", [True, False])
def test_story_starts_at_the_prologue(store, settings, tmp_path, outline):
    book = pdfs.make_novel_with_matter(tmp_path / "wrapped.pdf", outline=outline)
    s = keep_work(settings)
    job = run(store, s, new_job(store, s, book))
    assert job["status"] == "done", job
    assert [c["title"] for c in job["chapters"]] == ["Prologue", "Chapter 1", "Chapter 2", "Chapter 3"]
    direction = job["direction"]
    assert direction["story_starts_with"]["chapter"] == "Prologue"
    assert direction["removed_words"] > 20 and direction["provider"] == "rules"
    plan = json.loads((s.jobs_dir / job["id"] / "work" / "plan.json").read_text())
    spoken = " ".join(c["text"] for ch in plan["chapters"] for c in ch["chunks"])
    for furniture in ("ISBN", "All rights", "Tidewater", "my mother", "thank my editor", "awriter", "sea does not care"):
        assert furniture not in spoken, furniture
    assert "tea is hot" in spoken and "Chapter one." in spoken
    assert "*" not in spoken


def test_opening_credits_can_be_kept(store, settings, tmp_path):
    book = pdfs.make_novel_with_matter(tmp_path / "wrapped.pdf", outline=False)
    job_id = new_job(store, settings, book, start_at_story=False)
    job = run(store, settings, job_id)
    assert job["status"] == "done" and job["chapters"][0]["title"] == "Opening"


def test_pauses_in_the_audiobook_are_the_ones_the_director_chose(store, settings, tmp_path):
    book = pdfs.make_novel_with_matter(tmp_path / "wrapped.pdf", outline=False)
    s = keep_work(settings)
    job = run(store, s, new_job(store, s, book, ai="off"))
    work = s.jobs_dir / job["id"] / "work"
    plan = json.loads((work / "plan.json").read_text())
    pauses = sum(c["pause_after"] for ch in plan["chapters"] for c in ch["chunks"])
    pcm_seconds = sum(
        json.loads(p.read_text())["bytes"] / 2 / json.loads(p.read_text())["sample_rate"]
        for p in (work / "audio").glob("ch*.json"))
    # speech itself is trimmed tight, so the total is the pauses plus the speech and nothing else
    assert pcm_seconds >= pauses
    raw_speech = sum(len(c["text"]) / 120 / max(c["speed"], 0.1) for ch in plan["chapters"] for c in ch["chunks"])
    assert pcm_seconds <= pauses + raw_speech + 1.0


def test_ai_directed_job_and_resume_costs_nothing_extra(store, settings, tmp_path, monkeypatch):
    book = pdfs.make_novel_with_matter(tmp_path / "wrapped.pdf", outline=True)
    s = keep_work(settings)
    ai = FakeAI(locate={}, annotate={"paragraphs": [{"i": 0, "pause": "long"}], "pronunciations": {"Marta": "MAR-tuh"}})
    ai.locate = lambda user: json.dumps({"story_start": {"chapter": 2, "paragraph": 0},
                                         "story_end": {"chapter": 5, "paragraph": 9}, "reasons": ["Prologue first."]})
    monkeypatch.setattr(pipeline, "resolve_provider", lambda settings: ai)

    token = CancelToken()
    engine = CountingEngine(stop_after=10, token=token)
    monkeypatch.setattr(pipeline, "create_engine", lambda *a, **k: engine)
    job_id = new_job(store, s, book)
    job = run(store, s, job_id, token)
    assert job["status"] == "running" and job["direction"]["provider"] == "fake"
    assert job["direction"]["story_found_by"] == "ai" and job["direction"]["reasons"] == ["Prologue first."]
    calls_after_first_run = dict(ai.calls)
    assert calls_after_first_run["locate"] == 1 and calls_after_first_run["annotate"] >= 4

    monkeypatch.setattr(pipeline, "create_engine", lambda *a, **k: CountingEngine())
    job = run(store, s, job_id)
    assert job["status"] == "done"
    assert ai.calls == calls_after_first_run  # the saved plan was reused: no new AI calls
    assert job["direction"]["pronunciations"] == {"Marta": "MAR-tuh"}


def test_a_broken_ai_still_produces_the_audiobook(store, settings, tmp_path, monkeypatch):
    book = pdfs.make_novel_with_matter(tmp_path / "wrapped.pdf")
    monkeypatch.setattr(pipeline, "resolve_provider", lambda settings: FakeAI(fail=True))
    job = run(store, settings, new_job(store, settings, book))
    assert job["status"] == "done"
    assert job["direction"]["ai_failures"] >= 1
    assert [c["title"] for c in job["chapters"]][0] == "Prologue"


# ---- cloned voices -------------------------------------------------------------------------------------
class FakeClone(DummyEngine):
    """Records which reference clip it was asked to imitate."""

    seen: list = []

    def __init__(self, reference, engine="auto", speed=1.0):
        super().__init__(speed=speed)
        self.reference = reference
        FakeClone.seen.append(Path(reference))


def test_narration_uses_the_uploaded_voice(store, settings, novel, tmp_path, monkeypatch):
    import murmur.tts as tts
    from murmur.voices import VoiceLibrary
    from tests.speechwav import speechlike, write_wav

    voice = VoiceLibrary(settings).add("Me", write_wav(tmp_path / "me.wav", speechlike(14)), consent=True)
    FakeClone.seen = []
    monkeypatch.setattr(tts, "CloneEngine", FakeClone)
    real = dataclasses.replace(settings, tts_engine="auto")
    job = run(store, real, new_job(store, real, novel, engine=None, voice="Me"))
    assert job["status"] == "done", job
    assert FakeClone.seen and FakeClone.seen[0] == VoiceLibrary(settings).reference_path(voice.id)


def test_saved_default_voice_is_used_when_none_is_chosen(store, settings, novel, tmp_path, monkeypatch):
    import murmur.tts as tts
    from murmur.voices import VoiceLibrary, set_default_voice
    from tests.speechwav import speechlike, write_wav

    voice = VoiceLibrary(settings).add("Default me", write_wav(tmp_path / "me.wav", speechlike(14)), consent=True)
    set_default_voice(settings, voice.id)
    FakeClone.seen = []
    monkeypatch.setattr(tts, "CloneEngine", FakeClone)
    real = dataclasses.replace(settings, tts_engine="auto")
    job = run(store, real, new_job(store, real, novel, engine=None, voice=None))
    assert job["status"] == "done" and FakeClone.seen


def test_missing_cloning_engine_gives_clear_instructions(store, settings, novel, tmp_path, monkeypatch):
    from murmur.voices import VoiceLibrary
    from tests.speechwav import speechlike, write_wav

    voice = VoiceLibrary(settings).add("Me", write_wav(tmp_path / "me.wav", speechlike(14)), consent=True)
    monkeypatch.setattr("importlib.util.find_spec", lambda name, *a, **k: None)
    real = dataclasses.replace(settings, tts_engine="auto")
    job = run(store, real, new_job(store, real, novel, engine=None, voice=voice.id))
    assert job["status"] == "failed"
    assert "voice-cloning engine" in job["error"] and 'pip install -e ".[clone]"' in job["error"]
