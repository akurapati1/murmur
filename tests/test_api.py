import dataclasses
import json

import pytest
from fastapi.testclient import TestClient

from murmur.audio import ffmpeg_available
from murmur.main import create_app
from tests.conftest import wait_for

needs_ffmpeg = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


def post_pdf(client, path, **form):
    with open(path, "rb") as f:
        return client.post("/api/jobs", files={"file": (path.name, f, "application/pdf")}, data=form)


def poll(client, job_id, statuses):
    return wait_for(lambda: client.get(f"/api/jobs/{job_id}").json(), statuses)


def test_health_and_voices(client):
    health = client.get("/api/health").json()
    assert health["status"] == "ok"
    assert {e["name"] for e in health["engines"]} == {"kokoro", "piper", "espeak"}
    voices = client.get("/api/voices").json()
    assert voices["engine"] == "dummy" and voices["voices"]


def test_home_page_is_served(client):
    res = client.get("/")
    assert res.status_code == 200 and "Murmur" in res.text


def test_text_file_is_rejected_immediately(client, tmp_path):
    f = tmp_path / "notes.pdf"
    f.write_text("definitely not a pdf")
    res = post_pdf(client, f)
    assert res.status_code == 415
    assert res.json()["detail"]["code"] == "not_pdf"
    assert client.get("/api/jobs").json() == []  # nothing was queued


def test_oversized_upload_is_refused(settings, novel):
    tiny = dataclasses.replace(settings, max_upload_mb=0)
    with TestClient(create_app(tiny)) as c:
        res = post_pdf(c, novel)
    assert res.status_code == 413


def test_bad_options_are_refused(client, novel):
    assert post_pdf(client, novel, format="wav").status_code == 400
    assert post_pdf(client, novel, voice="../../etc/passwd").status_code == 400


def test_unknown_job_is_404(client):
    assert client.get("/api/jobs/doesnotexist").status_code == 404
    assert client.get("/api/jobs/aaaaaaaaaaaa").status_code == 404


@needs_ffmpeg
def test_upload_convert_stream_and_download(client, novel):
    res = post_pdf(client, novel, voice="tone", speed="1.1")
    assert res.status_code == 201
    job_id = res.json()["id"]

    job = poll(client, job_id, {"done", "failed", "rejected"})
    assert job["status"] == "done", job
    assert job["has_output"] and len(job["chapters"]) == 4

    audio = client.get(f"/api/jobs/{job_id}/download")
    assert audio.status_code == 200 and audio.headers["content-type"] == "audio/mp4"
    assert len(audio.content) > 5000

    part = client.get(f"/api/jobs/{job_id}/download", headers={"Range": "bytes=0-99"})
    assert part.status_code == 206 and len(part.content) == 100  # seeking works in the player

    attach = client.get(f"/api/jobs/{job_id}/download?attachment=true")
    assert "attachment" in attach.headers["content-disposition"]

    with client.stream("GET", f"/api/jobs/{job_id}/events") as stream:
        first = next(line for line in stream.iter_lines() if line.startswith("data: "))
    assert json.loads(first[6:])["status"] == "done"

    assert [j["id"] for j in client.get("/api/jobs").json()] == [job_id]
    assert client.delete(f"/api/jobs/{job_id}").status_code == 204
    assert client.get(f"/api/jobs/{job_id}").status_code == 404


@needs_ffmpeg
def test_technical_book_asks_before_converting(client, technical):
    job_id = post_pdf(client, technical).json()["id"]
    job = poll(client, job_id, {"needs_confirmation", "done", "failed"})
    assert job["status"] == "needs_confirmation"
    assert job["report"]["verdict"] == "needs_visuals"
    assert client.get(f"/api/jobs/{job_id}/download").status_code == 404

    assert client.post(f"/api/jobs/{job_id}/confirm").status_code == 200
    assert poll(client, job_id, {"done", "failed"})["status"] == "done"
    assert client.post(f"/api/jobs/{job_id}/confirm").status_code == 409


def test_discarding_a_technical_book(client, technical):
    job_id = post_pdf(client, technical).json()["id"]
    poll(client, job_id, {"needs_confirmation"})
    assert client.post(f"/api/jobs/{job_id}/cancel").json()["status"] == "cancelled"


def test_preview_returns_a_wav(client):
    res = client.post("/api/preview", json={"voice": "tone", "speed": 1.0})
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/wav" and res.content[:4] == b"RIFF"


def test_preview_explains_missing_engine(settings):
    no_engine = dataclasses.replace(settings, tts_engine="kokoro")
    with TestClient(create_app(no_engine)) as c:
        res = c.post("/api/preview", json={})
    assert res.status_code == 503
    assert "kokoro" in res.json()["detail"]["message"].lower()
