import json
import subprocess
import zipfile

import numpy as np
import pytest

from murmur.audio import ChapterTrack, assemble, ffmpeg_available, float_to_pcm, silence

needs_ffmpeg = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")
SR = 16000


def test_float_to_pcm_clips_and_scales():
    pcm = np.frombuffer(float_to_pcm(np.array([0.0, 1.0, -1.0, 5.0], dtype=np.float32)), dtype="<i2")
    assert list(pcm) == [0, 32767, -32767, 32767]


def test_silence_length():
    assert len(silence(0.5, SR)) == SR // 2 * 2


def test_track_resumes_from_last_checkpoint(tmp_path):
    track = ChapterTrack(tmp_path, 0)
    f, state = track.open_append()
    assert state["chunks_done"] == 0
    f.write(b"\x01\x00" * 100)
    track.commit(f, 1, SR)
    f.write(b"\x02\x00" * 50)  # written after the checkpoint, then the "crash"
    f.close()

    f2, state2 = ChapterTrack(tmp_path, 0).open_append()
    assert state2["chunks_done"] == 1
    assert f2.tell() == 200  # the un-checkpointed tail was discarded
    f2.close()


def _tracks(tmp_path, seconds=(2.0, 3.0)):
    tracks = []
    for i, secs in enumerate(seconds):
        t = ChapterTrack(tmp_path / "audio", i)
        f, _ = t.open_append()
        tone = 0.1 * np.sin(2 * np.pi * 220 * np.arange(int(secs * SR)) / SR).astype(np.float32)
        f.write(float_to_pcm(tone))
        t.commit(f, 1, SR, complete=True)
        f.close()
        tracks.append(t)
    return tracks


def _probe(path, *args):
    out = subprocess.run(["ffprobe", "-v", "error", "-of", "json", *args, str(path)], capture_output=True, text=True)
    return json.loads(out.stdout)


@needs_ffmpeg
def test_m4b_has_chapters_and_correct_length(tmp_path):
    tracks = _tracks(tmp_path)
    out = tmp_path / "book.m4b"
    timings = assemble(tracks, ["One", "Two"], out, fmt="m4b", sample_rate=SR, bitrate="48k",
                       book_title="Test Book", author="Someone", work_dir=tmp_path / "w")
    assert [t["title"] for t in timings] == ["One", "Two"]
    assert timings[1]["start"] > timings[0]["end"]  # a gap separates chapters
    info = _probe(out, "-show_chapters", "-show_format")
    assert len(info["chapters"]) == 2
    assert info["chapters"][0]["tags"]["title"] == "One"
    assert float(info["format"]["duration"]) == pytest.approx(2.0 + 1.2 + 3.0, abs=0.3)
    assert info["format"]["tags"]["title"] == "Test Book"


@needs_ffmpeg
def test_m4b_with_cover_art(tmp_path):
    import pymupdf

    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 80, 120), False)
    pix.set_rect(pix.irect, (200, 100, 30))
    cover = tmp_path / "cover.jpg"
    pix.save(cover)
    out = tmp_path / "covered.m4b"
    assemble(_tracks(tmp_path), ["One", "Two"], out, fmt="m4b", sample_rate=SR, bitrate="48k",
             book_title="Covered", author="", cover=cover, work_dir=tmp_path / "w")
    assert out.stat().st_size > 1000
    assert len(_probe(out, "-show_chapters")["chapters"]) == 2


@needs_ffmpeg
def test_mp3_output(tmp_path):
    out = tmp_path / "book.mp3"
    assemble(_tracks(tmp_path), ["One", "Two"], out, fmt="mp3", sample_rate=SR, bitrate="48k",
             book_title="T", author="A", work_dir=tmp_path / "w")
    assert float(_probe(out, "-show_format")["format"]["duration"]) > 5


@needs_ffmpeg
def test_zip_has_one_mp3_per_chapter(tmp_path):
    out = tmp_path / "book.zip"
    assemble(_tracks(tmp_path), ["One", "Two: The Return"], out, fmt="zip", sample_rate=SR, bitrate="48k",
             book_title="T", author="A", work_dir=tmp_path / "w")
    names = zipfile.ZipFile(out).namelist()
    assert len(names) == 2 and names[0].startswith("01 - One") and names[1].startswith("02 - Two")
