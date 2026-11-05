"""Audio plumbing: resumable per-chapter PCM tracks and ffmpeg assembly (m4b / mp3 / zip)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

SAMPLE_WIDTH = 2  # bytes per mono int16 sample
CHAPTER_GAP_SECONDS = 1.2

ProgressCb = Callable[[float], None]


class FFmpegError(RuntimeError):
    pass


def float_to_pcm(samples: np.ndarray) -> bytes:
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def trim_silence(samples: np.ndarray, sample_rate: int, threshold: float = 0.006, keep: float = 0.02,
                 fade: float = 0.008) -> np.ndarray:
    """Cut the engine's own leading/trailing silence so the pauses the director chose are exact.

    ``keep`` leaves a hair of room so word edges aren't clipped; ``fade`` removes clicks at the cut.
    """
    x = np.asarray(samples, dtype=np.float32)
    if x.size == 0:
        return x
    loud = np.flatnonzero(np.abs(x) > threshold)
    if loud.size == 0:
        return x[: int(0.05 * sample_rate)]  # all silence: keep a blink so timing stays sane
    pad = int(keep * sample_rate)
    out = x[max(loud[0] - pad, 0): min(loud[-1] + pad + 1, x.size)].copy()
    n = min(int(fade * sample_rate), out.size // 2)
    if n:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        out[:n] *= ramp
        out[-n:] *= ramp[::-1]
    return out


def silence(seconds: float, sample_rate: int) -> bytes:
    return b"\x00" * (int(seconds * sample_rate) * SAMPLE_WIDTH)


class ChapterTrack:
    """Append-only raw audio for one chapter, with a checkpoint file for crash recovery."""

    def __init__(self, directory: Path, index: int):
        directory.mkdir(parents=True, exist_ok=True)
        self.pcm_path = directory / f"ch{index:04d}.pcm"
        self.meta_path = directory / f"ch{index:04d}.json"

    def load(self) -> dict:
        if self.meta_path.exists() and self.pcm_path.exists():
            try:
                return json.loads(self.meta_path.read_text())
            except json.JSONDecodeError:
                pass
        return {"chunks_done": 0, "bytes": 0, "complete": False, "sample_rate": 0}

    def open_append(self):
        """Open for writing, discarding any bytes written after the last checkpoint."""
        state = self.load()
        mode = "r+b" if self.pcm_path.exists() else "w+b"
        f = open(self.pcm_path, mode)
        f.truncate(state["bytes"])
        f.seek(state["bytes"])
        return f, state

    def commit(self, f, chunks_done: int, sample_rate: int, complete: bool = False) -> None:
        f.flush()
        tmp = self.meta_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "chunks_done": chunks_done, "bytes": f.tell(), "complete": complete, "sample_rate": sample_rate,
        }))
        os.replace(tmp, self.meta_path)

    @property
    def size_bytes(self) -> int:
        return self.pcm_path.stat().st_size if self.pcm_path.exists() else 0


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _escape_meta(value: str) -> str:
    for ch in ("\\", "=", ";", "#", "\n"):
        value = value.replace(ch, "\\" + ch)
    return value


def _run_ffmpeg(args: list[str], total_seconds: float, progress: ProgressCb | None) -> None:
    if shutil.which("ffmpeg") is None:
        raise FFmpegError("ffmpeg isn't installed. Install it from https://ffmpeg.org/download.html")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostats", "-progress", "pipe:1", *args]
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            if progress and total_seconds > 0 and line.startswith(("out_time_us=", "out_time_ms=")):
                try:
                    micro = int(line.split("=", 1)[1])
                except ValueError:
                    continue
                progress(min(micro / 1_000_000 / total_seconds, 1.0))
        rc = proc.wait()
        if rc != 0:
            err.seek(0)
            raise FFmpegError(err.read().decode(errors="replace").strip() or f"ffmpeg exited with code {rc}")
    if progress:
        progress(1.0)


def assemble(
    tracks: list[ChapterTrack],
    titles: list[str],
    out_path: Path,
    *,
    fmt: str,
    sample_rate: int,
    bitrate: str,
    book_title: str,
    author: str,
    cover: Path | None = None,
    work_dir: Path,
    progress: ProgressCb | None = None,
) -> list[dict]:
    """Join chapter tracks and encode. Returns chapter timings in seconds."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    gap = silence(CHAPTER_GAP_SECONDS, sample_rate)
    bytes_per_second = sample_rate * SAMPLE_WIDTH

    timings: list[dict] = []
    joined = work_dir / "all.pcm"
    cursor = 0
    with open(joined, "wb") as dst:
        for i, (track, title) in enumerate(zip(tracks, titles)):
            start = cursor
            with open(track.pcm_path, "rb") as src:
                shutil.copyfileobj(src, dst, 1024 * 1024)
            cursor += track.size_bytes
            if i < len(tracks) - 1:
                dst.write(gap)
                cursor += len(gap)
            end_bytes = cursor - (len(gap) if i < len(tracks) - 1 else 0)
            timings.append({"title": title, "start": start / bytes_per_second, "end": end_bytes / bytes_per_second})
    total_seconds = cursor / bytes_per_second

    raw_in = ["-f", "s16le", "-ar", str(sample_rate), "-ac", "1"]

    if fmt == "zip":
        mp3_dir = work_dir / "mp3"
        mp3_dir.mkdir(exist_ok=True)
        files = []
        for i, (track, title) in enumerate(zip(tracks, titles), start=1):
            safe = "".join(c if c.isalnum() or c in " -_" else "" for c in title).strip()[:60] or f"Chapter {i}"
            target = mp3_dir / f"{i:02d} - {safe}.mp3"
            seconds = track.size_bytes / bytes_per_second
            _run_ffmpeg(
                [*raw_in, "-i", str(track.pcm_path), "-c:a", "libmp3lame", "-b:a", bitrate,
                 "-metadata", f"title={title}", "-metadata", f"album={book_title}",
                 "-metadata", f"artist={author}", "-metadata", f"track={i}", str(target)],
                seconds, (lambda p, base=i - 1: progress((base + p) / len(tracks))) if progress else None,
            )
            files.append(target)
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_STORED) as z:
            for f in files:
                z.write(f, f.name)
        return timings

    meta = work_dir / "chapters.ffmeta"
    lines = [";FFMETADATA1", f"title={_escape_meta(book_title)}", f"album={_escape_meta(book_title)}"]
    if author:
        lines.append(f"artist={_escape_meta(author)}")
    lines.append("genre=Audiobook")
    for t in timings:
        lines += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={int(t['start'] * 1000)}",
                  f"END={max(int(t['end'] * 1000), int(t['start'] * 1000) + 1)}",
                  f"title={_escape_meta(t['title'])}"]
    meta.write_text("\n".join(lines) + "\n", encoding="utf-8")

    codec = ["-c:a", "aac", "-b:a", bitrate] if fmt == "m4b" else ["-c:a", "libmp3lame", "-b:a", bitrate]
    container = ["-movflags", "+faststart", "-f", "ipod"] if fmt == "m4b" else ["-id3v2_version", "3", "-f", "mp3"]

    def build(with_cover: bool) -> list[str]:
        args = [*raw_in, "-i", str(joined), "-i", str(meta)]
        if with_cover:
            args += ["-i", str(cover)]
        args += ["-map", "0:a"]
        if with_cover:
            args += ["-map", "2:v", "-c:v", "mjpeg", "-disposition:v:0", "attached_pic"]
        args += ["-map_metadata", "1", "-map_chapters", "1", *codec, *container, str(out_path)]
        return args

    if cover and Path(cover).exists():
        try:
            _run_ffmpeg(build(True), total_seconds, progress)
            return timings
        except FFmpegError:
            pass  # some builds reject cover art in this container; retry without it
    _run_ffmpeg(build(False), total_seconds, progress)
    return timings
