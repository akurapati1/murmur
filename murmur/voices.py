"""Your voices: upload any recording, give it a name, and narrate whole books in it.

A "voice" here is a cleaned reference clip (mono WAV, a few seconds of clear speech) stored with a
name. The cloning engines in :mod:`murmur.tts.clone_engine` imitate whoever is speaking in that clip.

Library layout (``<data_dir>/voices/``)::

    clone-3fa9c1d2e7/
        voice.json       name, created_at, duration, warnings...
        reference.wav    the prepared clip
    ../prefs.json        {"default_voice": "clone-3fa9c1d2e7"}
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import uuid
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import Settings

SAMPLE_RATE = 24000
MIN_SPEECH_SECONDS = 5.0
IDEAL_SECONDS = (10.0, 30.0)
MAX_REFERENCE_SECONDS = 28.0
MAX_DECODE_SECONDS = 20 * 60
CLONE_PREFIX = "clone-"
_ID_RE = re.compile(r"^clone-[a-f0-9]{10}$")


class VoiceError(ValueError):
    """A problem with an uploaded voice that the user can fix; the message says how."""


@dataclass
class Voice:
    id: str
    name: str
    created_at: float
    duration_sec: float
    source_name: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_clone_id(voice: str | None) -> bool:
    return bool(voice) and voice.startswith(CLONE_PREFIX)  # type: ignore[union-attr]


def clean_name(name: str) -> str:
    name = re.sub(r"\s+", " ", (name or "").strip())
    name = "".join(ch for ch in name if ch.isprintable())
    if not name:
        raise VoiceError("Give this voice a name, for example \"My voice\".")
    if len(name) > 40:
        raise VoiceError("Voice names can be up to 40 characters.")
    return name


# ---- audio preparation ------------------------------------------------------------------------------
def _decode(path: Path) -> np.ndarray:
    if shutil.which("ffmpeg") is None:
        raise VoiceError("ffmpeg isn't installed, and it's needed to read audio files. "
                         "Install it from https://ffmpeg.org/download.html")
    try:
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-vn", "-t", str(MAX_DECODE_SECONDS),
             "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"],
            capture_output=True, timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        raise VoiceError("That file took too long to read. Try a shorter recording.") from exc
    if proc.returncode != 0 or not proc.stdout:
        raise VoiceError("I couldn't read that as audio. MP3, WAV, M4A, OGG, FLAC and most video files work.")
    return np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)


def _frame_rms(x: np.ndarray, frame: int) -> np.ndarray:
    n = x.size // frame
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    return np.sqrt(np.mean(np.square(x[: n * frame].reshape(n, frame)), axis=1))


def prepare_reference(samples: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Pick the best stretch of clear speech, squeeze long silences, and normalise the level.

    Returns the clip and a list of plain-language warnings. Raises :class:`VoiceError` if there is not
    enough speech to work with.
    """
    warnings: list[str] = []
    x = np.nan_to_num(samples.astype(np.float32))
    if x.size < SAMPLE_RATE:
        raise VoiceError("That recording is too short. Record at least 10 seconds of normal speech.")
    clipped = float(np.mean(np.abs(x) > 0.99))
    if clipped > 0.005:
        warnings.append("The recording is distorted (too loud). Re-record a bit further from the microphone.")
    x = x - float(np.mean(x))  # remove DC offset

    frame = int(0.03 * SAMPLE_RATE)
    rms = _frame_rms(x, frame)
    if rms.size == 0 or float(np.max(rms)) < 0.003:
        raise VoiceError("That recording is almost silent. Check the microphone and try again.")
    floor, ceil = float(np.percentile(rms, 10)), float(np.percentile(rms, 95))
    gate = max(0.004, floor * 2.5, ceil * 0.08)
    voiced = rms > gate
    if ceil / max(floor, 1e-6) < 8:
        warnings.append("There is background noise under the voice. A quiet room gives a much better copy.")

    # Squeeze silences longer than 0.35 s down to 0.35 s so more of the clip is actual speech.
    keep = np.ones(x.size, dtype=bool)
    max_gap = int(0.35 * SAMPLE_RATE / frame)
    run = 0
    for i, v in enumerate(voiced):
        run = 0 if v else run + 1
        if run > max_gap:
            keep[i * frame: (i + 1) * frame] = False
    x = x[keep] if keep.sum() > frame else x
    rms = _frame_rms(x, frame)
    voiced = rms > gate
    speech_seconds = float(voiced.sum() * frame / SAMPLE_RATE)
    if speech_seconds < MIN_SPEECH_SECONDS:
        if ceil / max(floor, 1e-6) < 3:
            raise VoiceError("The background noise is as loud as the voice, so I can't separate them. "
                             "Record in a quiet room, close to the microphone.")
        raise VoiceError(f"I only found {speech_seconds:.0f} seconds of speech. Please provide at least "
                         f"{int(MIN_SPEECH_SECONDS)} seconds, ideally 10 to 30, of you talking naturally.")

    # Choose the cleanest window of at most MAX_REFERENCE_SECONDS.
    win = int(MAX_REFERENCE_SECONDS * SAMPLE_RATE)
    if x.size > win:
        step = int(2 * SAMPLE_RATE)
        best, best_score = 0, -1.0
        for start in range(0, x.size - win + 1, step):
            seg = x[start: start + win]
            seg_rms = _frame_rms(seg, frame)
            score = float(np.mean(seg_rms > gate)) - 5 * float(np.mean(np.abs(seg) > 0.99))
            if score > best_score:
                best, best_score = start, score
        x = x[best: best + win]
    # Trim the outer silence.
    r = _frame_rms(x, frame)
    idx = np.flatnonzero(r > gate)
    if idx.size:
        x = x[max(int(idx[0]) - 1, 0) * frame: (int(idx[-1]) + 2) * frame]

    peak = float(np.max(np.abs(x)))
    if peak > 0:
        x = x * (0.9 / peak)
    seconds = x.size / SAMPLE_RATE
    if seconds < IDEAL_SECONDS[0]:
        warnings.append(f"Only {seconds:.0f} seconds of speech were usable. 10 to 30 seconds gives a closer match.")
    return x.astype(np.float32), warnings


def _write_wav(path: Path, samples: np.ndarray) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)


# ---- library ----------------------------------------------------------------------------------------
class VoiceLibrary:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = settings.voices_dir
        self.root.mkdir(parents=True, exist_ok=True)

    # -- reading --
    def _load(self, voice_id: str) -> Voice | None:
        meta = self.root / voice_id / "voice.json"
        try:
            d = json.loads(meta.read_text())
            return Voice(d["id"], d["name"], d["created_at"], d["duration_sec"], d.get("source_name", ""),
                         list(d.get("warnings", [])))
        except (OSError, KeyError, json.JSONDecodeError):
            return None

    def list(self) -> list[Voice]:
        found = [v for p in self.root.iterdir() if p.is_dir() and _ID_RE.match(p.name)
                 if (v := self._load(p.name)) is not None]
        return sorted(found, key=lambda v: v.name.lower())

    def get(self, ref: str | None) -> Voice | None:
        """Look a voice up by id or, case-insensitively, by its name."""
        if not ref:
            return None
        if _ID_RE.match(ref):
            return self._load(ref)
        ref_l = ref.strip().lower()
        return next((v for v in self.list() if v.name.lower() == ref_l), None)

    def reference_path(self, voice_id: str) -> Path:
        path = self.root / voice_id / "reference.wav"
        if not _ID_RE.match(voice_id) or not path.exists():
            raise VoiceError("That voice doesn't exist (anymore).")
        return path

    # -- writing --
    def add(self, name: str, audio_path: Path, *, source_name: str = "", consent: bool = False) -> Voice:
        if not consent:
            raise VoiceError("Please confirm that this is your own voice, or that you have permission to use it.")
        name = clean_name(name)
        if any(v.name.lower() == name.lower() for v in self.list()):
            raise VoiceError(f"You already have a voice called \"{name}\". Choose a different name.")
        clip, warnings = prepare_reference(_decode(Path(audio_path)))
        voice_id = f"{CLONE_PREFIX}{uuid.uuid4().hex[:10]}"
        folder = self.root / voice_id
        folder.mkdir(parents=True)
        try:
            _write_wav(folder / "reference.wav", clip)
            voice = Voice(voice_id, name, time.time(), round(clip.size / SAMPLE_RATE, 1), source_name[:120], warnings)
            tmp = folder / "voice.json.tmp"
            tmp.write_text(json.dumps(voice.to_dict()))
            os.replace(tmp, folder / "voice.json")
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        return voice

    def rename(self, voice_id: str, name: str) -> Voice:
        voice = self.get(voice_id)
        if voice is None:
            raise VoiceError("That voice doesn't exist (anymore).")
        name = clean_name(name)
        if any(v.name.lower() == name.lower() and v.id != voice.id for v in self.list()):
            raise VoiceError(f"You already have a voice called \"{name}\".")
        voice.name = name
        tmp = self.root / voice.id / "voice.json.tmp"
        tmp.write_text(json.dumps(voice.to_dict()))
        os.replace(tmp, self.root / voice.id / "voice.json")
        return voice

    def delete(self, voice_id: str) -> bool:
        if not _ID_RE.match(voice_id) or not (self.root / voice_id).exists():
            return False
        shutil.rmtree(self.root / voice_id)
        if get_default_voice(self.settings) == voice_id:
            set_default_voice(self.settings, None)
        return True


# ---- the user's default voice -----------------------------------------------------------------------
def _read_prefs(settings: Settings) -> dict[str, Any]:
    try:
        return json.loads(settings.prefs_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def get_default_voice(settings: Settings) -> str | None:
    voice = _read_prefs(settings).get("default_voice")
    if isinstance(voice, str) and is_clone_id(voice) and VoiceLibrary(settings).get(voice) is None:
        return None  # it was deleted
    return voice if isinstance(voice, str) and voice else None


def set_default_voice(settings: Settings, voice: str | None) -> None:
    prefs = _read_prefs(settings)
    if voice:
        prefs["default_voice"] = voice
    else:
        prefs.pop("default_voice", None)
    settings.prefs_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = settings.prefs_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(prefs))
    os.replace(tmp, settings.prefs_path)


def effective_voice(settings: Settings, requested: str | None) -> str | None:
    """The voice a job uses: what was asked for, else the saved default, else the engine's default.

    A name such as "My voice" is accepted anywhere an id is, and resolved to the stored voice id.
    """
    voice = requested or get_default_voice(settings)
    if voice and not re.fullmatch(r"[\w\-]{1,40}", voice):
        library_hit = VoiceLibrary(settings).get(voice)
        return library_hit.id if library_hit else voice
    if voice and not is_clone_id(voice):
        library_hit = VoiceLibrary(settings).get(voice)
        if library_hit:
            return library_hit.id
    return voice
