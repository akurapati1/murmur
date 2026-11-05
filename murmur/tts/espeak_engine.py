"""eSpeak NG: robotic, but installs anywhere with one command. The zero-setup fallback."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

from .base import EngineUnavailable, TTSEngine


def espeak_binary() -> str | None:
    return shutil.which("espeak-ng") or shutil.which("espeak")


class EspeakEngine(TTSEngine):
    name = "espeak"

    def __init__(self, voice: str = "en-us", speed: float = 1.0):
        exe = espeak_binary()
        if exe is None:
            raise EngineUnavailable(
                "eSpeak NG isn't installed. Install it with 'sudo apt install espeak-ng' or 'brew install espeak-ng'."
            )
        self._exe = exe
        self._voice = voice if voice.startswith("en") else "en-us"
        self._wpm = int(165 * speed)
        self.sample_rate = 22050

    def synthesize(self, text: str) -> np.ndarray:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.wav"
            subprocess.run(
                [self._exe, "-v", self._voice, "-s", str(self._wpm), "--stdin", "-w", str(out)],
                input=text.encode("utf-8"), capture_output=True, check=True,
            )
            with wave.open(str(out), "rb") as w:
                self.sample_rate = w.getframerate()
                pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        return pcm.astype(np.float32) / 32768.0
