"""Kokoro-82M through ONNX Runtime: natural-sounding, CPU-friendly, Apache-2.0."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import EngineUnavailable, TTSEngine

INSTALL_HINT = 'From the Murmur folder, run: pip install -e ".[kokoro]", then: murmur models kokoro'


def language_for(voice: str) -> str:
    return "en-gb" if voice[:1] == "b" else "en-us"


class KokoroEngine(TTSEngine):
    name = "kokoro"
    sample_rate = 24000

    def __init__(self, model_path: Path, voices_path: Path, voice: str = "af_heart", speed: float = 1.0):
        try:
            from kokoro_onnx import Kokoro
        except ImportError as exc:
            raise EngineUnavailable(f"The kokoro-onnx package isn't installed. {INSTALL_HINT}") from exc
        if not Path(model_path).exists() or not Path(voices_path).exists():
            raise EngineUnavailable(f"Kokoro model files were not found in {Path(model_path).parent}. {INSTALL_HINT}")
        self._engine = Kokoro(str(model_path), str(voices_path))
        self.voice = voice
        self.speed = speed
        self.lang = language_for(voice)

    def synthesize(self, text: str, *, speed: float = 1.0, intensity: float = 0.5) -> np.ndarray:
        samples, sr = self._engine.create(text, voice=self.voice, speed=self.speed * speed, lang=self.lang)
        self.sample_rate = int(sr)
        return np.asarray(samples, dtype=np.float32)
