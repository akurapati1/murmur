"""Silent-ish tone engine for tests and UI development. Never selected automatically."""
from __future__ import annotations

import numpy as np

from .base import TTSEngine


class DummyEngine(TTSEngine):
    name = "dummy"
    sample_rate = 16000
    CHARS_PER_SECOND = 120.0  # keeps test audio short so encoding stays quick

    def __init__(self, speed: float = 1.0):
        self.speed = speed

    def synthesize(self, text: str, *, speed: float = 1.0, intensity: float = 0.5) -> np.ndarray:
        seconds = max(len(text) / self.CHARS_PER_SECOND / max(self.speed * speed, 0.1), 0.2)
        n = int(seconds * self.sample_rate)
        t = np.arange(n, dtype=np.float32) / self.sample_rate
        tone = 0.04 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
        fade = min(int(0.02 * self.sample_rate), n // 2)
        if fade:
            ramp = np.linspace(0, 1, fade, dtype=np.float32)
            tone[:fade] *= ramp
            tone[-fade:] *= ramp[::-1]
        return tone
