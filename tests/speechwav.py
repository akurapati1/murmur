"""Make speech-like test recordings (syllable bursts with pauses and a faint noise floor)."""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def speechlike(seconds: float, sr: int = 22050, seed: int = 1, noise: float = 0.002, loud: float = 0.35) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = rng.normal(0, noise, int(seconds * sr)).astype(np.float32)
    t, pos = 0.0, 0
    while pos < out.size - sr // 4:
        dur = float(rng.uniform(0.12, 0.35))
        n = int(dur * sr)
        f0 = float(rng.uniform(110, 220))
        x = np.arange(n) / sr
        syl = sum(np.sin(2 * np.pi * f0 * k * x) / k for k in (1, 2, 3)) * np.hanning(n)
        out[pos:pos + n] += (loud * syl / 2).astype(np.float32)[: out.size - pos]
        pos += n + int(rng.uniform(0.02, 0.25) * sr)
        t += dur
    return np.clip(out, -1, 1)


def write_wav(path: Path, samples: np.ndarray, sr: int = 22050) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return path
