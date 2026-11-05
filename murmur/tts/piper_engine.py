"""Piper (ONNX): very fast on any CPU. Run as a separate process through its CLI, which keeps
Murmur independent of Piper's license and works across Piper versions."""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

from .base import EngineUnavailable, TTSEngine

INSTALL_HINT = "Run: pip install 'murmur[piper]' && murmur models piper"


def piper_command() -> list[str] | None:
    exe = shutil.which("piper")
    if exe:
        return [exe]
    if importlib.util.find_spec("piper") is not None:
        return [sys.executable, "-m", "piper"]
    return None


class PiperEngine(TTSEngine):
    name = "piper"

    def __init__(self, model_path: Path, speed: float = 1.0):
        cmd = piper_command()
        if cmd is None:
            raise EngineUnavailable(f"Piper isn't installed. {INSTALL_HINT}")
        model_path = Path(model_path)
        if not model_path.exists():
            raise EngineUnavailable(f"Piper voice {model_path.name} was not found. {INSTALL_HINT}")
        self._cmd = cmd
        self._model = model_path
        cfg = Path(f"{model_path}.json")
        self.sample_rate = 22050
        if cfg.exists():
            self.sample_rate = int(json.loads(cfg.read_text()).get("audio", {}).get("sample_rate", 22050))
        help_text = subprocess.run([*cmd, "--help"], capture_output=True, text=True).stdout
        flag = "--length-scale" if "--length-scale" in help_text else "--length_scale"
        self._length_flag = flag
        self._base_speed = speed

    def synthesize(self, text: str, *, speed: float = 1.0, intensity: float = 0.5) -> np.ndarray:
        extra = [self._length_flag, f"{1.0 / max(self._base_speed * speed, 0.1):.3f}"]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.wav"
            subprocess.run(
                [*self._cmd, "--model", str(self._model), "--output-file", str(out), *extra],
                input=text.encode("utf-8"), capture_output=True, check=True,
            )
            with wave.open(str(out), "rb") as w:
                self.sample_rate = w.getframerate()
                pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        return pcm.astype(np.float32) / 32768.0
