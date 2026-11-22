"""Zero-shot voice cloning: narrate in the voice of a short reference recording.

Two open-source engines are supported, both optional and loaded lazily (they pull in PyTorch):

* **Chatterbox** (MIT licence, Resemble AI): the default. Expressive, with an adjustable "exaggeration"
  that the director's per-sentence intensity maps onto.  ``pip install -e ".[clone]"``
* **Coqui XTTS-v2** (Coqui Public Model Licence: *non-commercial use only*).
  ``pip install -e ".[xtts]"`` and set ``COQUI_TOS_AGREED=1`` after reading the licence.

Cloning is far heavier than the built-in voices. A GPU (CUDA or Apple Silicon) is strongly
recommended; on a plain CPU expect several times slower than real time.
"""
from __future__ import annotations

import importlib.util
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

from .base import EngineUnavailable, TTSEngine

CHATTERBOX_HINT = 'From the Murmur folder, run: pip install -e ".[clone]"  (a GPU is strongly recommended)'
XTTS_HINT = (
    'From the Murmur folder, run: pip install -e ".[xtts]" and set COQUI_TOS_AGREED=1 (non-commercial licence)'
)

_load_lock = threading.Lock()


def clone_engines_status() -> list[dict]:
    """Which cloning engines are installed (without importing PyTorch)."""
    return [
        {"name": "chatterbox", "available": importlib.util.find_spec("chatterbox") is not None,
         "hint": CHATTERBOX_HINT, "license": "MIT"},
        {"name": "xtts", "available": importlib.util.find_spec("TTS") is not None,
         "hint": XTTS_HINT, "license": "Non-commercial (CPML)"},
    ]


def pick_clone_engine(requested: str = "auto") -> str:
    status = {e["name"]: e["available"] for e in clone_engines_status()}
    requested = (requested or "auto").lower()
    if requested != "auto":
        if requested not in status:
            raise EngineUnavailable(f"Unknown voice-cloning engine '{requested}'. Use chatterbox or xtts.")
        if not status[requested]:
            raise EngineUnavailable(f"The {requested} voice-cloning engine isn't installed. "
                                    f"{CHATTERBOX_HINT if requested == 'chatterbox' else XTTS_HINT}")
        return requested
    for name in ("chatterbox", "xtts"):
        if status[name]:
            return name
    raise EngineUnavailable(
        "Narrating in your own voice needs a voice-cloning engine, and none is installed yet. "
        'The recommended one is free and open source. From the Murmur folder, run: pip install -e ".[clone]". '
        "A GPU makes it dramatically faster."
    )


def _device() -> str:
    try:
        import torch
    except ImportError:  # pragma: no cover
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def _chatterbox_model():  # pragma: no cover - needs the real model
    from chatterbox.tts import ChatterboxTTS

    return ChatterboxTTS.from_pretrained(device=_device())


@lru_cache(maxsize=1)
def _xtts_model():  # pragma: no cover - needs the real model
    from TTS.api import TTS

    return TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(_device())


def _to_numpy(wav) -> np.ndarray:  # pragma: no cover - needs torch
    if hasattr(wav, "detach"):
        wav = wav.detach().cpu().numpy()
    return np.asarray(wav, dtype=np.float32).reshape(-1)


class CloneEngine(TTSEngine):
    """Narrates with a cloned voice. One loaded model is shared by every voice."""

    def __init__(self, reference: Path, engine: str = "auto", speed: float = 1.0):
        self.kind = pick_clone_engine(engine)
        self.name = self.kind
        self.reference = Path(reference)
        if not self.reference.exists():
            raise EngineUnavailable("The reference recording for that voice is missing. Upload the voice again.")
        self.speed = speed
        with _load_lock:
            try:  # pragma: no cover - needs the real model
                self._model = _chatterbox_model() if self.kind == "chatterbox" else _xtts_model()
            except Exception as exc:  # pragma: no cover
                raise EngineUnavailable(f"Couldn't load the {self.kind} voice model: {exc}") from exc
        self.sample_rate = int(getattr(self._model, "sr", 0) or 24000)  # pragma: no cover

    def synthesize(self, text: str, *, speed: float = 1.0, intensity: float = 0.5) -> np.ndarray:  # pragma: no cover
        rel = self.speed * speed
        if self.kind == "chatterbox":
            # exaggeration: 0.25 flat ... 1.0 very dramatic. Lower cfg_weight slows the delivery down.
            wav = self._model.generate(
                text, audio_prompt_path=str(self.reference),
                exaggeration=float(np.clip(0.3 + 0.8 * intensity, 0.25, 1.0)),
                cfg_weight=float(np.clip(0.5 * rel, 0.2, 0.8)),
            )
        else:
            wav = self._model.tts(text=text, speaker_wav=str(self.reference), language="en",
                                  speed=float(np.clip(rel, 0.6, 1.6)))
        return _to_numpy(wav)
