"""Engine registry: picks the best available open-source TTS engine."""
from __future__ import annotations

import importlib.util
from functools import lru_cache
from typing import Any

from ..config import Settings
from .base import EngineUnavailable, TTSEngine
from .dummy_engine import DummyEngine
from .espeak_engine import EspeakEngine, espeak_binary
from .kokoro_engine import KokoroEngine
from .piper_engine import PiperEngine, piper_command

__all__ = ["EngineUnavailable", "TTSEngine", "create_engine", "available_engines", "voice_catalog", "cached_engine"]

# (id, label, group)
KOKORO_VOICES = [
    ("af_heart", "Heart", "American, female"),
    ("af_bella", "Bella", "American, female"),
    ("af_nicole", "Nicole", "American, female"),
    ("af_sarah", "Sarah", "American, female"),
    ("af_sky", "Sky", "American, female"),
    ("af_nova", "Nova", "American, female"),
    ("am_adam", "Adam", "American, male"),
    ("am_michael", "Michael", "American, male"),
    ("am_eric", "Eric", "American, male"),
    ("am_liam", "Liam", "American, male"),
    ("bf_emma", "Emma", "British, female"),
    ("bf_isabella", "Isabella", "British, female"),
    ("bm_george", "George", "British, male"),
    ("bm_lewis", "Lewis", "British, male"),
]
PIPER_VOICES = [("default", "Default Piper voice", "English")]
ESPEAK_VOICES = [("en-us", "English (US)", "Robotic"), ("en-gb", "English (UK)", "Robotic")]
DUMMY_VOICES = [("tone", "Test tone", "Development")]

PREFERENCE = ["kokoro", "piper", "espeak"]


def _kokoro_status(settings: Settings) -> tuple[bool, str]:
    if importlib.util.find_spec("kokoro_onnx") is None:
        return False, "Install with: pip install 'murmur[kokoro]'"
    if not (settings.models_dir / "kokoro" / settings.kokoro_model).exists():
        return False, "Download the model with: murmur models kokoro"
    return True, ""


def _piper_status(settings: Settings) -> tuple[bool, str]:
    if piper_command() is None:
        return False, "Install with: pip install 'murmur[piper]'"
    if not (settings.models_dir / "piper" / f"{settings.piper_voice}.onnx").exists():
        return False, "Download a voice with: murmur models piper"
    return True, ""


def _espeak_status(_: Settings) -> tuple[bool, str]:
    if espeak_binary() is None:
        return False, "Install eSpeak NG (apt install espeak-ng / brew install espeak-ng)"
    return True, ""


_STATUS = {"kokoro": _kokoro_status, "piper": _piper_status, "espeak": _espeak_status}


def available_engines(settings: Settings) -> list[dict[str, Any]]:
    rows = []
    for name in PREFERENCE:
        ok, hint = _STATUS[name](settings)
        rows.append({"name": name, "available": ok, "hint": hint})
    return rows


def resolve_engine_name(settings: Settings, requested: str | None = None) -> str:
    name = (requested or settings.tts_engine or "auto").lower()
    if name == "dummy":
        return "dummy"
    if name != "auto":
        if name not in _STATUS:
            raise EngineUnavailable(f"Unknown TTS engine '{name}'.")
        return name
    for candidate in PREFERENCE:
        if _STATUS[candidate](settings)[0]:
            return candidate
    raise EngineUnavailable(
        "No text-to-speech engine is set up yet. The best free option is Kokoro: "
        "pip install 'murmur[kokoro]' && murmur models kokoro. "
        "For a quick start, install eSpeak NG instead."
    )


def voice_catalog(settings: Settings, engine: str | None = None) -> dict[str, Any]:
    try:
        name = resolve_engine_name(settings, engine)
    except EngineUnavailable:
        name = "kokoro"  # show the recommended voices even before setup
    voices = {
        "kokoro": KOKORO_VOICES, "piper": PIPER_VOICES, "espeak": ESPEAK_VOICES, "dummy": DUMMY_VOICES,
    }[name]
    default = settings.voice if any(v[0] == settings.voice for v in voices) else voices[0][0]
    return {
        "engine": name,
        "default": default,
        "voices": [{"id": i, "label": label, "group": group} for i, label, group in voices],
    }


def create_engine(settings: Settings, *, engine: str | None = None, voice: str | None = None,
                  speed: float | None = None) -> TTSEngine:
    name = resolve_engine_name(settings, engine)
    speed = speed or settings.speed
    voice = voice or settings.voice
    if name == "dummy":
        return DummyEngine(speed=speed)
    if name == "kokoro":
        base = settings.models_dir / "kokoro"
        return KokoroEngine(base / settings.kokoro_model, base / settings.kokoro_voices, voice=voice, speed=speed)
    if name == "piper":
        model = settings.models_dir / "piper" / f"{voice}.onnx"
        if not model.exists():
            model = settings.models_dir / "piper" / f"{settings.piper_voice}.onnx"
        return PiperEngine(model, speed=speed)
    return EspeakEngine(voice=voice, speed=speed)


@lru_cache(maxsize=4)
def _cached(settings: Settings, engine: str, voice: str, speed: float) -> TTSEngine:
    return create_engine(settings, engine=engine, voice=voice, speed=speed)


def cached_engine(settings: Settings, *, engine: str | None, voice: str, speed: float) -> TTSEngine:
    """Reuse loaded models for short requests such as voice previews."""
    return _cached(settings, resolve_engine_name(settings, engine), voice, round(speed, 2))
