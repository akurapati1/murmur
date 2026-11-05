"""Runtime configuration, read from MURMUR_* environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(f"MURMUR_{name}", default)


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    models_dir: Path
    max_upload_mb: int = 500
    workers: int = 1
    tts_engine: str = "auto"  # auto | kokoro | piper | espeak | dummy
    voice: str = "af_heart"
    speed: float = 1.0
    chunk_chars: int = 350
    sample_pages: int = 40
    bitrate: str = "64k"
    keep_work: bool = False
    skip_back_matter: bool = True
    skip_captions: bool = True
    kokoro_model: str = "kokoro-v1.0.onnx"
    kokoro_voices: str = "voices-v1.0.bin"
    piper_voice: str = "en_US-lessac-medium"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "murmur.db"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.models_dir, self.uploads_dir, self.jobs_dir):
            p.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    home = Path(_env("HOME", str(Path.home() / ".murmur")))
    return Settings(
        data_dir=Path(_env("DATA_DIR", str(home / "data"))),
        models_dir=Path(_env("MODELS_DIR", str(home / "models"))),
        max_upload_mb=int(_env("MAX_UPLOAD_MB", "500")),
        workers=max(1, int(_env("WORKERS", "1"))),
        tts_engine=_env("TTS_ENGINE", "auto").lower(),
        voice=_env("VOICE", "af_heart"),
        speed=float(_env("SPEED", "1.0")),
        chunk_chars=int(_env("CHUNK_CHARS", "350")),
        sample_pages=int(_env("SAMPLE_PAGES", "40")),
        bitrate=_env("BITRATE", "64k"),
        keep_work=_env_bool("KEEP_WORK", False),
        skip_back_matter=_env_bool("SKIP_BACK_MATTER", True),
        skip_captions=_env_bool("SKIP_CAPTIONS", True),
        kokoro_model=_env("KOKORO_MODEL", "kokoro-v1.0.onnx"),
        kokoro_voices=_env("KOKORO_VOICES", "voices-v1.0.bin"),
        piper_voice=_env("PIPER_VOICE", "en_US-lessac-medium"),
    )
