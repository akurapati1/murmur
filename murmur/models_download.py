"""Download the open-source voice models (one time, a few hundred MB at most)."""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

from .config import Settings

KOKORO_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
KOKORO_FULL = "kokoro-v1.0.onnx"  # ~310 MB, best quality
KOKORO_SMALL = "kokoro-v1.0.int8.onnx"  # ~90 MB, slightly lower quality, faster on weak laptops
KOKORO_VOICES = "voices-v1.0.bin"
PIPER_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


def _download(url: str, dest: Path, label: str) -> None:
    if dest.exists():
        print(f"  {label}: already downloaded")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "murmur-model-downloader"})
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        got, last_pct = 0, -1
        while block := resp.read(1024 * 256):
            out.write(block)
            got += len(block)
            pct = int(got * 100 / total) if total else -1
            if total and pct != last_pct:
                last_pct = pct
                sys.stdout.write(f"\r  {label}: {pct:3d}% of {total / 1e6:.0f} MB")
                sys.stdout.flush()
    tmp.replace(dest)
    print(f"\r  {label}: done{' ' * 30}")


def download_kokoro(settings: Settings, small: bool = False) -> None:
    base = settings.models_dir / "kokoro"
    print("Kokoro (Apache-2.0):")
    # The small model is stored under the configured filename so no extra setting is needed.
    _download(f"{KOKORO_BASE}/{KOKORO_SMALL if small else KOKORO_FULL}", base / settings.kokoro_model,
              "voice model" + (" (small)" if small else ""))
    _download(f"{KOKORO_BASE}/{KOKORO_VOICES}", base / settings.kokoro_voices, "voice pack")


def download_piper(settings: Settings, voice: str | None = None) -> None:
    voice = voice or settings.piper_voice
    try:
        region, name, quality = voice.split("-")
    except ValueError:
        raise SystemExit(f"Piper voice names look like en_US-lessac-medium, not '{voice}'.")
    lang = region.split("_")[0]
    url = f"{PIPER_BASE}/{lang}/{region}/{name}/{quality}/{voice}.onnx"
    base = settings.models_dir / "piper"
    print("Piper:")
    _download(url, base / f"{voice}.onnx", voice)
    _download(url + ".json", base / f"{voice}.onnx.json", f"{voice} config")


def download_clone(settings: Settings) -> None:
    """Fetch the voice-cloning model weights (about 2 GB) by loading the engine once."""
    from .tts.base import EngineUnavailable
    from .tts.clone_engine import _chatterbox_model, _xtts_model, pick_clone_engine

    try:
        kind = pick_clone_engine(settings.clone_engine)
    except EngineUnavailable as exc:
        raise SystemExit(str(exc))
    print(f"Downloading the {kind} voice-cloning model (one time, about 2 GB)...")
    (_chatterbox_model if kind == "chatterbox" else _xtts_model)()
    print("Done.")
