"""Command line: murmur serve | convert | analyze | doctor | models."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from pathlib import Path

from . import __version__
from .audio import ffmpeg_available
from .classifier import analyze_pdf, ocr_available
from .config import get_settings
from .pipeline import process_job
from .store import JobStore
from .tts import available_engines
from .validation import validate_pdf


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("murmur.main:create_app", factory=True, host=args.host, port=args.port, log_level="info")
    return 0


def _analyze(args: argparse.Namespace) -> int:
    settings = get_settings()
    path = Path(args.file)
    check = validate_pdf(path)
    if not check.ok:
        print(f"Not usable: {check.reason}")
        return 2
    report = analyze_pdf(path, settings.sample_pages)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"{check.page_count} pages, about {report.estimated_words:,} words "
              f"(~{report.estimated_hours} hours of audio)")
        print(f"Verdict: {report.verdict.value}")
        for reason in report.reasons:
            print(f"  - {reason}")
    return 0 if report.convertible else 1


def _convert(args: argparse.Namespace) -> int:
    settings = get_settings()
    settings.ensure_dirs()
    src = Path(args.file)
    if not src.exists():
        print(f"No such file: {src}")
        return 2
    store = JobStore(settings.db_path)
    job_id = uuid.uuid4().hex[:12]
    dest = settings.uploads_dir / f"{job_id}.pdf"
    shutil.copyfile(src, dest)
    options = {"voice": args.voice, "speed": args.speed, "format": args.format, "engine": args.engine,
               "force": args.force}
    store.create(job_id=job_id, filename=src.name, size_bytes=dest.stat().st_size, input_path=dest, options=options)

    def show(fields: dict) -> None:
        pct = int(fields["progress"] * 100)
        eta = fields.get("eta_sec")
        tail = (f"  ~{int(eta // 60)} min left" if eta >= 90 else f"  ~{int(eta)} s left") if eta else ""
        sys.stderr.write(f"\r[{pct:3d}%] {fields['message'][:70]:<70}{tail:<18}")
        sys.stderr.flush()

    process_job(job_id, store, settings, None, show)
    sys.stderr.write("\n")
    job = store.get(job_id)
    assert job is not None
    if job["status"] == "done":
        out = Path(args.out or ".") / Path(job["output_path"]).name
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(job["output_path"], out)
        print(f"Saved {out}")
        return 0
    if job["status"] == "needs_confirmation":
        print(f"{job['message']}\nRun again with --force to convert it anyway.")
        return 1
    print(job.get("error") or job["message"])
    return 1


def _doctor(_: argparse.Namespace) -> int:
    settings = get_settings()
    print(f"Murmur {__version__}")
    print(f"  data:   {settings.data_dir}\n  models: {settings.models_dir}")
    print(f"  ffmpeg: {'ok' if ffmpeg_available() else 'MISSING (needed to build audiobooks)'}")
    print(f"  OCR:    {'ok (Tesseract found)' if ocr_available() else 'not installed (only needed for scanned PDFs)'}")
    print("Voice engines:")
    ready = False
    for row in available_engines(settings):
        ready |= row["available"]
        print(f"  {row['name']:<8} {'ready' if row['available'] else row['hint']}")
    return 0 if ready and ffmpeg_available() else 1


def _models(args: argparse.Namespace) -> int:
    from .models_download import download_kokoro, download_piper

    settings = get_settings()
    if args.engine in ("kokoro", "all"):
        download_kokoro(settings, small=args.small)
    if args.engine in ("piper", "all"):
        download_piper(settings, args.voice)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="murmur", description="Turn any book PDF into an audiobook.")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("serve", help="start the web app")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(fn=_serve)

    p = sub.add_parser("convert", help="convert a PDF without the web app")
    p.add_argument("file")
    p.add_argument("--out", help="folder for the finished audiobook (default: current folder)")
    p.add_argument("--voice")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--format", choices=["m4b", "mp3", "zip"], default="m4b")
    p.add_argument("--engine", choices=["kokoro", "piper", "espeak"])
    p.add_argument("--force", action="store_true", help="convert even if it looks like a technical book")
    p.set_defaults(fn=_convert)

    p = sub.add_parser("analyze", help="check whether a PDF will work as an audiobook")
    p.add_argument("file")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=_analyze)

    p = sub.add_parser("doctor", help="check that everything is installed")
    p.set_defaults(fn=_doctor)

    p = sub.add_parser("models", help="download voice models")
    p.add_argument("engine", choices=["kokoro", "piper", "all"])
    p.add_argument("--small", action="store_true", help="Kokoro: smaller, faster model (~90 MB)")
    p.add_argument("--voice", help="Piper: voice name, e.g. en_US-lessac-medium")
    p.set_defaults(fn=_models)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
