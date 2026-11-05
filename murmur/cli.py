"""Command line: murmur serve | convert | analyze | doctor | models."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from pathlib import Path

from . import __version__
from .ai import AIError, ai_status, resolve_provider
from .audio import ffmpeg_available
from .classifier import analyze_pdf, ocr_available
from .config import get_settings
from .director import DirectorOptions, direct_book
from .extract import extract_book
from .models import Verdict
from .pipeline import process_job
from .prosody import describe_pauses
from .store import JobStore
from .tts import available_engines, clone_engines_status
from .validation import validate_pdf
from .voices import VoiceError, VoiceLibrary, get_default_voice, set_default_voice


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
               "force": args.force, "ai": args.ai, "start_at_story": not args.keep_opening,
               "pause_style": args.pause_style or settings.pause_style}
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


def _direct(args: argparse.Namespace) -> int:
    """Show what Murmur would cut and how it would perform the book, without making any audio."""
    settings = get_settings()
    path = Path(args.file)
    check = validate_pdf(path)
    if not check.ok:
        print(f"Not usable: {check.reason}")
        return 2
    report = analyze_pdf(path, settings.sample_pages)
    print("Reading the book...", file=sys.stderr)
    book = extract_book(path, ocr=report.verdict == Verdict.SCANNED, skip_back_matter=settings.skip_back_matter,
                        skip_captions=settings.skip_captions, title_hint=path.stem)
    provider = None
    if args.ai != "off":
        try:
            provider = resolve_provider(settings)
        except AIError as exc:
            print(f"AI unavailable: {exc}", file=sys.stderr)

    def show(frac: float, msg: str) -> None:
        sys.stderr.write(f"\r[{int(frac * 100):3d}%] {msg[:60]:<60}")
        sys.stderr.flush()

    script = direct_book(book, settings, DirectorOptions(use_ai=args.ai != "off", start_at_story=not args.keep_opening,
                                                         pause_style=args.pause_style or settings.pause_style),
                         provider=provider, progress=show)
    sys.stderr.write("\n")
    if args.script:
        Path(args.script).write_text(json.dumps(script.to_dict(), indent=2))
        print(f"Wrote the full narration script to {args.script}")
    r = script.report
    if args.json:
        print(json.dumps(r, indent=2))
        return 0
    who = f"{r['provider']} ({r['model']})" if r["model"] else "built-in rules"
    print(f"Director: {who}; story located by {r['story_found_by']}; {'fiction' if r['fiction'] else 'non-fiction'}")
    print(f"Narration starts at: {r['story_starts_with']['chapter']}")
    print(f"  \"{r['story_starts_with']['text']}\"")
    print(f"Cut {r['removed_words']:,} words, narrating {r['narrated_words']:,} words in {r['chapters']} chapters:")
    for item in r["removed"]:
        print(f"  - {item['where']:<5} {item['title'][:50]:<50} {item['words']:>6} words")
    if r["pronunciations"]:
        print("Pronunciations: " + ", ".join(f"{k} -> {v}" for k, v in r["pronunciations"].items()))
    for note in r["notes"]:
        print(f"Note: {note}")
    print("First passages [pause after, relative speed, energy]:")
    for line in describe_pauses(script.chapters[0].chunks, 12) if script.chapters else []:
        print("  " + line)
    return 0


def _voices(args: argparse.Namespace) -> int:
    settings = get_settings()
    settings.ensure_dirs()
    library = VoiceLibrary(settings)
    try:
        if args.action == "add":
            consent = args.yes
            if not consent and sys.stdin.isatty():
                consent = input("Is this your own voice, or do you have permission to use it? [y/N] ").lower() \
                    .startswith("y")
            voice = library.add(args.name, Path(args.file), source_name=Path(args.file).name, consent=consent)
            print(f"Saved voice \"{voice.name}\" ({voice.duration_sec:.0f} s of speech) as {voice.id}")
            for w in voice.warnings:
                print(f"  Note: {w}")
            if args.default:
                set_default_voice(settings, voice.id)
                print("It is now your default voice.")
        elif args.action == "list":
            default = get_default_voice(settings)
            rows = library.list()
            if not rows:
                print("No voices yet. Add one with: murmur voices add \"My voice\" recording.mp3")
            for v in rows:
                print(f"{'*' if v.id == default else ' '} {v.name:<24} {v.duration_sec:>5.0f} s   {v.id}")
        elif args.action == "rm":
            voice = library.get(args.name)
            if voice is None or not library.delete(voice.id):
                print(f"No voice called \"{args.name}\".")
                return 1
            print(f"Deleted \"{voice.name}\".")
        elif args.action == "default":
            if args.clear:
                set_default_voice(settings, None)
                print("Default voice cleared.")
            else:
                voice = library.get(args.name)
                set_default_voice(settings, voice.id if voice else args.name)
                print(f"Default voice: {voice.name if voice else args.name}")
    except VoiceError as exc:
        print(exc)
        return 1
    return 0


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
    ai = ai_status(settings)
    print("Narration director (AI):")
    print(f"  {ai.provider:<17} " + (f"ready ({ai.model})" if ai.available else (ai.hint or "off")))
    print("Voice cloning (for narrating in your own voice):")
    for row in clone_engines_status():
        print(f"  {row['name']:<10} {'ready' if row['available'] else row['hint']}  [{row['license']}]")
    return 0 if ready and ffmpeg_available() else 1


def _models(args: argparse.Namespace) -> int:
    from .models_download import download_clone, download_kokoro, download_piper

    settings = get_settings()
    if args.engine == "clone":
        download_clone(settings)
        return 0
    if args.engine in ("kokoro", "all"):
        download_kokoro(settings, small=args.small)
    if args.engine in ("piper", "all"):
        download_piper(settings, args.voice)
    return 0


def _direction_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ai", choices=["auto", "off"], default="auto",
                   help="let an AI guide where the story starts and how it is read (default: use one if available)")
    p.add_argument("--pause-style", choices=["snappy", "natural", "relaxed"],
                   help="how long the narrator pauses (default: natural)")
    p.add_argument("--keep-opening", action="store_true",
                   help="read everything, including copyright pages, dedications and acknowledgments")


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
    p.add_argument("--voice", help="a built-in voice, or the name of one you saved with 'murmur voices add'")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--format", choices=["m4b", "mp3", "zip"], default="m4b")
    p.add_argument("--engine", choices=["kokoro", "piper", "espeak"])
    p.add_argument("--force", action="store_true", help="convert even if it looks like a technical book")
    _direction_flags(p)
    p.set_defaults(fn=_convert)

    p = sub.add_parser("direct", help="preview what will be cut and how it will be performed (no audio)")
    p.add_argument("file")
    p.add_argument("--json", action="store_true", help="print the report as JSON")
    p.add_argument("--script", help="write the complete narration script (every sentence and pause) to this file")
    _direction_flags(p)
    p.set_defaults(fn=_direct)

    p = sub.add_parser("voices", help="manage your own narrator voices")
    vsub = p.add_subparsers(dest="action", required=True)
    v = vsub.add_parser("add", help="save a recording as a named voice")
    v.add_argument("name")
    v.add_argument("file", help="any audio or video file with 10-30 seconds of clear speech")
    v.add_argument("--yes", action="store_true", help="confirm this is your voice or you have permission to use it")
    v.add_argument("--default", action="store_true", help="use it for every audiobook from now on")
    vsub.add_parser("list", help="list saved voices")
    v = vsub.add_parser("rm", help="delete a saved voice")
    v.add_argument("name")
    v = vsub.add_parser("default", help="choose the voice used when none is specified")
    v.add_argument("name", nargs="?")
    v.add_argument("--clear", action="store_true")
    p.set_defaults(fn=_voices)

    p = sub.add_parser("analyze", help="check whether a PDF will work as an audiobook")
    p.add_argument("file")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=_analyze)

    p = sub.add_parser("doctor", help="check that everything is installed")
    p.set_defaults(fn=_doctor)

    p = sub.add_parser("models", help="download voice models")
    p.add_argument("engine", choices=["kokoro", "piper", "clone", "all"])
    p.add_argument("--small", action="store_true", help="Kokoro: smaller, faster model (~90 MB)")
    p.add_argument("--voice", help="Piper: voice name, e.g. en_US-lessac-medium")
    p.set_defaults(fn=_models)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
