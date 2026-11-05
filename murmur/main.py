"""FastAPI app: upload a book PDF, watch it become an audiobook, listen or download."""
from __future__ import annotations

import asyncio
import io
import json
import re
import shutil
import uuid
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__
from .ai import ai_status
from .audio import ffmpeg_available, float_to_pcm, trim_silence
from .config import Settings, get_settings
from .jobs import JobManager
from .models import ACTIVE_STATUSES, TERMINAL_STATUSES, JobStatus
from .store import JobStore
from .tts import (
    EngineUnavailable,
    available_engines,
    cached_engine,
    clone_engines_status,
    resolve_engine_name,
    voice_catalog,
)
from .validation import describe_non_pdf, sniff_pdf
from .voices import VoiceError, VoiceLibrary, effective_voice, is_clone_id, set_default_voice

STATIC_DIR = Path(__file__).parent / "static"
JOB_ID_RE = re.compile(r"^[a-f0-9]{12}$")
FORMATS = {"m4b", "mp3", "zip"}
MEDIA_TYPES = {"m4b": "audio/mp4", "mp3": "audio/mpeg", "zip": "application/zip"}
PAUSE_STYLES = {"snappy", "natural", "relaxed"}
MAX_VOICE_UPLOAD_MB = 200
PUBLIC_FIELDS = (
    "id", "filename", "size_bytes", "status", "stage", "progress", "message", "error", "options",
    "report", "direction", "chapters", "title", "author", "output_format", "duration_sec", "eta_sec",
    "created_at", "updated_at",
)
PREVIEW_TEXT = "Hello, I'm your narrator. This is how your audiobook will sound."


class PreviewRequest(BaseModel):
    voice: str | None = None
    speed: float = 1.0
    engine: str | None = None


class DefaultVoiceRequest(BaseModel):
    voice: str | None = None


class RenameVoiceRequest(BaseModel):
    name: str


def public_job(job: dict[str, Any]) -> dict[str, Any]:
    out = {k: job.get(k) for k in PUBLIC_FIELDS}
    path = job.get("output_path")
    out["has_output"] = bool(path and Path(path).exists())
    return out


def error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    settings.ensure_dirs()
    store = JobStore(settings.db_path)
    manager = JobManager(store, settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        manager.recover()
        yield
        await run_in_threadpool(manager.shutdown)

    app = FastAPI(title="Murmur", version=__version__, lifespan=lifespan)
    app.state.settings, app.state.store, app.state.manager = settings, store, manager

    def get_job(job_id: str) -> dict[str, Any]:
        job = store.get(job_id) if JOB_ID_RE.match(job_id) else None
        if job is None:
            raise error(404, "not_found", "That audiobook doesn't exist.")
        return job

    # ---- system -------------------------------------------------------------------------------
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        try:
            active = resolve_engine_name(settings)
        except EngineUnavailable:
            active = None
        return {
            "active_engine": active,
            "status": "ok", "version": __version__, "ffmpeg": ffmpeg_available(),
            "engines": available_engines(settings), "max_upload_mb": settings.max_upload_mb,
            "ai": ai_status(settings).to_dict(), "clone_engines": clone_engines_status(),
        }

    @app.get("/api/ai")
    def ai() -> dict[str, Any]:
        return ai_status(settings).to_dict()

    # ---- voices -------------------------------------------------------------------------------
    @app.get("/api/voices")
    def voices(engine: str | None = None) -> dict[str, Any]:
        return voice_catalog(settings, engine)

    @app.post("/api/voices", status_code=201)
    async def add_voice(
        name: str = Form(...),
        consent: bool = Form(False),
        file: UploadFile = File(...),
    ) -> dict[str, Any]:
        """Upload a recording (any audio or video file), name it, and keep it as a narrator voice."""
        tmp = settings.uploads_dir / f"voice-{uuid.uuid4().hex[:12]}{Path(file.filename or 'x').suffix[:8]}"
        limit, size = MAX_VOICE_UPLOAD_MB * 1024 * 1024, 0
        try:
            with open(tmp, "wb") as out:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > limit:
                        raise error(413, "too_large", f"That recording is larger than {MAX_VOICE_UPLOAD_MB} MB.")
                    await run_in_threadpool(out.write, chunk)
            if size == 0:
                raise error(400, "empty", "The file is empty.")
            voice = await run_in_threadpool(
                lambda: VoiceLibrary(settings).add(name, tmp, source_name=Path(file.filename or "").name,
                                                   consent=consent))
        except VoiceError as exc:
            raise error(400, "bad_voice", str(exc)) from exc
        finally:
            tmp.unlink(missing_ok=True)
        return voice.to_dict()

    @app.put("/api/voices/default")
    def default_voice(req: DefaultVoiceRequest) -> dict[str, Any]:
        if req.voice:
            if not re.fullmatch(r"[\w\-]{1,40}", req.voice):
                raise error(400, "bad_voice", "Unknown voice.")
            if is_clone_id(req.voice) and VoiceLibrary(settings).get(req.voice) is None:
                raise error(404, "not_found", "That voice doesn't exist.")
        set_default_voice(settings, req.voice)
        return {"default": req.voice}

    @app.patch("/api/voices/{voice_id}")
    def rename_voice(voice_id: str, req: RenameVoiceRequest) -> dict[str, Any]:
        try:
            return VoiceLibrary(settings).rename(voice_id, req.name).to_dict()
        except VoiceError as exc:
            raise error(400, "bad_voice", str(exc)) from exc

    @app.delete("/api/voices/{voice_id}", status_code=204)
    def delete_voice(voice_id: str) -> Response:
        if not VoiceLibrary(settings).delete(voice_id):
            raise error(404, "not_found", "That voice doesn't exist.")
        return Response(status_code=204)

    @app.get("/api/voices/{voice_id}/sample")
    def voice_sample(voice_id: str) -> FileResponse:
        """The cleaned reference clip, so you can hear what the copy is being made from."""
        try:
            return FileResponse(VoiceLibrary(settings).reference_path(voice_id), media_type="audio/wav")
        except VoiceError as exc:
            raise error(404, "not_found", str(exc)) from exc

    @app.post("/api/preview")
    async def preview(req: PreviewRequest) -> Response:
        speed = min(max(req.speed, 0.6), 1.5)
        voice = effective_voice(settings, req.voice) or settings.voice
        if not re.fullmatch(r"[\w\-]{1,40}", voice):
            raise error(400, "bad_voice", "Unknown voice.")

        def render() -> bytes:
            engine = cached_engine(settings, engine=req.engine, voice=voice, speed=speed)
            pcm = float_to_pcm(trim_silence(engine.synthesize(PREVIEW_TEXT), int(engine.sample_rate)))
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(int(engine.sample_rate))
                w.writeframes(pcm)
            return buf.getvalue()

        try:
            return Response(await run_in_threadpool(render), media_type="audio/wav")
        except EngineUnavailable as exc:
            raise error(503, "engine_unavailable", str(exc)) from exc

    # ---- jobs ---------------------------------------------------------------------------------
    @app.post("/api/jobs", status_code=201)
    async def create_job(
        file: UploadFile = File(...),
        voice: str | None = Form(None),
        speed: float = Form(1.0),
        format: str = Form("m4b"),
        engine: str | None = Form(None),
        ai: str = Form("auto"),
        start_at_story: bool = Form(True),
        pause_style: str | None = Form(None),
    ) -> dict[str, Any]:
        voice = voice or None
        if format not in FORMATS:
            raise error(400, "bad_format", f"Format must be one of: {', '.join(sorted(FORMATS))}.")
        if voice is not None and not re.fullmatch(r"[\w\-]{1,40}", voice):
            raise error(400, "bad_voice", "Unknown voice.")
        if voice is not None and is_clone_id(voice) and VoiceLibrary(settings).get(voice) is None:
            raise error(404, "voice_not_found", "That voice doesn't exist. Upload it again or choose another.")
        if ai not in ("auto", "off"):
            raise error(400, "bad_ai", "ai must be 'auto' or 'off'.")
        if pause_style is not None and pause_style not in PAUSE_STYLES:
            raise error(400, "bad_pause_style", f"pause_style must be one of: {', '.join(sorted(PAUSE_STYLES))}.")
        job_id = uuid.uuid4().hex[:12]
        dest = settings.uploads_dir / f"{job_id}.pdf"
        limit = settings.max_upload_mb * 1024 * 1024
        size = 0
        try:
            with open(dest, "wb") as out:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > limit:
                        raise error(413, "too_large", f"That file is larger than the {settings.max_upload_mb} MB limit.")
                    await run_in_threadpool(out.write, chunk)
        except HTTPException:
            dest.unlink(missing_ok=True)
            raise
        # Fast, synchronous check so the user hears "not a PDF" immediately.
        if size == 0 or not sniff_pdf(dest):
            message = "The file is empty." if size == 0 else describe_non_pdf(dest)
            dest.unlink(missing_ok=True)
            raise error(415, "not_pdf", message)

        options = {
            "voice": voice or None, "speed": min(max(speed, 0.6), 1.5), "format": format,
            "engine": engine, "force": False, "ai": ai, "start_at_story": start_at_story,
            "pause_style": pause_style or settings.pause_style,
        }
        job = store.create(job_id=job_id, filename=Path(file.filename or "book.pdf").name,
                           size_bytes=size, input_path=dest, options=options)
        manager.submit(job_id)
        return public_job(job)

    @app.get("/api/jobs")
    def list_jobs() -> list[dict[str, Any]]:
        return [public_job(j) for j in store.list()]

    @app.get("/api/jobs/{job_id}")
    def read_job(job_id: str) -> dict[str, Any]:
        return public_job(get_job(job_id))

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str, request: Request) -> StreamingResponse:
        get_job(job_id)

        async def stream():
            last = None
            while not await request.is_disconnected():
                job = store.get(job_id)
                if job is None:
                    yield "event: gone\ndata: {}\n\n"
                    return
                payload = json.dumps(public_job(job))
                if payload != last:
                    yield f"data: {payload}\n\n"
                    last = payload
                if job["status"] in TERMINAL_STATUSES and not manager.is_active(job_id):
                    return
                await asyncio.sleep(0.7)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/jobs/{job_id}/confirm")
    def confirm_job(job_id: str) -> dict[str, Any]:
        job = get_job(job_id)
        if job["status"] != JobStatus.NEEDS_CONFIRMATION.value:
            raise error(409, "not_waiting", "This audiobook isn't waiting for confirmation.")
        options = {**(job.get("options") or {}), "force": True}
        store.update(job_id, options=options)
        manager.submit(job_id)
        return public_job(store.get(job_id))  # type: ignore[arg-type]

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict[str, Any]:
        job = get_job(job_id)
        if job["status"] in ACTIVE_STATUSES:
            if not manager.cancel(job_id):
                store.update(job_id, status=JobStatus.CANCELLED.value, message="Cancelled.")
        elif job["status"] == JobStatus.NEEDS_CONFIRMATION.value:
            store.update(job_id, status=JobStatus.CANCELLED.value, message="Discarded.")
        return public_job(store.get(job_id))  # type: ignore[arg-type]

    @app.delete("/api/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str) -> Response:
        job = get_job(job_id)
        if manager.is_active(job_id):
            raise error(409, "active", "Cancel this audiobook before deleting it.")
        shutil.rmtree(settings.jobs_dir / job_id, ignore_errors=True)
        Path(job["input_path"]).unlink(missing_ok=True)
        store.delete(job_id)
        return Response(status_code=204)

    @app.get("/api/jobs/{job_id}/download")
    def download(job_id: str, attachment: bool = False) -> FileResponse:
        job = get_job(job_id)
        path = Path(job["output_path"]) if job.get("output_path") else None
        if job["status"] != JobStatus.DONE.value or path is None or not path.exists():
            raise error(404, "not_ready", "This audiobook isn't ready yet.")
        return FileResponse(
            path, media_type=MEDIA_TYPES.get(job.get("output_format") or "m4b", "application/octet-stream"),
            filename=path.name, content_disposition_type="attachment" if attachment else "inline",
        )

    # ---- UI -----------------------------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
