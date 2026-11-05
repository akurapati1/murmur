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
from .audio import ffmpeg_available, float_to_pcm
from .config import Settings, get_settings
from .jobs import JobManager
from .models import ACTIVE_STATUSES, TERMINAL_STATUSES, JobStatus
from .store import JobStore
from .tts import EngineUnavailable, available_engines, cached_engine, resolve_engine_name, voice_catalog
from .validation import describe_non_pdf, sniff_pdf

STATIC_DIR = Path(__file__).parent / "static"
JOB_ID_RE = re.compile(r"^[a-f0-9]{12}$")
FORMATS = {"m4b", "mp3", "zip"}
MEDIA_TYPES = {"m4b": "audio/mp4", "mp3": "audio/mpeg", "zip": "application/zip"}
PUBLIC_FIELDS = (
    "id", "filename", "size_bytes", "status", "stage", "progress", "message", "error", "options",
    "report", "chapters", "title", "author", "output_format", "duration_sec", "eta_sec",
    "created_at", "updated_at",
)
PREVIEW_TEXT = "Hello, I'm your narrator. This is how your audiobook will sound."


class PreviewRequest(BaseModel):
    voice: str | None = None
    speed: float = 1.0
    engine: str | None = None


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
        }

    @app.get("/api/voices")
    def voices(engine: str | None = None) -> dict[str, Any]:
        return voice_catalog(settings, engine)

    @app.post("/api/preview")
    async def preview(req: PreviewRequest) -> Response:
        speed = min(max(req.speed, 0.6), 1.5)
        voice = req.voice or settings.voice
        if not re.fullmatch(r"[\w\-]{1,40}", voice):
            raise error(400, "bad_voice", "Unknown voice.")

        def render() -> bytes:
            engine = cached_engine(settings, engine=req.engine, voice=voice, speed=speed)
            pcm = float_to_pcm(engine.synthesize(PREVIEW_TEXT))
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
    ) -> dict[str, Any]:
        if format not in FORMATS:
            raise error(400, "bad_format", f"Format must be one of: {', '.join(sorted(FORMATS))}.")
        if voice is not None and not re.fullmatch(r"[\w\-]{1,40}", voice):
            raise error(400, "bad_voice", "Unknown voice.")
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
            "voice": voice, "speed": min(max(speed, 0.6), 1.5), "format": format,
            "engine": engine, "force": False,
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
