"""In-process job queue. Swap this class for Celery/RQ if you ever need multiple machines."""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from .config import Settings
from .models import ACTIVE_STATUSES, JobStatus
from .pipeline import process_job
from .store import JobStore

log = logging.getLogger("murmur.jobs")

Runner = Callable[[str, JobStore, Settings, threading.Event], None]


class CancelToken(threading.Event):
    """A cancel flag that also records *why*: a user cancel is final, a shutdown is resumable."""

    def __init__(self) -> None:
        super().__init__()
        self.shutdown = False


class JobManager:
    def __init__(self, store: JobStore, settings: Settings, runner: Runner = process_job):
        self.store = store
        self.settings = settings
        self._runner = runner
        self._pool = ThreadPoolExecutor(max_workers=settings.workers, thread_name_prefix="murmur-worker")
        self._events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def submit(self, job_id: str) -> None:
        with self._lock:
            if job_id in self._events:
                return  # already queued or running
            event = CancelToken()
            self._events[job_id] = event
        self.store.update(job_id, status=JobStatus.QUEUED.value, message="Waiting for a free worker", eta_sec=None)
        self._pool.submit(self._run, job_id, event)

    def _run(self, job_id: str, event: threading.Event) -> None:
        try:
            self._runner(job_id, self.store, self.settings, event)
        except Exception:  # pragma: no cover - runner handles its own errors
            log.exception("worker crashed on job %s", job_id)
        finally:
            with self._lock:
                self._events.pop(job_id, None)

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            event = self._events.get(job_id)
        if event is None:
            return False
        event.set()
        return True

    def is_active(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._events

    def recover(self) -> int:
        """Re-queue jobs that were interrupted by a restart. Their checkpoints let them resume."""
        jobs = self.store.list_by_status(ACTIVE_STATUSES)
        for job in jobs:
            self.submit(job["id"])
        return len(jobs)

    def shutdown(self) -> None:
        with self._lock:
            for event in self._events.values():
                event.shutdown = True  # keep checkpoints; recover() resumes these on next start
                event.set()
        self._pool.shutdown(wait=True, cancel_futures=True)
