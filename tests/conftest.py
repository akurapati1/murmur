import time

import pytest

from murmur.config import Settings
from murmur.store import JobStore
from tests import pdfs


@pytest.fixture
def settings(tmp_path):
    s = Settings(data_dir=tmp_path / "data", models_dir=tmp_path / "models", tts_engine="dummy", chunk_chars=300)
    s.ensure_dirs()
    return s


@pytest.fixture
def store(settings):
    return JobStore(settings.db_path)


@pytest.fixture
def novel(tmp_path):
    return pdfs.make_novel(tmp_path / "novel.pdf")


@pytest.fixture
def technical(tmp_path):
    return pdfs.make_technical(tmp_path / "technical.pdf")


def new_job(store, settings, src, **options):
    """Register a job whose upload is a copy of ``src``."""
    import shutil
    import uuid

    job_id = uuid.uuid4().hex[:12]
    dest = settings.uploads_dir / f"{job_id}.pdf"
    shutil.copyfile(src, dest)
    opts = {"voice": None, "speed": 1.0, "format": "m4b", "engine": "dummy", "force": False, **options}
    store.create(job_id=job_id, filename=src.name, size_bytes=dest.stat().st_size, input_path=dest, options=opts)
    return job_id


def wait_for(fetch, statuses, timeout=60.0):
    """Poll ``fetch()`` (returns a job dict) until its status is one of ``statuses``."""
    deadline = time.time() + timeout
    job = fetch()
    while job["status"] not in statuses:
        if time.time() > deadline:
            raise AssertionError(f"timed out; last status {job['status']!r}: {job.get('message')} {job.get('error')}")
        time.sleep(0.1)
        job = fetch()
    return job
