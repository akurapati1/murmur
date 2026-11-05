# Murmur

**Read any book to me.** Drop in a book as a PDF and get back an audiobook with chapters. It's free, open source, and everything runs on your own computer: no accounts, no API keys, no per-book cost.

![status](https://img.shields.io/badge/cost%20per%20book-%240-4F35A0) ![license](https://img.shields.io/badge/license-MIT-1B1938)

## What it does

1. **Checks the file is really a PDF.** It looks at the file's contents, not its extension, and tells you plainly if it's an EPUB, an image, a damaged file or a password-protected PDF.
2. **Decides whether the book works as audio.** It samples pages across the book and measures how much of the meaning lives in figures (big images, dense diagrams, numbered captions) versus prose. A few graphs, tables or equations are fine. If the book looks technical, Murmur says so and lets you choose: *Convert anyway* or pick another file.
3. **Reads the book properly.** Running headers, footers and page numbers are removed. Hyphenated words are rejoined. Figure captions, tables of contents and back matter (index, bibliography) are skipped. Chapters come from the PDF outline, or from headings when there is no outline.
4. **Narrates it** with a free, open-source voice, one chapter at a time, saving progress as it goes.
5. **Packages it** as an `.m4b` audiobook with chapter markers and cover art (also: a single MP3, or a zip with one MP3 per chapter).

Scanned books (pages that are photos of text) are read with OCR if [Tesseract](https://github.com/tesseract-ocr/tesseract) is installed.

## Quick start

You need Python 3.10+ and [ffmpeg](https://ffmpeg.org/download.html).

```bash
git clone <your-repo-url> murmur && cd murmur
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[kokoro]"

murmur models kokoro --small     # one-time download, about 120 MB
murmur doctor                    # checks that everything is ready
murmur serve                     # then open http://127.0.0.1:8000
```

No browser needed? Convert from the terminal:

```bash
murmur analyze  my-book.pdf                  # will it work as an audiobook?
murmur convert  my-book.pdf --out ~/Audiobooks
murmur convert  textbook.pdf --force         # convert even if it looks technical
```

### Docker

```bash
docker compose up -d --build
docker compose run --rm murmur murmur models kokoro --small
# open http://127.0.0.1:8000
```

## Cost and voice options

Everything below is free. The only "cost" is your computer's time and electricity.

| Engine | Quality | Speed | Size | Setup |
|---|---|---|---|---|
| **Kokoro** (default) | Natural, expressive | Slower | 90 MB (`--small`) or 310 MB | `pip install -e ".[kokoro]"` then `murmur models kokoro` |
| **Piper** | Clear, a bit flatter | Much faster | about 60 MB per voice | `pip install -e ".[piper]"` then `murmur models piper` |
| **eSpeak NG** | Robotic | Fastest | tiny | `apt install espeak-ng` or `brew install espeak-ng` |

Murmur picks the best one that's installed (Kokoro, then Piper, then eSpeak). Force one with `MURMUR_TTS_ENGINE=piper`.

### How long does a book take?

Narration speed depends heavily on your CPU, so measure on your machine rather than trusting a number from mine. For reference, on a small 2-core cloud machine the small Kokoro model took about 5 minutes to produce 3 minutes of audio (roughly 1.6x slower than real time). A modern laptop with more cores should be noticeably faster, but a long novel with Kokoro can still take several hours. Because progress is saved after every passage, you can stop and resume (restart `murmur serve` and unfinished books continue where they left off).

If speed matters more than polish, use Piper. The progress bar shows a live time estimate once narration starts.

## How it decides "novel or technical?"

Murmur samples up to 40 pages evenly across the book and measures:

- the share of pages dominated by a large image or a dense vector diagram,
- how often numbered "Figure 3." or "Table 2." captions appear,
- how much of the text is equations or source code,
- how many pages are page-photos with no text (scans), and whether the pages look like slides.

| Verdict | What happens |
|---|---|
| Works well as an audiobook | Converts automatically |
| Works, with a few figures skipped | Converts automatically |
| Looks like a technical book / not a book | Asks you first (*Convert anyway*) |
| Scanned pages | Converts with OCR if Tesseract is installed, otherwise explains how to enable it |

The thresholds are heuristics in [`murmur/classifier.py`](murmur/classifier.py). Tune them for your library.

## Architecture

```
 Browser UI ──upload──▶ FastAPI ──▶ SQLite job store ◀── status / SSE progress
                           │
                           ▼
                     JobManager (worker thread, resumes after restart)
                           │
   validate ─▶ evaluate ─▶ extract ─▶ narrate ─────────▶ encode
   (PyMuPDF)   (classifier) (chapters,   (Kokoro/Piper/    (ffmpeg:
                             cleaning)    eSpeak, per-chunk  m4b/mp3/zip,
                                          checkpoints)       chapters, cover)
```

- **Resumable.** Audio is written chunk by chunk with a checkpoint, so a crash or restart loses at most one passage.
- **Large files.** Uploads stream to disk in 1 MB pieces (default limit 500 MB, set `MURMUR_MAX_UPLOAD_MB`). Pages are read one at a time and audio is concatenated on disk, so memory use stays flat however long the book is.
- **Scaling out.** `JobManager` is a small in-process queue. To run many books in parallel across machines, replace it with Celery or RQ and point every worker at the same job store; the pipeline function `process_job(job_id, store, settings, cancel)` is already independent of the web layer.

## Configuration

All settings are environment variables prefixed with `MURMUR_`.

| Variable | Default | Meaning |
|---|---|---|
| `MURMUR_HOME` | `~/.murmur` | Where data and models live |
| `MURMUR_TTS_ENGINE` | `auto` | `auto`, `kokoro`, `piper`, `espeak` |
| `MURMUR_VOICE` | `af_heart` | Default voice |
| `MURMUR_WORKERS` | `1` | Books converted at once (narration already uses all CPU cores) |
| `MURMUR_MAX_UPLOAD_MB` | `500` | Upload size limit |
| `MURMUR_BITRATE` | `64k` | Audio bitrate (64k is plenty for speech) |
| `MURMUR_SKIP_BACK_MATTER` | `1` | Skip index, bibliography, notes |
| `MURMUR_SKIP_CAPTIONS` | `1` | Skip "Figure N." captions |
| `MURMUR_KEEP_WORK` | `0` | Keep scratch files after a book finishes |

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/jobs` | Upload a PDF (`file`, optional `voice`, `speed`, `format`). Returns `415` immediately if it isn't a PDF |
| `GET` | `/api/jobs`, `/api/jobs/{id}` | List jobs, read one |
| `GET` | `/api/jobs/{id}/events` | Live progress (server-sent events) |
| `POST` | `/api/jobs/{id}/confirm` | Convert a book that was flagged as technical |
| `POST` | `/api/jobs/{id}/cancel` | Cancel or discard |
| `GET` | `/api/jobs/{id}/download` | The audiobook (supports seeking; add `?attachment=true` to download) |
| `DELETE` | `/api/jobs/{id}` | Delete the job and its files |
| `POST` | `/api/preview` | Short voice sample |
| `GET` | `/api/health`, `/api/voices` | Setup status, voice list |

Interactive docs are at `/docs` while the server is running.

## Known limitations

- **Multi-column layouts** (some academic papers, newspapers) can interleave columns. Single-column books, which is nearly all novels and non-fiction, read correctly.
- **Equations and code** are read as plain text, which sounds odd. That's why heavy ones trigger the technical-book check.
- **English only** for now. Kokoro and Piper both support other languages; wiring in language detection is a natural next step.
- **Encrypted PDFs** must be unlocked first.
- Only the Kokoro and dummy (test) engines were run against real models during development. The Piper and eSpeak paths are covered by logic but not yet exercised against the real binaries, so report anything odd.
- Only upload books you have the right to convert. Audiobooks you make are for your own use.

## Development

```bash
pip install -e ".[dev]"
pytest          # about 35 seconds; builds synthetic novels, technical books and scans on the fly
ruff check .
```

The test suite uses a silent "dummy" voice engine (`MURMUR_TTS_ENGINE=dummy`) so it needs no model downloads. It needs ffmpeg; OCR tests run automatically when Tesseract is present.

## Licenses

Murmur is MIT. Its dependencies are separate projects with their own licenses: [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) and [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) (Apache-2.0), PyMuPDF (AGPL-3.0 or commercial), ffmpeg (LGPL/GPL depending on build), and Piper and eSpeak NG (GPL-3.0, run as separate programs). Individual Piper voices carry their own licenses; check a voice's model card before redistributing audio made with it. **PyMuPDF is AGPL:** fine for personal use and for open-source projects under a compatible license, but if you turn Murmur into a hosted service you need to comply with the AGPL or buy a commercial licence.
