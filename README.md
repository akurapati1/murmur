# Murmur

**Read any book to me.** Drop in a book as a PDF and get back an audiobook with chapters. It's free, open source, and everything runs on your own computer: no accounts, no API keys, no per-book cost.

![status](https://img.shields.io/badge/cost%20per%20book-%240-4F35A0) ![license](https://img.shields.io/badge/license-MIT-1B1938)

## What it does

1. **Checks the file is really a PDF.** It looks at the file's contents, not its extension, and tells you plainly if it's an EPUB, an image, a damaged file or a password-protected PDF.
2. **Decides whether the book works as audio.** It samples pages across the book and measures how much of the meaning lives in figures (big images, dense diagrams, numbered captions) versus prose. A few graphs, tables or equations are fine. If the book looks technical, Murmur says so and lets you choose: *Convert anyway* or pick another file.
3. **Starts at the story.** Cover and title pages, copyright and ISBN blocks, dedications, epigraphs, praise, "also by" lists, acknowledgments and author bios are cut. Narration begins at the prologue, prelude or first chapter, and ends where the book does. See [the narration director](#the-narration-director).
4. **Reads the book properly.** Running headers, footers and page numbers are removed. Hyphenated words are rejoined. Figure captions, tables of contents and back matter (index, bibliography) are skipped. Chapters come from the PDF outline, or from headings when there is no outline.
5. **Narrates it like a person would**, sentence by sentence, with deliberate pauses and pacing, in a built-in voice or in **your own voice**. Progress is saved as it goes.
6. **Packages it** as an `.m4b` audiobook with chapter markers and cover art (also: a single MP3, or a zip with one MP3 per chapter).

Scanned books (pages that are photos of text) are read with OCR if [Tesseract](https://github.com/tesseract-ocr/tesseract) is installed.

## Quick start

You need Python 3.10+ and [ffmpeg](https://ffmpeg.org/download.html).

```bash
git clone <your-repo-url> murmur
cd murmur
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
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

## The narration director

Plain text-to-speech reads everything at one pace with one length of pause. Murmur puts a *director* between the PDF and the voice. It works in two layers, and the second one is optional.

**Rules, always on, fully offline** ([`matter.py`](murmur/matter.py), [`prosody.py`](murmur/prosody.py), [`speech.py`](murmur/speech.py)):

- Finds where the story begins and ends and cuts the rest. It recognises front and back matter by chapter title and by paragraph content (copyright and ISBN text, publisher blocks, dedications, epigraphs, "About the author"). For fiction it also skips a foreword or preface by someone else; for non-fiction it keeps the author's introduction. It will never cut more than 40% of a book, and `murmur direct` shows exactly what it removed.
- Pauses are chosen per sentence: full stop, question, exclamation, ellipsis, a speaker change in dialogue, the end of a paragraph, a scene break (`* * *`, kept from the PDF), a chapter heading. The voice's own leading and trailing silence is trimmed so the pauses are exactly the ones chosen.
- Numbers, years, money, ordinals, Roman numerals in headings and abbreviations are spoken properly ("1995" becomes "nineteen ninety-five", "Mr." becomes "Mister"), and SHOUTED words are not spelled out.

**AI, optional:** an AI reads the book the way a studio director would, and guides the rules.

- It confirms where the story starts and ends (its answer is validated and ignored if implausible).
- For every passage it can set pace (slow for grief, fast for action), the length of the pause afterwards (up to "dramatic" for a cliffhanger), intensity (a whisper versus a shout) and which text is not part of the book at all.
- It supplies pronunciations for names and invented words ("Hermione" becomes "her-MY-oh-nee").

Choose one:

| Provider | Setup | Notes |
|---|---|---|
| **Claude** | `pip install -e ".[ai]"` and `export ANTHROPIC_API_KEY=...` | Uses the official `anthropic` SDK. Default model `claude-opus-5-5`; set `MURMUR_AI_MODEL=claude-haiku-5-5` for a much cheaper run. Annotation calls use low effort. **The book's text is sent to Anthropic.** |
| **A local model** | Run [Ollama](https://ollama.com) (detected automatically) or set `MURMUR_AI_BASE_URL` and `MURMUR_AI_MODEL` for any OpenAI-compatible server | Free, and the text never leaves your computer. Quality depends on the model; 7B+ instruction models work. |
| **None** | nothing | The rules above. |

An AI is purely advisory: if it errors, times out or answers nonsense, that part of the book silently falls back to the rules and the job continues. Answers are cached per job, so resuming an interrupted book never pays twice. Turn it off per book with the checkbox in the UI or `--ai off`.

**Check before you narrate.** This prints what would be cut, where the story starts, pronunciations, and the first passages with their pause lengths, without producing audio:

```bash
murmur direct my-book.pdf
murmur direct my-book.pdf --script script.json    # every sentence with its pause, speed and energy
murmur convert my-book.pdf --pause-style relaxed  # snappy | natural | relaxed
murmur convert my-book.pdf --keep-opening         # read everything, credits included
```

## Your own voice

Upload any recording of yourself (10 to 30 seconds of clear speech; MP3, WAV, M4A, OGG, FLAC and most video files work), give it a name, and narrate whole books in it.

```bash
pip install -e ".[clone]"          # Chatterbox, MIT licence. Also: murmur models clone
murmur voices add "My voice" recording.m4a --default
murmur convert my-book.pdf --voice "My voice"
murmur voices list
```

In the web app, use **Your voices**: choose the file, name it, confirm it's your own voice (or that you have permission), and save. The voice appears in the voice menu, and **the voice you select is remembered and used by default** for the next audiobook. You can play the saved clip, switch voices, or delete one.

How it works and what to expect:

- The recording is decoded with ffmpeg, the cleanest stretch of speech (up to 28 s) is picked, long silences are squeezed out and the level is normalised. Murmur warns about background noise or distortion, and refuses clips with under 5 s of speech.
- A cloning engine imitates that voice for every sentence. **Chatterbox** (MIT) is the default; **Coqui XTTS-v2** is also supported (`pip install -e ".[xtts]"`, `MURMUR_CLONE_ENGINE=xtts`, and note its licence is **non-commercial**). The director's per-sentence energy maps to Chatterbox's expressiveness.
- **Cloning is heavy.** It pulls in PyTorch and about 2 GB of weights. Use a GPU (CUDA or Apple Silicon) if you can; on a plain CPU expect it to run several times slower than real time, so a novel can take days. Kokoro remains the fast option.
- It will not be a perfect copy. Quality follows the recording: a quiet room and natural speech matter more than length.
- Only clone your own voice, or one you have explicit permission to use. Murmur asks you to confirm this every time.
- Voices are stored in `<data>/voices/` on your computer and are never uploaded.

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
   validate ─▶ evaluate ─▶ extract ─▶ direct ─────▶ narrate ─────────▶ encode
   (PyMuPDF)   (classifier) (chapters,  (where the     (Kokoro/Piper/     (ffmpeg:
                             cleaning)   story starts,  eSpeak/your clone, m4b/mp3/zip,
                                         pauses, pace;  per-sentence,      chapters, cover)
                                         rules + AI)    checkpoints)
```

- **Resumable.** The directed script is saved (`plan.json`) before narration starts, and audio is written sentence by sentence with a checkpoint, so a crash or restart loses at most one sentence and never repeats AI calls.
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
| `MURMUR_AI` | `auto` | `auto`, `anthropic`, `openai` (any OpenAI-compatible server, incl. Ollama), `off` |
| `MURMUR_AI_MODEL` | `claude-opus-5-5` / first Ollama model | Model used by the director |
| `MURMUR_AI_BASE_URL`, `MURMUR_AI_API_KEY` | | OpenAI-compatible endpoint, e.g. `http://localhost:1234/v1` |
| `MURMUR_AI_EFFORT` | `low` | Claude effort for annotation calls (`""` to omit, for older models) |
| `MURMUR_AI_CONCURRENCY` | `4` | Annotation calls in flight at once |
| `MURMUR_PAUSE_STYLE` | `natural` | `snappy`, `natural`, `relaxed` |
| `MURMUR_START_AT_STORY` | `1` | Cut front and back matter |
| `MURMUR_CLONE_ENGINE` | `auto` | `auto`, `chatterbox`, `xtts` |

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/jobs` | Upload a PDF (`file`, optional `voice`, `speed`, `format`, `ai`=`auto\|off`, `start_at_story`, `pause_style`). Returns `415` immediately if it isn't a PDF |
| `GET` | `/api/jobs`, `/api/jobs/{id}` | List jobs, read one |
| `GET` | `/api/jobs/{id}/events` | Live progress (server-sent events) |
| `POST` | `/api/jobs/{id}/confirm` | Convert a book that was flagged as technical |
| `POST` | `/api/jobs/{id}/cancel` | Cancel or discard |
| `GET` | `/api/jobs/{id}/download` | The audiobook (supports seeking; add `?attachment=true` to download) |
| `DELETE` | `/api/jobs/{id}` | Delete the job and its files |
| `POST` | `/api/preview` | Short voice sample (works for your voices too) |
| `GET` | `/api/health`, `/api/ai` | Setup status, which AI director is active |
| `GET` | `/api/voices` | Built-in voices, your voices (`custom`), the default, cloning engine status |
| `POST` | `/api/voices` | Save a voice: multipart `name`, `file`, `consent=true` |
| `PATCH` / `DELETE` | `/api/voices/{id}` | Rename or delete a voice |
| `PUT` | `/api/voices/default` | `{"voice": "<id>"}` or `null`: the voice used when none is chosen |
| `GET` | `/api/voices/{id}/sample` | The cleaned reference clip |

Interactive docs are at `/docs` while the server is running.

## Known limitations

- **Multi-column layouts** (some academic papers, newspapers) can interleave columns. Single-column books, which is nearly all novels and non-fiction, read correctly.
- **Equations and code** are read as plain text, which sounds odd. That's why heavy ones trigger the technical-book check.
- **English only** for now. Kokoro and Piper both support other languages; wiring in language detection is a natural next step.
- **Encrypted PDFs** must be unlocked first.
- **Not yet run against real models:** the Claude and Ollama directors (tested with scripted fakes and the logic is validated, but not against a live model), and the Chatterbox and XTTS cloning engines (the voice library, routing and audio preparation are tested; the model calls are thin and untested here). Please report anything odd.
- Front-matter detection is heuristic. A book with unusual structure may keep a stray page or lose a short real one. Run `murmur direct` first on anything important, and use `--keep-opening` if it cuts too much.
- Pronunciation fixes and per-sentence pacing are advisory: the built-in voices ignore some of them, and Chatterbox does not support a direct speed control.
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
