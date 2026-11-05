from murmur.chunker import (
    HEADING_PAUSE,
    PARAGRAPH_PAUSE,
    build_chunks,
    normalize_for_speech,
    pack_paragraph,
    split_sentences,
)
from murmur.models import Chapter


def test_abbreviations_do_not_end_sentences():
    assert split_sentences("Dr. Smith met Mr. Jones at noon. They talked for hours.") == [
        "Dr. Smith met Mr. Jones at noon.", "They talked for hours."]


def test_initials_do_not_end_sentences():
    out = split_sentences("J. K. Rowling wrote it. It sold well.")
    assert len(out) == 2 and out[0].startswith("J. K. Rowling")


def test_questions_and_quotes():
    out = split_sentences('"Where are you going?" she asked. He said nothing.')
    assert len(out) == 2


def test_chunks_respect_the_limit():
    text = " ".join(f"This is sentence number {i} in a long paragraph." for i in range(60))
    chunks = pack_paragraph(text, 200)
    assert len(chunks) > 5
    assert all(len(c) <= 200 for c in chunks)
    assert " ".join(chunks).count("sentence number") == 60


def test_one_endless_sentence_is_still_split():
    text = "word " * 400
    assert all(len(c) <= 100 for c in pack_paragraph(text, 100))


def test_build_chunks_adds_heading_and_pauses():
    chapter = Chapter("Chapter 1: The Sea", ["First paragraph here. It has two sentences.", "Second paragraph."])
    chunks = build_chunks(chapter, 350)
    assert chunks[0].text == "Chapter 1: The Sea."
    assert chunks[0].pause_after == HEADING_PAUSE
    assert chunks[1].pause_after == PARAGRAPH_PAUSE
    assert chunks[-1].text == "Second paragraph."


def test_normalize_for_speech():
    out = normalize_for_speech("See https://example.com/a?b=1 now\u2014or never[12] & so on\u200b")
    assert "link" in out and "http" not in out
    assert "[12]" not in out and "&" not in out and "\u200b" not in out
    assert "\u2014" not in out
