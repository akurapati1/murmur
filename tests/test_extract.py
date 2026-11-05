from murmur.extract import extract_book
from tests import pdfs


def all_text(book):
    return "\n".join(p for c in book.chapters for p in c.paragraphs)


def test_chapters_from_outline(novel):
    book = extract_book(novel)
    assert [c.title for c in book.chapters] == [f"Chapter {i}" for i in range(1, 5)]
    assert book.title == "The Lighthouse Keeper"
    assert book.author == "A. Writer"
    assert book.word_count > 3000


def test_running_headers_and_page_numbers_are_removed(novel):
    text = all_text(extract_book(novel))
    assert "LIGHTHOUSE KEEPER" not in text.upper().replace("THE OLD LIGHTHOUSE KEEPER", "")
    paragraphs = [p for c in extract_book(novel).chapters for p in c.paragraphs]
    assert not any(p.strip().isdigit() for p in paragraphs)


def test_heading_paragraph_is_not_repeated_in_body(novel):
    for chapter in extract_book(novel).chapters:
        assert chapter.paragraphs[0].lower() != chapter.title.lower()


def test_chapters_detected_without_an_outline(tmp_path):
    path = pdfs.make_novel(tmp_path / "flat.pdf", toc=False)
    book = extract_book(path)
    assert len(book.chapters) == 4
    assert book.chapters[0].title == "Chapter 1"


def test_hyphenated_words_are_rejoined(tmp_path):
    book = extract_book(pdfs.make_hyphenated(tmp_path / "hy.pdf"))
    text = all_text(book)
    assert "extraordinary" in text
    assert "extra-" not in text


def test_back_matter_is_skipped(tmp_path):
    path = pdfs.make_novel(tmp_path / "idx.pdf", back_matter=True)
    titles = [c.title for c in extract_book(path).chapters]
    assert "Index" not in titles
    assert len(titles) == 4


def test_back_matter_can_be_kept(tmp_path):
    path = pdfs.make_novel(tmp_path / "idx.pdf", back_matter=True)
    titles = [c.title for c in extract_book(path, skip_back_matter=False).chapters]
    assert "Index" in titles


def test_captions_are_dropped(technical):
    text = all_text(extract_book(technical))
    assert "Block diagram" not in text


def test_progress_callback_reports_pages(novel):
    seen = []
    extract_book(novel, progress=lambda frac, msg: seen.append(frac))
    assert seen and seen[-1] == 1.0
