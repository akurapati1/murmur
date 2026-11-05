from murmur.classifier import analyze_pdf
from murmur.models import Verdict
from tests import pdfs


def test_novel_is_audiobook_friendly(novel):
    report = analyze_pdf(novel)
    assert report.verdict == Verdict.AUDIOBOOK_FRIENDLY
    assert report.convertible
    assert report.estimated_words > 3000
    assert report.stats["figure_page_ratio"] == 0


def test_technical_book_needs_visuals(technical):
    report = analyze_pdf(technical)
    assert report.verdict == Verdict.NEEDS_VISUALS
    assert not report.convertible
    assert report.visual_score >= 0.6
    assert any("figure" in r.lower() for r in report.reasons)


def test_a_few_figures_is_borderline(tmp_path):
    path = pdfs.make_novel(tmp_path / "mixed.pdf", chapters=6, pages_per_chapter=4, figure_pages=3)
    report = analyze_pdf(path)
    assert report.verdict == Verdict.BORDERLINE
    assert report.convertible


def test_scanned_pages_are_detected(tmp_path):
    report = analyze_pdf(pdfs.make_scanned(tmp_path / "scan.pdf"))
    assert report.verdict == Verdict.SCANNED
    assert report.needs_ocr


def test_slide_deck_is_not_a_book(tmp_path):
    report = analyze_pdf(pdfs.make_slides(tmp_path / "slides.pdf"))
    assert report.verdict == Verdict.NOT_A_BOOK
    assert not report.convertible


def test_sampling_handles_long_documents(tmp_path):
    path = pdfs.make_novel(tmp_path / "long.pdf", chapters=30, pages_per_chapter=5)
    report = analyze_pdf(path, sample_pages=20)
    assert report.page_count == 150
    assert report.sampled_pages == 20
    assert report.verdict == Verdict.AUDIOBOOK_FRIENDLY
