import pymupdf

from murmur.validation import validate_pdf


def test_valid_pdf(novel):
    result = validate_pdf(novel)
    assert result.ok
    assert result.page_count == 12
    assert result.title == "The Lighthouse Keeper"
    assert result.author == "A. Writer"


def test_plain_text_is_not_a_pdf(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("just some words")
    r = validate_pdf(f)
    assert not r.ok and r.code == "not_pdf"


def test_extension_is_not_trusted(tmp_path):
    f = tmp_path / "fake.pdf"
    f.write_text("this is not really a pdf")
    assert validate_pdf(f).code == "not_pdf"


def test_zip_based_files_get_a_helpful_message(tmp_path):
    f = tmp_path / "book.pdf"
    f.write_bytes(b"PK\x03\x04" + b"\x00" * 100)
    r = validate_pdf(f)
    assert r.code == "not_pdf" and "EPUB" in r.reason


def test_empty_file(tmp_path):
    f = tmp_path / "empty.pdf"
    f.write_bytes(b"")
    assert validate_pdf(f).code == "empty"


def test_truncated_pdf(novel, tmp_path):
    f = tmp_path / "cut.pdf"
    f.write_bytes(novel.read_bytes()[:40])
    assert validate_pdf(f).code in {"corrupt", "empty"}


def test_password_protected_pdf(tmp_path):
    doc = pymupdf.open()
    doc.new_page().insert_text((50, 50), "secret")
    f = tmp_path / "locked.pdf"
    doc.save(f, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner")
    r = validate_pdf(f)
    assert not r.ok and r.code == "encrypted"
