"""User-facing install hints must be commands that actually work for this project."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [p for p in (ROOT / "murmur").rglob("*") if p.suffix in {".py", ".html"}]


def _declared_extras() -> set[str]:
    section = (ROOT / "pyproject.toml").read_text(encoding="utf-8").split("[project.optional-dependencies]")[1]
    section = section.split("\n[", 1)[0]
    return set(re.findall(r"^(\w+)\s*=", section, flags=re.M))


def test_hints_do_not_name_the_unrelated_murmur_package():
    # The distribution is murmur-audiobooks, so "murmur[x]" would look for a different project on PyPI.
    offenders = [p.name for p in SOURCES if "murmur[" in p.read_text(encoding="utf-8")]
    assert offenders == []


def test_hints_only_name_extras_that_exist():
    extras = _declared_extras()
    hinted = {m for p in SOURCES for m in re.findall(r'pip install -e \\?"\.\[(\w+)\]', p.read_text(encoding="utf-8"))}
    assert hinted, "expected to find install hints in the source"
    assert hinted <= extras
