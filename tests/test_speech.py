import pytest

from murmur.speech import (
    apply_pronunciations,
    calm_shouting,
    number_to_words,
    ordinal_to_words,
    roman_to_int,
    speak_heading,
    speak_numbers,
    speak_ready,
    year_to_words,
)


@pytest.mark.parametrize("n,words", [
    (0, "zero"), (7, "seven"), (13, "thirteen"), (21, "twenty-one"), (100, "one hundred"),
    (342, "three hundred forty-two"), (1000, "one thousand"), (1234, "one thousand two hundred thirty-four"),
    (2_000_001, "two million one"), (45_000_000_000, "forty-five billion"),
])
def test_number_to_words(n, words):
    assert number_to_words(n) == words


@pytest.mark.parametrize("n,words", [
    (1, "first"), (2, "second"), (3, "third"), (4, "fourth"), (12, "twelfth"), (20, "twentieth"),
    (21, "twenty-first"), (100, "one hundredth"), (101, "one hundred first"),
])
def test_ordinals(n, words):
    assert ordinal_to_words(n) == words


@pytest.mark.parametrize("year,words", [
    (1995, "nineteen ninety-five"), (1905, "nineteen oh five"), (1900, "nineteen hundred"),
    (2005, "two thousand five"), (2010, "twenty ten"), (1812, "eighteen twelve"),
])
def test_years(year, words):
    assert year_to_words(year) == words


def test_numbers_in_running_text():
    assert speak_numbers("In 1995 she paid $5.50 for 3 books, 50% off.") == (
        "In nineteen ninety-five she paid five dollars and fifty cents for three books, fifty percent off.")
    assert speak_numbers("The 21st of May, 1,200 people, 3.14 miles") == (
        "The twenty-first of May, one thousand two hundred people, three point one four miles")
    assert speak_numbers("the 1990s") == "the nineteen nineties"
    assert speak_numbers("It cost £2 million") == "It cost two million pounds"


def test_roman_numerals_and_headings():
    assert roman_to_int("XIV") == 14 and roman_to_int("MCMXC") == 1990
    assert roman_to_int("MIX") == 1009 and roman_to_int("ABC") is None
    assert speak_heading("Chapter XII") == "Chapter twelve"
    assert speak_heading("Part IV: The Storm") == "Part four: The Storm"
    assert speak_heading("IX") == "Chapter nine"
    assert speak_heading("Chapter 7") == "Chapter seven"
    assert speak_heading("Civil") == "Civil"  # not a numeral


def test_abbreviations_and_latin():
    out = speak_ready("Dr. Smith and Mrs. Jones met Mr. Lee, e.g. at noon, etc.")
    assert "Doctor Smith" in out and "Missus Jones" in out and "Mister Lee" in out
    assert "for example," in out and "et cetera" in out


def test_shouting_is_not_spelled_out():
    assert calm_shouting("GET OUT OF MY HOUSE!") == "Get out of my house!"
    assert calm_shouting("He yelled STOP and left.") == "He yelled Stop and left."
    assert calm_shouting("The FBI arrived at NASA.") == "The FBI arrived at NASA."


def test_pronunciation_overrides_whole_words_only():
    out = apply_pronunciations("Hermione met Hermiones and Hermione.", {"Hermione": "her-MY-oh-nee"})
    assert out == "her-MY-oh-nee met Hermiones and her-MY-oh-nee."
    assert apply_pronunciations("text", None) == "text"
    assert apply_pronunciations("text", {"": "x", "text": " "}) == "text"
