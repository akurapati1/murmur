"""Make written text sayable: numbers, years, money, abbreviations, shouting and pronunciation fixes.

Everything here is deterministic and offline. The AI director adds *book-specific* pronunciation
overrides (names, invented words) on top, which are applied by :func:`apply_pronunciations`.
"""
from __future__ import annotations

import re

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
         "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = [(10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (10**3, "thousand")]
_ORDINAL_IRREGULAR = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
                      "nine": "ninth", "twelve": "twelfth"}

_TITLES = {
    "Mr": "Mister", "Mrs": "Missus", "Ms": "Miz", "Dr": "Doctor", "Prof": "Professor", "Sr": "Senior",
    "Jr": "Junior", "Capt": "Captain", "Col": "Colonel", "Gen": "General", "Lt": "Lieutenant",
    "Sgt": "Sergeant", "Rev": "Reverend", "Hon": "Honorable", "Mt": "Mount",
}
_TITLE_RE = re.compile(r"\b(" + "|".join(_TITLES) + r")\.(?=\s|$)")
_LATIN = {"e.g.": "for example,", "i.e.": "that is,", "etc.": "et cetera", "vs.": "versus", "cf.": "compare"}
_LATIN_RE = re.compile("|".join(re.escape(k) for k in _LATIN), re.IGNORECASE)

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
_ROMAN_RE = re.compile(r"^(?=[IVXLCDM])M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")
_HEADING_ROMAN_RE = re.compile(r"\b(Chapter|Part|Book|Volume|Act|Section|Canto)\s+([IVXLCDM]{1,8})\b")

_YEAR_RE = re.compile(r"(?<![\d,.$£€])\b(1[1-9]\d\d|20\d\d)(s)?\b(?![,\d]*\d)(?!%)")
_MONEY_RE = re.compile(r"([$£€])\s?(\d[\d,]*)(?:\.(\d{2}))?\b(?:\s?(million|billion|thousand|trillion)\b)?")
_PERCENT_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s?%")
_ORDINAL_RE = re.compile(r"\b(\d[\d,]*)(st|nd|rd|th)\b", re.IGNORECASE)
_DECIMAL_RE = re.compile(r"\b(\d[\d,]*)\.(\d+)\b")
_NUMBER_RE = re.compile(r"\b\d[\d,]*\b")
_SHOUT_RE = re.compile(r"\b[A-Z][A-Z'’\-]{2,}\b")

_CURRENCY = {"$": ("dollar", "cent"), "£": ("pound", "penny"), "€": ("euro", "cent")}


def number_to_words(n: int) -> str:
    """1234 -> 'one thousand two hundred thirty-four'."""
    if n < 0:
        return "minus " + number_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + (f"-{_ONES[ones]}" if ones else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return f"{_ONES[hundreds]} hundred" + (f" {number_to_words(rest)}" if rest else "")
    for size, name in _SCALES:
        if n >= size:
            head, rest = divmod(n, size)
            return f"{number_to_words(head)} {name}" + (f" {number_to_words(rest)}" if rest else "")
    return str(n)  # pragma: no cover - unreachable


def ordinal_to_words(n: int) -> str:
    words = number_to_words(n)
    head, sep, last = words.rpartition(" ")
    if not sep:
        head, sep, last = words.rpartition("-")
    if last in _ORDINAL_IRREGULAR:
        last = _ORDINAL_IRREGULAR[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return f"{head}{sep}{last}"


def year_to_words(year: int) -> str:
    """1995 -> 'nineteen ninety-five', 1905 -> 'nineteen oh five', 2005 -> 'two thousand five'."""
    if 2000 <= year <= 2009:
        return number_to_words(year)
    high, low = divmod(year, 100)
    if low == 0:
        return f"{number_to_words(high)} hundred"
    if low < 10:
        return f"{number_to_words(high)} oh {_ONES[low]}"
    return f"{number_to_words(high)} {number_to_words(low)}"


def roman_to_int(text: str) -> int | None:
    text = text.upper()
    if not _ROMAN_RE.match(text):
        return None
    total = 0
    for i, ch in enumerate(text):
        value = _ROMAN_VALUES[ch]
        total += -value if i + 1 < len(text) and _ROMAN_VALUES[text[i + 1]] > value else value
    return total


def _to_int(digits: str) -> int:
    return int(digits.replace(",", ""))


def _decimal_digits(frac: str) -> str:
    return " ".join(_ONES[int(d)] for d in frac)


def _money(m: re.Match) -> str:
    symbol, whole, cents, scale = m.group(1), _to_int(m.group(2)), m.group(3), m.group(4)
    major, minor = _CURRENCY[symbol]
    if scale:
        return f"{number_to_words(whole)} {scale} {major}s"
    out = f"{number_to_words(whole)} {major}{'' if whole == 1 else 's'}"
    if cents and int(cents):
        c = int(cents)
        out += f" and {number_to_words(c)} {'pence' if minor == 'penny' and c != 1 else minor if c == 1 else minor + 's'}"
    return out


def speak_numbers(text: str) -> str:
    """Spell out years, money, percentages, ordinals, decimals and plain numbers."""
    text = _MONEY_RE.sub(_money, text)
    text = _PERCENT_RE.sub(lambda m: _percent(m.group(1)), text)
    text = _ORDINAL_RE.sub(lambda m: ordinal_to_words(_to_int(m.group(1))), text)
    text = _YEAR_RE.sub(lambda m: _year(m.group(1), bool(m.group(2))), text)
    text = _DECIMAL_RE.sub(lambda m: f"{number_to_words(_to_int(m.group(1)))} point {_decimal_digits(m.group(2))}", text)
    text = _NUMBER_RE.sub(lambda m: number_to_words(_to_int(m.group(0))) if len(m.group(0)) < 16 else m.group(0), text)
    return text


def _year(digits: str, plural: bool) -> str:
    words = year_to_words(int(digits))
    if not plural:
        return words
    return words[:-1] + "ies" if words.endswith("y") else words + "s"


def _percent(num: str) -> str:
    if "." in num:
        whole, frac = num.split(".", 1)
        return f"{number_to_words(_to_int(whole))} point {_decimal_digits(frac)} percent"
    return f"{number_to_words(_to_int(num))} percent"


def speak_heading(title: str) -> str:
    """'Chapter XII' -> 'Chapter twelve', 'Chapter 7: The Sea' -> 'Chapter seven: The Sea'."""
    def roman(m: re.Match) -> str:
        value = roman_to_int(m.group(2))
        return f"{m.group(1)} {number_to_words(value)}" if value else m.group(0)

    title = _HEADING_ROMAN_RE.sub(roman, title)
    bare = re.fullmatch(r"\s*([IVXLCDM]{1,8})\.?\s*", title)
    if bare and roman_to_int(bare.group(1)):
        title = f"Chapter {number_to_words(roman_to_int(bare.group(1)) or 0)}"
    return speak_numbers(title)


def expand_abbreviations(text: str) -> str:
    text = _TITLE_RE.sub(lambda m: _TITLES[m.group(1)], text)
    return _LATIN_RE.sub(lambda m: _LATIN[m.group(0).lower()], text)


def calm_shouting(text: str) -> str:
    """'He yelled STOP right NOW' keeps its emphasis words but is not spelled out letter by letter."""
    words = text.split()
    letters = [w for w in words if any(c.isalpha() for c in w)]
    if len(letters) >= 3 and all(w.upper() == w for w in letters):
        return text.lower().capitalize() if text.endswith((".", "!", "?")) else text.lower()
    return _SHOUT_RE.sub(lambda m: m.group(0).capitalize() if len(m.group(0)) <= 5 and m.group(0) not in _ACRONYMS
                         else m.group(0), text)


_ACRONYMS = {"FBI", "CIA", "NASA", "USA", "UK", "USSR", "NATO", "DNA", "CEO", "TV", "NYPD", "LAPD", "MI5", "MI6",
             "BBC", "CNN", "UFO", "AI", "GPS", "NBA", "NFL", "HIV", "AIDS", "SOS", "DJ", "VIP", "RSVP", "OK", "USB"}


def apply_pronunciations(text: str, pronunciations: dict[str, str] | None) -> str:
    """Replace whole words with respellings the voice reads correctly (``{"Hermione": "Her-my-oh-nee"}``)."""
    if not pronunciations:
        return text
    for word, respelling in sorted(pronunciations.items(), key=lambda kv: -len(kv[0])):
        if not word.strip() or not respelling.strip():
            continue
        text = re.sub(rf"(?<![\w]){re.escape(word)}(?![\w])", respelling.replace("\\", ""), text)
    return text


def speak_ready(text: str, pronunciations: dict[str, str] | None = None) -> str:
    """The full chain applied to one sentence just before it is sent to the voice."""
    text = expand_abbreviations(text)
    text = calm_shouting(text)
    text = speak_numbers(text)
    text = apply_pronunciations(text, pronunciations)
    return re.sub(r"\s+", " ", text).strip()
