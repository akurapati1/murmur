from murmur.matter import (
    apply_span,
    boilerplate_hits,
    classify_title,
    is_back_signal,
    is_boilerplate,
    locate_story,
    looks_like_fiction,
)
from murmur.models import Chapter

PROSE = ("The old lighthouse keeper walked along the shore each morning, counting the gulls and listening "
         "to the tide while the grey light came up over the water.")
DIALOGUE = '"You should come inside," said Marta from the doorway. "The tea is hot."'
COPYRIGHT = ("Copyright (c) 2020 by A. Writer. All rights reserved. No part of this book may be reproduced. "
             "Published by Tidewater Press. ISBN 978-1-23456-789-0.")


def story(n=12, dialogue=True):
    paras = []
    for i in range(n):
        paras.append(DIALOGUE if dialogue and i % 3 == 1 else PROSE)
    return paras


def test_title_classification():
    assert classify_title("Copyright") == "matter"
    assert classify_title("Praise for The Lighthouse Keeper") == "matter"
    assert classify_title("Also by A. Writer") == "matter"
    assert classify_title("About the Author") == "matter"
    assert classify_title("Acknowledgments") == "matter"
    assert classify_title("Foreword") == "optional"
    assert classify_title("Introduction") == "optional"
    assert classify_title("Prologue") == "story"
    assert classify_title("Chapter One: The Storm") == "story"
    assert classify_title("Chapter 12") == "story"
    assert classify_title("The Harbour") == "neutral"
    assert classify_title("Opening") == "neutral"


def test_boilerplate_and_back_signals():
    assert is_boilerplate(COPYRIGHT) and boilerplate_hits(COPYRIGHT) >= 3
    assert not is_boilerplate(PROSE) and not is_boilerplate(DIALOGUE)
    assert is_back_signal("About the Author")
    assert is_back_signal("Thank you for reading!")
    assert not is_back_signal(PROSE)


def test_fiction_detection():
    assert looks_like_fiction([Chapter("Chapter 1", story())])
    assert not looks_like_fiction([Chapter("Chapter 1", story(dialogue=False))])


def test_chapter_level_front_and_back_matter_is_removed():
    chapters = [
        Chapter("Cover", ["THE LIGHTHOUSE KEEPER"]),
        Chapter("Copyright", [COPYRIGHT]),
        Chapter("Dedication", ["For my mother."]),
        Chapter("Foreword", ["A foreword by somebody famous. " * 10]),
        Chapter("Prologue", story()),
        Chapter("Chapter 1", story()),
        Chapter("Epilogue", story()),
        Chapter("Acknowledgments", ["Thanks to everybody. " * 5]),
        Chapter("About the Author", ["A. Writer lives by the sea."]),
    ]
    span = locate_story(chapters)
    narrated = apply_span(chapters, span)
    assert [c.title for c in narrated] == ["Prologue", "Chapter 1", "Epilogue"]
    assert span.fiction
    assert {r["where"] for r in span.removed} == {"front", "back"}


def test_nonfiction_keeps_the_introduction():
    chapters = [
        Chapter("Copyright", [COPYRIGHT]),
        Chapter("Introduction", story(dialogue=False)),
        Chapter("Chapter 1", story(dialogue=False)),
        Chapter("Chapter 2", story(dialogue=False)),
    ]
    narrated = apply_span(chapters, locate_story(chapters))
    assert [c.title for c in narrated] == ["Introduction", "Chapter 1", "Chapter 2"]
    # ...unless the caller says otherwise
    narrated = apply_span(chapters, locate_story(chapters, keep_intro=False))
    assert [c.title for c in narrated] == ["Chapter 1", "Chapter 2"]


def test_fiction_drops_a_foreword_by_someone_else():
    chapters = [Chapter("Foreword", story()), Chapter("Chapter 1", story()), Chapter("Chapter 2", story())]
    assert [c.title for c in apply_span(chapters, locate_story(chapters))] == ["Chapter 1", "Chapter 2"]


def test_front_matter_inside_one_opening_chapter():
    opening = ["THE LIGHTHOUSE KEEPER", "A Novel", "A. WRITER", COPYRIGHT, "For my mother.",
               '"The sea does not care."', "- Old proverb", *story(8)]
    chapters = [Chapter("Opening", opening), Chapter("The Harbour", story()), Chapter("The Storm", story())]
    span = locate_story(chapters)
    narrated = apply_span(chapters, span)
    assert narrated[0].paragraphs[0] == PROSE
    assert not any("ISBN" in p or "mother" in p or "proverb" in p for c in narrated for p in c.paragraphs)
    assert narrated[0].announce is False  # never say "Opening" out loud
    assert len(narrated[0].paragraphs) == 8


def test_back_matter_inside_the_last_chapter():
    last = [*story(15), "Acknowledgments", "I wish to thank my editor.", "About the Author",
            "A. Writer lives by the sea. Visit www.example.com", "Also by A. Writer", "Another Book"]
    chapters = [Chapter("Chapter 1", story()), Chapter("Chapter 2", last)]
    narrated = apply_span(chapters, locate_story(chapters))
    assert len(narrated[-1].paragraphs) == 15
    assert narrated[-1].paragraphs[-1] in (PROSE, DIALOGUE)


def test_a_story_without_front_matter_is_left_alone():
    chapters = [Chapter("Chapter 1", story()), Chapter("Chapter 2", story())]
    span = locate_story(chapters)
    assert span.removed == []
    narrated = apply_span(chapters, span)
    assert [len(c.paragraphs) for c in narrated] == [12, 12]


def test_never_cuts_more_than_a_safe_share():
    # Eight "matter" chapters with a lot of text and one tiny story chapter: don't gut the book.
    chapters = [Chapter(t, ["Words go here. " * 400]) for t in ("Cover", "Copyright", "Dedication", "Contents")]
    chapters.append(Chapter("Chapter 1", story(6)))
    span = locate_story(chapters)
    narrated = apply_span(chapters, span)
    assert narrated  # something is always left to read
    assert span.notes
