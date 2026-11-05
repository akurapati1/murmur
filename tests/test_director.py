import json
import threading

import pytest

from murmur.ai import AIError, parse_json_object
from murmur.director import ANNOTATE_SYSTEM, LOCATE_SYSTEM, DirectorOptions, Script, direct_book
from murmur.extract import Cancelled
from murmur.models import Book, Chapter

PROSE = ("The old lighthouse keeper walked along the shore each morning, counting the gulls and listening "
         "to the tide while the grey light came up over the water.")
DIALOGUE = '"You should come inside," said Marta from the doorway. "The tea is hot."'
COPYRIGHT = "Copyright (c) 2020 by A. Writer. All rights reserved. Published by Tidewater Press. ISBN 978-1-23456-789-0."


def story(n=10):
    return [DIALOGUE if i % 3 == 1 else PROSE for i in range(n)]


def make_book():
    return Book("Test", "Author", [
        Chapter("Opening", ["TEST", "A NOVEL", COPYRIGHT, "For my mother."]),
        Chapter("Prologue", story()),
        Chapter("Chapter 1", story()),
        Chapter("Acknowledgments", ["Thanks to my editor, my agent and my family for all of their help."]),
    ])


class FakeAI:
    """A scripted stand-in for a real model: no network, fully deterministic."""

    name, model = "fake", "fake-1"

    def __init__(self, locate=None, annotate=None, fail=False):
        self.locate, self.annotate, self.fail = locate, annotate, fail
        self.calls = {"locate": 0, "annotate": 0}
        self.lock = threading.Lock()

    def complete(self, system, user, max_tokens=4096):
        kind = "locate" if system == LOCATE_SYSTEM else "annotate"
        assert system in (LOCATE_SYSTEM, ANNOTATE_SYSTEM)
        with self.lock:
            self.calls[kind] += 1
        if self.fail:
            raise AIError("boom")
        reply = self.locate if kind == "locate" else self.annotate
        return reply(user) if callable(reply) else json.dumps(reply if reply is not None else {})


def run(book, settings, provider=None, **kw):
    return direct_book(book, settings, DirectorOptions(), provider=provider, **kw)


def titles(script):
    return [c.title for c in script.chapters]


def test_rules_alone_cut_front_and_back_matter(settings):
    script = run(make_book(), settings)
    assert titles(script) == ["Prologue", "Chapter 1"]
    r = script.report
    assert r["provider"] == "rules" and r["story_found_by"] == "rules"
    assert r["story_starts_with"]["chapter"] == "Prologue"
    assert {x["where"] for x in r["removed"]} == {"front", "back"} and r["removed_words"] > 0
    assert script.chapters[0].chunks[0].text == "Prologue."


def test_start_at_story_can_be_switched_off(settings):
    script = direct_book(make_book(), settings, DirectorOptions(start_at_story=False))
    assert titles(script)[0] == "Opening" and titles(script)[-1] == "Acknowledgments"
    assert script.chapters[0].chunks[0].text != "Opening."  # a synthetic title is never spoken


def test_ai_decides_where_the_story_starts(settings):
    # The AI says the story starts at the second paragraph of "Prologue" and ends mid "Chapter 1".
    ai = FakeAI(locate={"fiction": True, "story_start": {"chapter": 1, "paragraph": 2},
                        "story_end": {"chapter": 2, "paragraph": 7}, "reasons": ["Prologue begins after a recap."]})
    script = run(make_book(), settings, ai)
    assert script.report["story_found_by"] == "ai" and script.report["reasons"] == ["Prologue begins after a recap."]
    assert ai.calls["locate"] == 1
    assert titles(script) == ["Prologue", "Chapter 1"]
    # Prologue lost 2 paragraphs at the start, Chapter 1 lost 2 at the end
    spoken = [c.text for c in script.chapters[0].chunks[1:]]
    assert len(spoken) > 0 and sum("lighthouse" in t for t in spoken) >= 5


def test_a_wild_ai_answer_is_ignored(settings):
    book = Book("T", "A", [Chapter(f"Chapter {i}", story(10)) for i in range(1, 6)])
    ai = FakeAI(locate={"story_start": {"chapter": 3, "paragraph": 0}, "story_end": {"chapter": 4, "paragraph": 9}})
    script = run(book, settings, ai)
    assert script.report["story_found_by"] == "rules" and len(script.chapters) == 5  # nothing was cut
    assert any("looked wrong" in n for n in script.report["notes"])


@pytest.mark.parametrize("reply", ["not json at all", "{}", '{"story_start": {"chapter": 99, "paragraph": 0}}'])
def test_garbage_ai_answers_fall_back_to_rules(settings, reply):
    ai = FakeAI(locate=lambda _: reply, annotate=lambda _: reply)
    script = run(make_book(), settings, ai)
    assert titles(script) == ["Prologue", "Chapter 1"]
    assert script.report["story_found_by"] == "rules"


def test_a_failing_ai_never_fails_the_book(settings):
    ai = FakeAI(fail=True)
    script = run(make_book(), settings, ai)
    assert titles(script) == ["Prologue", "Chapter 1"]
    assert script.report["ai_failures"] >= 1 and any("unavailable" in n for n in script.report["notes"])


def test_ai_annotations_shape_the_narration(settings):
    def annotate(user):
        return json.dumps({
            "paragraphs": [{"i": 0, "pace": "slow", "pause": "dramatic", "intensity": "calm"}, {"i": 1, "skip": True}],
            "pronunciations": {"Marta": "MAR-tuh", "lighthouse": "x", "Bad": "<script>"},
        })

    script = run(make_book(), settings, FakeAI(locate={}, annotate=annotate))
    first = script.chapters[0].chunks[1]  # chunk 0 is the heading
    assert first.speed < 1.0 and first.intensity < 0.5 and first.pause_after > 1.5
    spoken = " ".join(c.text for ch in script.chapters for c in ch.chunks)
    assert "MAR-tuh" in spoken and "Marta" not in spoken
    assert "<script>" not in spoken
    assert script.report["pronunciations"] == {"Marta": "MAR-tuh"}  # invalid respellings are rejected
    # paragraph 1 of each chapter was skipped, so 2 of the 6 dialogue lines never reach the narrator
    assert spoken.count("tea is hot") == 4


def test_ai_answers_are_cached_so_a_resume_costs_nothing(settings, tmp_path):
    cache = tmp_path / "ai_cache.json"
    annotate = {"paragraphs": [{"i": 0, "pace": "slow"}]}
    first_ai = FakeAI(locate={}, annotate=annotate)
    first = run(make_book(), settings, first_ai, cache_path=cache)
    assert first_ai.calls["annotate"] >= 2

    second_ai = FakeAI(locate={}, annotate=annotate)
    second = run(make_book(), settings, second_ai, cache_path=cache)
    assert second_ai.calls == {"locate": 0, "annotate": 0}
    assert second.report["ai_cached"] >= 3
    assert second.to_dict()["chapters"] == first.to_dict()["chapters"]


def test_ai_that_annotates_everything_is_distrusted(settings):
    every = {"paragraphs": [{"i": i, "pause": "dramatic"} for i in range(40)]}
    script = run(make_book(), settings, FakeAI(locate={}, annotate=every))
    assert script.report["ai_failures"] >= 1
    assert max(c.pause_after for ch in script.chapters for c in ch.chunks) < 2.5


def test_ai_cannot_make_the_narrator_skip_a_third_of_the_book(settings):
    skip_all = {"paragraphs": [{"i": i, "skip": True} for i in range(4)]}
    script = run(make_book(), settings, FakeAI(locate={}, annotate=skip_all))
    assert sum(len(c.chunks) for c in script.chapters) > 20


def test_ai_off_never_calls_the_provider(settings):
    ai = FakeAI(locate={}, annotate={})
    direct_book(make_book(), settings, DirectorOptions(use_ai=False), provider=ai)
    assert ai.calls == {"locate": 0, "annotate": 0}


def test_cancel_stops_directing(settings):
    token = threading.Event()
    token.set()
    with pytest.raises(Cancelled):
        run(make_book(), settings, FakeAI(locate={}, annotate={}), cancel=token)


def test_script_round_trips_through_json(settings):
    script = run(make_book(), settings)
    again = Script.from_dict(json.loads(json.dumps(script.to_dict())))
    assert again.to_dict() == script.to_dict() and again.total_chars > 100


def test_parse_json_object_is_tolerant():
    assert parse_json_object('Sure! Here you go:\n```json\n{"a": 1}\n```\nHope that helps') == {"a": 1}
    assert parse_json_object('prefix {"a": {"b": [1, 2]}} suffix') == {"a": {"b": [1, 2]}}
    with pytest.raises(AIError):
        parse_json_object("no braces here")
    with pytest.raises(AIError):
        parse_json_object("[1, 2, 3]")
