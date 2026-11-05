import numpy as np

from murmur.audio import trim_silence
from murmur.models import Chapter
from murmur.prosody import PAUSE, ParaNote, build_chunks, heuristic_note, style_scale


def chunks_for(paragraphs, title="Chapter 1", notes=None, **kw):
    return build_chunks(Chapter(title, paragraphs), 300, notes=notes, **kw)


def test_pauses_differ_by_punctuation():
    cs = chunks_for(["It was cold. Was it cold? It was cold! It was cold... And then, silence fell over the sea."][:1])
    texts = {c.text: c.pause_after for c in cs}
    assert texts["It was cold."] == PAUSE["sentence"]
    assert texts["Was it cold?"] == PAUSE["question"]
    assert texts["It was cold!"] == PAUSE["exclaim"]
    assert texts["It was cold..."] == PAUSE["ellipsis"]
    assert PAUSE["question"] > PAUSE["sentence"] and PAUSE["ellipsis"] > PAUSE["question"]


def test_paragraph_end_is_longer_than_a_sentence_end():
    cs = chunks_for(["The first sentence is here. The second sentence ends the paragraph.", "Another paragraph follows it."])
    body = cs[1:]
    assert body[0].pause_after == PAUSE["sentence"]
    assert body[1].pause_after == PAUSE["paragraph"]


def test_dialogue_turns_and_runs():
    cs = chunks_for(['"Come here." "Now, please." He waited by the door.', '"I will not," she said, turning away.',
                     "The kettle began to whistle in the kitchen behind them."])
    by_text = {c.text: c.pause_after for c in cs}
    assert by_text['"Come here."'] == PAUSE["dialogue_run"]  # same speaker keeps going
    last_of_first_dialogue = by_text["He waited by the door."]
    assert last_of_first_dialogue == PAUSE["dialogue_turn"]  # next paragraph is dialogue: change of speaker
    assert by_text['"I will not," she said, turning away.'] == PAUSE["paragraph"]


def test_scene_break_stretches_the_previous_pause_and_is_not_spoken():
    cs = chunks_for(["The first scene ends here, quietly.", "* * *", "The next scene begins somewhere else."])
    assert cs[1].pause_after == PAUSE["scene_break"]
    assert not any("*" in c.text for c in cs)


def test_chapter_heading_is_announced_with_a_long_pause():
    cs = chunks_for(["Some text follows the heading."], title="Chapter XII")
    assert cs[0].text == "Chapter twelve." and cs[0].pause_after == PAUSE["heading"]
    silent = build_chunks(Chapter("Opening", ["Some text follows."], announce=False), 300)
    assert silent[0].text == "Some text follows."


def test_pause_styles_scale_everything():
    base = chunks_for(["One sentence. Two sentences here."], style="natural")
    slow = chunks_for(["One sentence. Two sentences here."], style="relaxed")
    fast = chunks_for(["One sentence. Two sentences here."], style="snappy")
    assert style_scale("relaxed") > 1 > style_scale("snappy") and style_scale("nonsense") == 1.0
    assert all(s.pause_after > b.pause_after > f.pause_after for s, b, f in zip(slow, base, fast))


def test_director_notes_change_pace_pause_and_intensity():
    paras = ["He opened the door and stepped inside slowly.", "Then everything went quiet around him."]
    notes = [ParaNote(pace="slow", intensity="calm", pause="dramatic"), ParaNote(pace="fast", intensity="intense")]
    cs = chunks_for(paras, notes=notes)[1:]
    assert cs[0].speed < 1.0 < cs[1].speed
    assert cs[0].intensity < cs[1].intensity
    assert cs[0].pause_after > PAUSE["paragraph"] * 2


def test_skipped_paragraphs_are_not_read():
    paras = ["Keep this sentence in the book.", "Visit our website for more books like this one."]
    cs = chunks_for(paras, notes=[ParaNote(), ParaNote(skip=True)])
    assert [c.text for c in cs[1:]] == ["Keep this sentence in the book."]


def test_numbers_and_pronunciations_are_applied_per_sentence():
    cs = chunks_for(["Hermione paid $5 in 1995."], pronunciations={"Hermione": "her-MY-oh-nee"})
    assert cs[1].text == "her-MY-oh-nee paid five dollars in nineteen ninety-five."


def test_long_sentences_get_clause_pauses():
    sentence = ", ".join(f"clause number {i} of the very long sentence" for i in range(12)) + "."
    cs = build_chunks(Chapter("Chapter 1", [sentence]), 120)[1:]
    assert len(cs) > 3
    assert all(c.pause_after == PAUSE["clause"] for c in cs[:-1])
    assert all(len(c.text) <= 120 for c in cs)


def test_heuristic_notes():
    assert heuristic_note("* * *").kind == "scene_break"
    assert heuristic_note("THE STORM").kind == "heading"
    assert heuristic_note('"Run!" he screamed. "Run, now!"').intensity == "intense"
    assert heuristic_note("And then, nothing...").pause == "long"
    assert heuristic_note("He was gone.").pause == "long"  # a one-line beat lands
    plain = heuristic_note("A perfectly ordinary paragraph with plenty of words in it, nothing special at all.")
    assert (plain.kind, plain.pace, plain.pause, plain.intensity) == ("narration", "normal", "normal", "normal")


def test_ai_notes_are_sanitised():
    note = ParaNote.from_dict({"kind": "SHOUTING", "pace": "Slow", "pause": "forever", "intensity": 9, "skip": 1})
    assert (note.kind, note.pace, note.pause, note.intensity, note.skip) == ("narration", "slow", "normal", "normal", True)


# ---- trimming engine silence so the director's pauses are exact -----------------------------------
def test_trim_silence_removes_padding_but_keeps_the_voice():
    sr = 16000
    voice = 0.3 * np.sin(2 * np.pi * 200 * np.arange(sr // 2) / sr).astype(np.float32)
    padded = np.concatenate([np.zeros(sr), voice, np.zeros(sr)])
    trimmed = trim_silence(padded, sr)
    assert abs(len(trimmed) - (len(voice) + 2 * int(0.02 * sr))) < 0.02 * sr
    assert np.max(np.abs(trimmed)) > 0.25
    assert abs(trimmed[0]) < 1e-3 and abs(trimmed[-1]) < 1e-3  # faded: no click


def test_trim_silence_handles_silence_and_empty():
    sr = 16000
    assert len(trim_silence(np.zeros(sr * 3, dtype=np.float32), sr)) <= sr // 10
    assert trim_silence(np.zeros(0, dtype=np.float32), sr).size == 0
