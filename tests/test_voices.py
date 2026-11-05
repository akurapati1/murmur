import numpy as np
import pytest

from murmur.audio import ffmpeg_available
from murmur.voices import (
    SAMPLE_RATE,
    VoiceError,
    VoiceLibrary,
    clean_name,
    effective_voice,
    get_default_voice,
    is_clone_id,
    prepare_reference,
    set_default_voice,
)
from tests.speechwav import speechlike, write_wav

pytestmark = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")


@pytest.fixture
def library(settings):
    return VoiceLibrary(settings)


@pytest.fixture
def recording(tmp_path):
    return write_wav(tmp_path / "me.wav", speechlike(14))


def test_upload_name_and_list(library, recording):
    voice = library.add("My voice", recording, source_name="me.wav", consent=True)
    assert is_clone_id(voice.id) and voice.name == "My voice"
    assert 5 < voice.duration_sec <= 28
    assert [v.name for v in library.list()] == ["My voice"]
    assert library.get(voice.id).name == "My voice"
    assert library.get("my VOICE").id == voice.id  # lookups by name ignore case
    assert library.reference_path(voice.id).exists()


def test_reference_clip_is_cleaned(library, recording):
    voice = library.add("Clean", recording, consent=True)
    import wave

    with wave.open(str(library.reference_path(voice.id))) as w:
        assert (w.getnchannels(), w.getframerate(), w.getsampwidth()) == (1, SAMPLE_RATE, 2)
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2") / 32768.0
    assert 0.8 < np.max(np.abs(pcm)) <= 0.95  # normalised level


def test_other_audio_formats_work(library, tmp_path):
    import subprocess

    wav = write_wav(tmp_path / "src.wav", speechlike(12))
    mp3 = tmp_path / "voice memo.mp3"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(wav), str(mp3)], check=True)
    assert library.add("From mp3", mp3, consent=True).duration_sec > 5


def test_consent_is_required(library, recording):
    with pytest.raises(VoiceError, match="your own voice"):
        library.add("Me", recording, consent=False)
    assert library.list() == []


def test_names_are_validated_and_unique(library, recording):
    with pytest.raises(VoiceError):
        library.add("   ", recording, consent=True)
    with pytest.raises(VoiceError):
        library.add("x" * 41, recording, consent=True)
    library.add("Narrator", recording, consent=True)
    with pytest.raises(VoiceError, match="already have"):
        library.add("narrator", recording, consent=True)
    assert clean_name("  Two   words ") == "Two words"


def test_unusable_recordings_are_explained(library, tmp_path):
    with pytest.raises(VoiceError, match="too short"):
        library.add("Short", write_wav(tmp_path / "s.wav", speechlike(0.5)), consent=True)
    with pytest.raises(VoiceError, match="silent"):
        library.add("Silent", write_wav(tmp_path / "q.wav", np.zeros(22050 * 12, dtype=np.float32)), consent=True)
    with pytest.raises(VoiceError, match="seconds of speech"):
        library.add("Sparse", write_wav(tmp_path / "p.wav", speechlike(12, loud=0.35) * 0
                    + np.pad(speechlike(2), (0, 22050 * 10))), consent=True)
    junk = tmp_path / "notes.mp3"
    junk.write_text("this is not audio")
    with pytest.raises(VoiceError, match="couldn't read"):
        library.add("Junk", junk, consent=True)
    assert library.list() == []  # failed uploads leave nothing behind
    assert not any(library.root.iterdir())


def test_long_recordings_are_cut_to_the_best_stretch():
    clip, warnings = prepare_reference(speechlike(120, sr=SAMPLE_RATE))
    assert len(clip) / SAMPLE_RATE <= 28.5
    assert not any("seconds of speech were usable" in w for w in warnings)


def test_noise_and_distortion_are_flagged():
    noisy, w1 = prepare_reference(speechlike(15, sr=SAMPLE_RATE, noise=0.02, loud=0.35))
    assert any("background noise" in w for w in w1)
    _, w2 = prepare_reference(np.clip(speechlike(15, sr=SAMPLE_RATE) * 12, -1, 1))
    assert any("distorted" in w for w in w2)


def test_drowned_out_voice_is_explained():
    with pytest.raises(VoiceError, match="background noise"):
        prepare_reference(speechlike(15, sr=SAMPLE_RATE, noise=0.2, loud=0.2))


def test_rename_and_delete(library, recording):
    a = library.add("One", recording, consent=True)
    b = library.add("Two", recording, consent=True)
    assert library.rename(a.id, "Uno").name == "Uno"
    with pytest.raises(VoiceError):
        library.rename(a.id, "two")
    set_default_voice(library.settings, a.id)
    assert get_default_voice(library.settings) == a.id
    assert library.delete(a.id) and not library.delete(a.id)
    assert get_default_voice(library.settings) is None  # a deleted default is forgotten
    assert [v.id for v in library.list()] == [b.id]


def test_effective_voice_prefers_request_then_default(library, recording, settings):
    me = library.add("Abhilash", recording, consent=True)
    assert effective_voice(settings, None) is None
    assert effective_voice(settings, "af_heart") == "af_heart"
    assert effective_voice(settings, "Abhilash") == me.id  # a name is as good as an id
    assert effective_voice(settings, "abhilash") == me.id
    set_default_voice(settings, me.id)
    assert effective_voice(settings, None) == me.id  # the saved default applies when nothing is chosen
    assert effective_voice(settings, "af_heart") == "af_heart"
