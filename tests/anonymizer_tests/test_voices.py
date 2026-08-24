"""Reference (donor) voice resolution."""

import pytest

from anonymizer.voices import VoiceResolutionError, resolve_reference


@pytest.fixture
def voice_dir(tmp_path):
    """A directory with two wavs and one file that is not audio."""
    for name in ("b_voice.wav", "a_voice.wav", "notes.txt"):
        (tmp_path / name).write_bytes(b"stub")
    return tmp_path


def test_single_file_resolves_to_one_path(voice_dir):
    path = str(voice_dir / "a_voice.wav")
    assert resolve_reference(path) == [path]


def test_directory_resolves_to_sorted_audio_files(voice_dir):
    resolved = resolve_reference(str(voice_dir))
    assert [p.rsplit("/", 1)[-1] for p in resolved] == ["a_voice.wav", "b_voice.wav"]


def test_directory_ignores_non_audio(voice_dir):
    assert all(not p.endswith(".txt") for p in resolve_reference(str(voice_dir)))


def test_comma_separated_list(voice_dir):
    a, b = str(voice_dir / "a_voice.wav"), str(voice_dir / "b_voice.wav")
    assert resolve_reference(f"{a},{b}") == [a, b]


def test_comma_separated_tolerates_whitespace(voice_dir):
    a, b = str(voice_dir / "a_voice.wav"), str(voice_dir / "b_voice.wav")
    assert resolve_reference(f" {a} , {b} ") == [a, b]


def test_explicit_sequence(voice_dir):
    a, b = str(voice_dir / "a_voice.wav"), str(voice_dir / "b_voice.wav")
    assert resolve_reference([a, b]) == [a, b]


def test_missing_file_names_the_path(tmp_path):
    missing = str(tmp_path / "nope.wav")
    with pytest.raises(VoiceResolutionError, match="not found"):
        resolve_reference(missing)


def test_no_reference_and_no_pool_explains_what_to_do():
    with pytest.raises(VoiceResolutionError, match="No reference voice given"):
        resolve_reference(None)


def test_falls_back_to_voice_pool(voice_dir):
    resolved = resolve_reference(None, voice_pool_dir=str(voice_dir))
    assert len(resolved) == 2


def test_empty_voice_pool_is_an_error(tmp_path):
    (tmp_path / "readme.txt").write_text("no audio here")
    with pytest.raises(VoiceResolutionError, match="no audio files"):
        resolve_reference(None, voice_pool_dir=str(tmp_path))


def test_missing_voice_pool_dir_is_an_error(tmp_path):
    with pytest.raises(VoiceResolutionError, match="does not exist"):
        resolve_reference(None, voice_pool_dir=str(tmp_path / "absent"))


def test_empty_directory_is_an_error(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(VoiceResolutionError, match="no audio files"):
        resolve_reference(str(empty))
