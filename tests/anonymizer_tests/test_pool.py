"""Loading a donor voice pool from a directory tree or a CSV manifest."""

import os

import pytest

from anonymizer.voices import VoicePool, VoiceResolutionError, normalize_gender


@pytest.fixture
def pool_dir(tmp_path):
    """pool/<speaker>/*.wav, plus one loose file and one non-audio file."""
    root = tmp_path / "pool"
    root.mkdir()
    for speaker, names in (("spk_a", ("one.wav", "two.wav")), ("spk_b", ("one.flac",))):
        folder = root / speaker
        folder.mkdir()
        for name in names:
            (folder / name).write_bytes(b"stub")
    (root / "loose.wav").write_bytes(b"stub")
    (root / "spk_a" / "notes.txt").write_text("not audio")
    return root


def write_csv(path, rows, header):
    path.write_text("\n".join([",".join(header)] + [",".join(r) for r in rows]) + "\n")
    return str(path)


class TestDirectoryPool:
    def test_subdirectory_names_become_speaker_ids(self, pool_dir):
        assert set(VoicePool.from_directory(str(pool_dir)).speakers) == {"spk_a", "spk_b", "loose"}

    def test_clips_are_grouped_under_their_speaker(self, pool_dir):
        assert len(VoicePool.from_directory(str(pool_dir)).clips_for("spk_a")) == 2

    def test_non_audio_files_are_ignored(self, pool_dir):
        assert all(not c.path.endswith(".txt") for c in VoicePool.from_directory(str(pool_dir)))

    def test_a_loose_file_is_its_own_speaker(self, pool_dir):
        clips = VoicePool.from_directory(str(pool_dir)).clips_for("loose")
        assert [os.path.basename(c.path) for c in clips] == ["loose.wav"]

    def test_missing_directory_is_an_error(self, tmp_path):
        with pytest.raises(VoiceResolutionError, match="does not exist"):
            VoicePool.from_directory(str(tmp_path / "absent"))

    def test_directory_without_audio_is_an_error(self, tmp_path):
        (tmp_path / "readme.md").write_text("nothing here")
        with pytest.raises(VoiceResolutionError, match="no audio files"):
            VoicePool.from_directory(str(tmp_path))


class TestCsvPool:
    def test_reads_paths_speakers_and_metadata(self, tmp_path):
        (tmp_path / "a.wav").write_bytes(b"stub")
        csv_path = write_csv(
            tmp_path / "pool.csv", [["a.wav", "3124", "female", "de"]],
            ["path", "speaker_id", "gender", "language"],
        )
        clip = VoicePool.from_csv(csv_path).clips[0]
        assert (clip.speaker_id, clip.gender, clip.language) == ("3124", "female", "de")

    def test_relative_paths_resolve_against_the_csv(self, tmp_path):
        (tmp_path / "a.wav").write_bytes(b"stub")
        csv_path = write_csv(tmp_path / "pool.csv", [["a.wav", "s1"]], ["path", "speaker_id"])
        assert VoicePool.from_csv(csv_path).clips[0].path == str(tmp_path / "a.wav")

    def test_audio_root_overrides_where_relative_paths_point(self, tmp_path):
        audio = tmp_path / "audio"
        audio.mkdir()
        (audio / "a.wav").write_bytes(b"stub")
        csv_path = write_csv(tmp_path / "pool.csv", [["a.wav", "s1"]], ["path", "speaker_id"])
        pool = VoicePool.from_csv(csv_path, audio_root=str(audio))
        assert pool.clips[0].path == str(audio / "a.wav")

    def test_column_aliases_are_accepted(self, tmp_path):
        """The cluster manifests name these columns file_path/predicted_gender/lang."""
        (tmp_path / "a.wav").write_bytes(b"stub")
        csv_path = write_csv(
            tmp_path / "pool.csv", [["a.wav", "3124", "f", "DE"]],
            ["file_path", "speaker", "predicted_gender", "lang"],
        )
        clip = VoicePool.from_csv(csv_path).clips[0]
        assert (clip.speaker_id, clip.gender, clip.language) == ("3124", "female", "de")

    def test_missing_path_column_names_what_it_looked_for(self, tmp_path):
        csv_path = write_csv(tmp_path / "pool.csv", [["3124"]], ["speaker_id"])
        with pytest.raises(VoiceResolutionError, match="no audio-path column"):
            VoicePool.from_csv(csv_path)

    def test_rows_pointing_at_absent_files_are_skipped(self, tmp_path, capsys):
        (tmp_path / "a.wav").write_bytes(b"stub")
        csv_path = write_csv(
            tmp_path / "pool.csv", [["a.wav", "s1"], ["gone.wav", "s2"]], ["path", "speaker_id"]
        )
        pool = VoicePool.from_csv(csv_path)
        assert len(pool) == 1
        assert "do not exist" in capsys.readouterr().out

    def test_a_csv_whose_audio_is_all_missing_is_an_error(self, tmp_path):
        csv_path = write_csv(tmp_path / "pool.csv", [["gone.wav", "s1"]], ["path", "speaker_id"])
        with pytest.raises(VoiceResolutionError, match="no usable audio"):
            VoicePool.from_csv(csv_path)

    def test_empty_csv_is_an_error(self, tmp_path):
        (tmp_path / "pool.csv").write_text("path,speaker_id\n")
        with pytest.raises(VoiceResolutionError, match="no rows"):
            VoicePool.from_csv(str(tmp_path / "pool.csv"))

    def test_speaker_falls_back_to_the_parent_folder(self, tmp_path):
        folder = tmp_path / "spk9"
        folder.mkdir()
        (folder / "a.wav").write_bytes(b"stub")
        csv_path = write_csv(tmp_path / "pool.csv", [["spk9/a.wav"]], ["path"])
        assert VoicePool.from_csv(csv_path).clips[0].speaker_id == "spk9"


class TestPoolFiltering:
    def test_load_dispatches_on_what_the_path_is(self, tmp_path, pool_dir):
        (tmp_path / "a.wav").write_bytes(b"stub")
        csv_path = write_csv(tmp_path / "pool.csv", [["a.wav", "s1"]], ["path", "speaker_id"])
        assert len(VoicePool.load(csv_path)) == 1
        assert len(VoicePool.load(str(pool_dir))) == 4

    def test_load_explains_a_path_that_is_neither(self, tmp_path):
        with pytest.raises(VoiceResolutionError, match="voice pool not found"):
            VoicePool.load(str(tmp_path / "nope"))

    def test_filtering_on_a_field_the_pool_lacks_drops_everything(self, pool_dir):
        """A directory pool records no gender, so it cannot honour a gender filter."""
        assert len(VoicePool.from_directory(str(pool_dir)).filter(gender="female")) == 0

    def test_gender_normalization(self):
        assert [normalize_gender(v) for v in ("M", "female", "", None, "nan")] == [
            "male", "female", None, None, None,
        ]
