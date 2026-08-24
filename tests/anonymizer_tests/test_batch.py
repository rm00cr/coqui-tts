"""Folder batch driving, checkpointing and resume — with a stubbed anonymizer."""

import json
import os

import numpy as np
import pytest

from anonymizer.batch import (
    CHECKPOINT_NAME,
    anonymize_directory,
    find_audio_files,
    load_checkpoint,
    save_checkpoint,
    write_manifest,
)
from anonymizer.pipeline import AnonymizationResult


class StubAnonymizer:
    """Stands in for `Anonymizer`, recording calls instead of loading 2 GB of weights."""

    def __init__(self, fail_on=()):
        self.calls = []
        self.fail_on = set(fail_on)

    def anonymize(self, target, reference=None, output_path=None, **kwargs):
        self.calls.append(target)
        if os.path.basename(target) in self.fail_on:
            raise RuntimeError("synthetic failure")
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "wb") as fh:
                fh.write(b"stub-wav")
        return AnonymizationResult(
            wav=np.zeros(16, dtype=np.float32),
            sample_rate=24000,
            text=f"transcript of {os.path.basename(target)}",
            reference=["donor.wav"],
            mode="single",
            info={"mode": "single"},
            output_path=output_path,
        )


@pytest.fixture
def input_dir(tmp_path):
    d = tmp_path / "inputs"
    d.mkdir()
    for name in ("two.wav", "one.wav", "notes.txt"):
        (d / name).write_bytes(b"stub")
    return d


class TestFindAudioFiles:
    def test_returns_sorted_audio_only(self, input_dir):
        found = [os.path.basename(p) for p in find_audio_files(str(input_dir))]
        assert found == ["one.wav", "two.wav"]

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            find_audio_files(str(tmp_path / "absent"))


class TestCheckpoint:
    def test_roundtrip(self, tmp_path):
        path = str(tmp_path / "ckpt.json")
        save_checkpoint(path, ["a.wav"], [{"input_path": "a.wav"}])
        loaded = load_checkpoint(path)
        assert loaded["completed"] == ["a.wav"]
        assert loaded["rows"] == [{"input_path": "a.wav"}]

    def test_missing_file_gives_empty_state(self, tmp_path):
        assert load_checkpoint(str(tmp_path / "absent.json")) == {"completed": [], "rows": []}

    def test_corrupt_file_does_not_abort_the_run(self, tmp_path):
        path = tmp_path / "ckpt.json"
        path.write_text("{ truncated by a crash")
        assert load_checkpoint(str(path)) == {"completed": [], "rows": []}

    def test_lock_file_is_cleaned_up(self, tmp_path):
        path = str(tmp_path / "ckpt.json")
        save_checkpoint(path, [], [])
        assert not os.path.exists(path + ".lock")


class TestManifest:
    def test_unions_columns_across_rows(self, tmp_path):
        path = str(tmp_path / "m.csv")
        write_manifest(path, [{"a": 1}, {"a": 2, "b": 3}])
        header = open(path).readline().strip()
        assert header == "a,b"

    def test_no_rows_writes_nothing(self, tmp_path):
        path = str(tmp_path / "m.csv")
        assert write_manifest(path, []) is None
        assert not os.path.exists(path)


class TestAnonymizeDirectory:
    def test_processes_every_audio_file(self, input_dir, tmp_path):
        stub = StubAnonymizer()
        summary = anonymize_directory(
            str(input_dir), str(tmp_path / "out"), reference="donor.wav", anonymizer=stub
        )
        assert summary["total"] == 2
        assert summary["completed"] == 2
        assert len(stub.calls) == 2

    def test_writes_outputs_and_manifest(self, input_dir, tmp_path):
        out = tmp_path / "out"
        summary = anonymize_directory(
            str(input_dir), str(out), reference="donor.wav", anonymizer=StubAnonymizer()
        )
        assert os.path.exists(out / "one_anonymized.wav")
        assert os.path.exists(summary["manifest"])
        assert "input_path" in open(summary["manifest"]).readline()

    def test_resume_skips_completed_files(self, input_dir, tmp_path):
        out = tmp_path / "out"
        first = StubAnonymizer()
        anonymize_directory(str(input_dir), str(out), anonymizer=first, resume=True)
        assert len(first.calls) == 2

        second = StubAnonymizer()
        summary = anonymize_directory(str(input_dir), str(out), anonymizer=second, resume=True)
        assert second.calls == []  # everything already done
        assert summary["completed"] == 2  # rows carried over from the checkpoint

    def test_without_resume_reprocesses_everything(self, input_dir, tmp_path):
        out = tmp_path / "out"
        anonymize_directory(str(input_dir), str(out), anonymizer=StubAnonymizer(), resume=True)
        second = StubAnonymizer()
        anonymize_directory(str(input_dir), str(out), anonymizer=second, resume=False)
        assert len(second.calls) == 2

    def test_one_bad_file_does_not_sink_the_batch(self, input_dir, tmp_path):
        stub = StubAnonymizer(fail_on=["one.wav"])
        summary = anonymize_directory(str(input_dir), str(tmp_path / "out"), anonymizer=stub)
        assert summary["completed"] == 1
        assert len(summary["failures"]) == 1
        assert "synthetic failure" in summary["failures"][0]["error"]

    def test_checkpoint_written_next_to_outputs(self, input_dir, tmp_path):
        out = tmp_path / "out"
        anonymize_directory(str(input_dir), str(out), anonymizer=StubAnonymizer())
        checkpoint = json.load(open(out / CHECKPOINT_NAME))
        assert sorted(checkpoint["completed"]) == ["one.wav", "two.wav"]

    def test_progress_callback_reports_position(self, input_dir, tmp_path):
        seen = []
        anonymize_directory(
            str(input_dir),
            str(tmp_path / "out"),
            anonymizer=StubAnonymizer(),
            on_progress=lambda path, i, total: seen.append((i, total)),
        )
        assert seen == [(1, 2), (2, 2)]
