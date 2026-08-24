"""Guards on existing behaviour.

The anonymizer package is a façade over code the research scripts and cluster runs also
use. These tests fail loudly if that shared code drifts.
"""

import inspect
import os

import pytest


class TestImportSideEffects:
    def test_importing_xtts_does_not_pull_the_scoring_stack(self):
        """Importing the model must not drag in jiwer/sacrebleu or build a BLEU scorer."""
        import subprocess
        import sys

        code = (
            "import sys, TTS.tts.models.xtts as x;"
            "print(('jiwer' in sys.modules, 'sacrebleu' in sys.modules, x._bleu_scorer is None))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=900
        )
        assert out.returncode == 0, out.stderr
        assert "(False, False, True)" in out.stdout

    def test_constructing_a_session_loads_nothing(self):
        """A session must be free to build; the 2 GB load happens on first property use."""
        from anonymizer.session import AnonymizerSession

        session = AnonymizerSession()
        assert session._model is None
        assert session._asr is None
        assert session._ecapa is None


class TestSharedSignatures:
    """The run scripts call these positionally; changing them breaks the cluster runs."""

    def test_forward_from_audios_and_text_signature(self):
        from TTS.tts.models.xtts import Xtts

        params = list(inspect.signature(Xtts.forward_from_audios_and_text).parameters)
        assert params == [
            "self", "lang", "text", "target_sample", "ref_sample",
            "train_model", "max_conditioning_length", "min_conditioning_length",
        ]

    def test_forward_iteration_keeps_its_original_parameters(self):
        """output_dir was added; everything before it must stay put and keep its default."""
        from TTS.tts.models.xtts import Xtts

        sig = inspect.signature(Xtts.forward_iteration)
        params = list(sig.parameters)
        assert params[:11] == [
            "self", "lang", "text", "target_sample", "ref_sample", "train_model",
            "max_conditioning_length", "min_conditioning_length", "tts", "ecapa", "asr_model",
        ]
        assert sig.parameters["n"].default == 5
        assert sig.parameters["output_dir"].default is None

    def test_forward_iteration_hdf5_signature_unchanged(self):
        from TTS.tts.models.xtts import Xtts

        params = list(inspect.signature(Xtts.forward_iteration_hdf5).parameters)
        assert "target_hdf5_path" in params and "ref_hdf5_path" in params

    def test_iterative_segment_refinement_accepts_its_original_call(self):
        """run_test.py passes these as keywords; device was appended with a default."""
        from development.utils import iterative_segment_refinement

        sig = inspect.signature(iterative_segment_refinement)
        for name in (
            "model", "target_speaker", "ref_speaker", "max_conditioning_length",
            "min_conditioning_length", "train_model", "asr_model", "ecapa_model",
            "lang", "threshold", "max_attempts",
        ):
            assert name in sig.parameters
        assert sig.parameters["device"].default is None


class TestBackwardsCompatibleShims:
    def test_model_conf_still_exports_its_old_names(self):
        """Notebooks and run scripts do `from model_conf import ModelPaths, ...`."""
        import model_conf

        assert hasattr(model_conf, "ModelPaths")
        assert hasattr(model_conf, "load_tts_and_trainer")
        assert hasattr(model_conf, "build_gpt_config")
        assert hasattr(model_conf, "get_device")

    def test_model_paths_defaults_to_the_configured_directory(self, monkeypatch):
        monkeypatch.setenv("XTTS_MODEL_DIR", "/tmp/model-dir-under-test")
        import importlib

        import anonymizer.model_setup as model_setup

        importlib.reload(model_setup)
        assert model_setup.ModelPaths().checkpoints_out_path == "/tmp/model-dir-under-test"
        importlib.reload(model_setup)  # restore for other tests

    def test_temp_files_do_not_land_in_the_working_directory(self):
        from development.utils import get_unique_temp_filename

        assert os.path.isabs(get_unique_temp_filename())


class TestDownloadTargets:
    def test_every_required_checkpoint_file_has_a_url(self):
        from anonymizer.download import CHECKPOINT_FILES
        from anonymizer.session import REQUIRED_CHECKPOINT_FILES

        assert set(REQUIRED_CHECKPOINT_FILES) == set(CHECKPOINT_FILES)

    def test_missing_files_reports_all_when_directory_is_empty(self, tmp_path):
        from anonymizer.download import CHECKPOINT_FILES, missing_files

        assert set(missing_files(str(tmp_path))) == set(CHECKPOINT_FILES)


class TestModelFileChecking:
    def test_absent_directory_explains_how_to_fix_it(self, tmp_path):
        from anonymizer.config import AnonymizerConfig
        from anonymizer.session import AnonymizerSession, ModelFilesMissing

        session = AnonymizerSession(AnonymizerConfig(model_dir=str(tmp_path / "absent")))
        with pytest.raises(ModelFilesMissing, match="download-model"):
            session.check_model_files()

    def test_incomplete_directory_names_the_missing_files(self, tmp_path):
        from anonymizer.config import AnonymizerConfig
        from anonymizer.session import AnonymizerSession, ModelFilesMissing

        (tmp_path / "config.json").write_text("{}")
        session = AnonymizerSession(AnonymizerConfig(model_dir=str(tmp_path)))
        with pytest.raises(ModelFilesMissing, match="model.pth"):
            session.check_model_files()
