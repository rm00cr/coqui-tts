"""End-to-end anonymization against the real checkpoints.

Skipped unless the XTTS weights are present, so the rest of the suite stays runnable
in CI. Run with:

    XTTS_MODEL_DIR=./XTTS_v2.0_original_model_files pytest tests/anonymizer_tests/test_end_to_end.py -v
"""

import os

import numpy as np
import pytest

from anonymizer.download import missing_files

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.getenv("XTTS_MODEL_DIR", os.path.join(REPO_ROOT, "XTTS_v2.0_original_model_files"))
TARGET_WAV = os.path.join(REPO_ROOT, "data", "TARGET_5s.wav")
REFERENCE_WAV = os.path.join(REPO_ROOT, "data", "REF.wav")

pytestmark = [
    pytest.mark.skipif(
        bool(missing_files(MODEL_DIR)),
        reason=f"XTTS checkpoints not available in {MODEL_DIR}",
    ),
    pytest.mark.skipif(
        not (os.path.isfile(TARGET_WAV) and os.path.isfile(REFERENCE_WAV)),
        reason="sample audio not available",
    ),
]


@pytest.fixture(scope="module")
def anonymizer():
    """One instance for the whole module — the checkpoints load once."""
    from anonymizer import Anonymizer

    return Anonymizer(model_dir=MODEL_DIR, mode="single")


def test_single_mode_produces_speech(anonymizer, tmp_path):
    out = tmp_path / "anonymized.wav"
    result = anonymizer.anonymize(TARGET_WAV, reference=REFERENCE_WAV, output_path=str(out))

    assert result.wav.ndim == 1
    assert result.sample_rate == 24000
    assert np.isfinite(result.wav).all(), "output contains NaN or inf"
    assert np.abs(result.wav).max() > 0.01, "output is silent"
    assert len(result.wav) > 24000, "output is shorter than a second"
    assert result.text.strip(), "no transcript was produced"
    assert out.exists() and out.stat().st_size > 1000


def test_output_is_readable_at_the_configured_rate(anonymizer, tmp_path):
    import soundfile as sf

    out = tmp_path / "anonymized.wav"
    anonymizer.anonymize(TARGET_WAV, reference=REFERENCE_WAV, output_path=str(out))
    wav, sr = sf.read(str(out))
    assert sr == 24000
    assert len(wav) > 0


def test_supplying_text_skips_transcription(anonymizer):
    result = anonymizer.anonymize(TARGET_WAV, reference=REFERENCE_WAV, text="hello world")
    assert result.text == "hello world"


def test_reference_directory_is_accepted(anonymizer):
    """A directory of donor wavs averages the conditioning across them."""
    result = anonymizer.anonymize(TARGET_WAV, reference=os.path.dirname(REFERENCE_WAV))
    assert len(result.reference) > 1
    assert np.abs(result.wav).max() > 0.01


def test_missing_target_is_reported_clearly(anonymizer):
    with pytest.raises(FileNotFoundError, match="target audio not found"):
        anonymizer.anonymize("/nonexistent/audio.wav", reference=REFERENCE_WAV)


@pytest.mark.slow
def test_scoring_moves_identity_away_from_the_original(anonymizer):
    """The point of the whole pipeline: sound less like the speaker, more like the donor."""
    result = anonymizer.anonymize(TARGET_WAV, reference=REFERENCE_WAV)
    report = anonymizer.score(result, TARGET_WAV)

    assert 0.0 <= report.wer
    assert -1.0 <= report.target_similarity <= 1.0
    assert -1.0 <= report.reference_similarity <= 1.0
    assert report.reference_similarity > report.target_similarity, (
        f"output resembles the original speaker ({report.target_similarity:.3f}) more than "
        f"the donor ({report.reference_similarity:.3f}) — anonymization did not take"
    )
