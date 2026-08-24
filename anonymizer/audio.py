"""Audio I/O helpers for the anonymizer.

Thin wrappers over soundfile/librosa plus the denoiser that already lives in
``development.utils``, so the package has one place to load, save and normalize audio.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Union

import numpy as np

#: The XTTS hifigan decoder emits 24 kHz.
MODEL_SAMPLE_RATE = 24000

AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg", ".m4a")


def to_numpy(wav) -> np.ndarray:
    """Normalize a model output (tensor, list, or array) to a 1-D float32 array.

    ``iterative_segment_refinement`` returns a torch.Tensor when regenerate=False and a
    numpy array when regenerate=True; the model's forward methods return arrays. This
    collapses all of those to one type so callers do not have to care.
    """
    if hasattr(wav, "detach"):  # torch.Tensor
        wav = wav.detach().cpu().numpy()
    wav = np.asarray(wav, dtype=np.float32)
    return np.squeeze(wav)


def save_wav(wav, path: str, sample_rate: int = MODEL_SAMPLE_RATE) -> str:
    """Write audio to `path`, resampling if the requested rate differs from the model's."""
    import soundfile as sf

    wav = to_numpy(wav)
    if sample_rate != MODEL_SAMPLE_RATE:
        import librosa

        wav = librosa.resample(wav, orig_sr=MODEL_SAMPLE_RATE, target_sr=sample_rate)

    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    sf.write(path, wav, sample_rate)
    return path


def load_wav(path: str, sample_rate: Union[int, None] = None):
    """Load an audio file as (samples, sample_rate), optionally resampled."""
    import librosa

    wav, sr = librosa.load(path, sr=sample_rate, mono=True)
    return wav.astype(np.float32), sr


def duration_seconds(path: str) -> float:
    import librosa

    return float(librosa.get_duration(path=path))


@contextmanager
def maybe_denoised(path: str, enabled: bool):
    """Yield a denoised copy of `path` when enabled, otherwise `path` itself.

    Reuses ``development.utils.denoised_temp_file``, which trims silence, runs
    non-stationary noise reduction and peak-normalizes, cleaning up after itself.
    """
    if not enabled:
        yield path
        return

    from development.utils import denoised_temp_file

    with denoised_temp_file(path) as denoised_path:
        yield denoised_path
