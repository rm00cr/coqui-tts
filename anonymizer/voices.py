"""Resolving the donor ("reference") voice.

The anonymizer needs two inputs: the *target* (the audio being anonymized, whose words
are preserved) and the *reference* (the donor voice the output will sound like). This
module turns whatever the user passed for the reference into the ``list[str]`` of wav
paths that ``XTTS.prep_batch`` expects — it averages the conditioning across all of them.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Union

AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg", ".m4a")

ReferenceSpec = Union[str, Sequence[str], None]


class VoiceResolutionError(ValueError):
    """Raised when a reference voice cannot be resolved to usable audio files."""


def _audio_files_in(directory: str) -> List[str]:
    entries = sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.lower().endswith(AUDIO_SUFFIXES)
    )
    return entries


def resolve_reference(reference: ReferenceSpec, voice_pool_dir: Optional[str] = None) -> List[str]:
    """Resolve a reference spec to a non-empty list of audio file paths.

    Accepts a single file path, a directory (every audio file in it is used, and the
    conditioning is averaged), a comma-separated string, or an explicit sequence of
    paths. When ``reference`` is None, falls back to ``voice_pool_dir`` if one is
    configured — this is the hook for a bundled default voice pool.

    Raises:
        VoiceResolutionError: if nothing usable could be resolved.
    """
    if reference is None:
        if voice_pool_dir:
            if not os.path.isdir(voice_pool_dir):
                raise VoiceResolutionError(
                    f"voice_pool_dir does not exist: {voice_pool_dir}"
                )
            pool = _audio_files_in(voice_pool_dir)
            if not pool:
                raise VoiceResolutionError(
                    f"voice pool {voice_pool_dir} contains no audio files "
                    f"({', '.join(AUDIO_SUFFIXES)})"
                )
            return pool
        raise VoiceResolutionError(
            "No reference voice given. Pass --reference with a wav file or a directory "
            "of wavs (the donor voice the output should sound like), or set "
            "voice_pool_dir in your config."
        )

    if isinstance(reference, str):
        candidates = [part.strip() for part in reference.split(",") if part.strip()]
    else:
        candidates = [str(part) for part in reference]

    if not candidates:
        raise VoiceResolutionError("Reference voice resolved to an empty list.")

    resolved: List[str] = []
    for candidate in candidates:
        if os.path.isdir(candidate):
            files = _audio_files_in(candidate)
            if not files:
                raise VoiceResolutionError(
                    f"Reference directory {candidate} contains no audio files "
                    f"({', '.join(AUDIO_SUFFIXES)})"
                )
            resolved.extend(files)
        elif os.path.isfile(candidate):
            resolved.append(candidate)
        else:
            raise VoiceResolutionError(f"Reference voice not found: {candidate}")

    return resolved
