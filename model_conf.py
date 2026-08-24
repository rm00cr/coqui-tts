"""Backwards-compatible shim.

The model setup moved to ``anonymizer/model_setup.py`` so that it ships with the
installed package instead of living as a loose module at the repo root. Existing code
that does ``from model_conf import ModelPaths, load_tts_and_trainer`` keeps working.
"""

from anonymizer.model_setup import (  # noqa: F401
    DEFAULT_CHECKPOINTS_DIR,
    ModelPaths,
    build_gpt_config,
    get_device,
    load_tts_and_trainer,
)

__all__ = [
    "DEFAULT_CHECKPOINTS_DIR",
    "ModelPaths",
    "build_gpt_config",
    "get_device",
    "load_tts_and_trainer",
]
