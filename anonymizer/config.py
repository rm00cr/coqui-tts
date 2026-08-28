"""Configuration for the speaker anonymizer.

One dataclass holds every knob. Values resolve with the precedence:

    explicit argument  >  --config YAML file  >  environment variable  >  default
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Optional

#: Anonymization strategies, cheapest first. See `anonymizer.modes` for what each runs.
MODES = ("single", "refine", "iterate")

#: Donor-choosing strategies. See `anonymizer.selection`.
SELECTION_STRATEGIES = ("most_distant", "none")

#: Environment variable fallbacks, applied when a field is not set explicitly or in YAML.
_ENV_VARS = {
    "model_dir": "XTTS_MODEL_DIR",
    "device": "ANONYMIZER_DEVICE",
    "mode": "ANONYMIZER_MODE",
    "whisper_model": "ANONYMIZER_WHISPER_MODEL",
    "voice_pool": "ANONYMIZER_VOICE_POOL",
}

#: Applied after the environment, for fields that would otherwise need a non-None default.
#: Keeping their declared default as None is what lets `_ENV_VARS` reach them at all —
#: the resolver can only fill a field it can tell was left unset.
_FALLBACKS = {
    "mode": "single",
    "whisper_model": "base",
    "selection": "most_distant",
}


@dataclass
class AnonymizerConfig:
    """Settings for an anonymization run.

    Args:
        model_dir: directory holding the XTTS v2 checkpoint files (model.pth, dvae.pth,
            mel_stats.pth, vocab.json, config.json). See `anonymizer.download`.
        device: "cuda", "cpu", or None to auto-detect.
        mode: one of `MODES`. "single" is one fast pass; "refine" rescores and
            regenerates weak segments; "iterate" runs n whole-utterance passes and keeps
            the best-scoring one.
        language: two-letter language code of the input speech.
        whisper_model: Whisper size used to transcribe when no text is supplied.
        iterations: number of passes for mode="iterate".
        threshold: segment quality below which mode="refine" regenerates a segment.
        max_attempts: regeneration attempts per weak segment in mode="refine".
        denoise: run noise reduction on the target before anonymizing.
        output_sample_rate: sample rate of the written wav. The model emits 24 kHz.
        reference: default donor voice. A wav path, a directory of wavs, or a
            comma-separated list. Required unless supplied per call.
        voice_pool: pool of donor voices to choose from when no reference is given —
            a directory (``pool/<speaker>/*.wav``) or a CSV manifest with an audio-path
            column and, ideally, speaker_id/gender/language columns. One donor is picked
            per target; see `selection`. Env: ANONYMIZER_VOICE_POOL.
        selection: how to pick a donor out of `voice_pool`. "most_distant" (the default)
            chooses the pool speaker whose voice is furthest from the target's, which is
            what makes the anonymization strong. "none" falls back to averaging the whole
            pool, as before.
        select_top_k: how many of the chosen speaker's clips to condition on.
        voice_pool_dir: legacy — a flat directory of donor voices, all of which are
            averaged together with no selection. Prefer `voice_pool`.
    """

    model_dir: Optional[str] = None
    device: Optional[str] = None
    mode: Optional[str] = None          # -> "single"; see _FALLBACKS
    language: str = "en"
    whisper_model: Optional[str] = None  # -> "base"; see _FALLBACKS
    iterations: int = 5
    threshold: float = 0.6
    max_attempts: int = 10
    denoise: bool = False
    output_sample_rate: int = 24000
    reference: Optional[str] = None
    voice_pool: Optional[str] = None
    selection: Optional[str] = None      # -> "most_distant"; see _FALLBACKS
    select_top_k: int = 10
    voice_pool_dir: Optional[str] = None

    def __post_init__(self):
        for f in fields(self):
            env_var = _ENV_VARS.get(f.name)
            if getattr(self, f.name) is None and env_var and os.getenv(env_var):
                setattr(self, f.name, os.getenv(env_var))

        for name, fallback in _FALLBACKS.items():
            if getattr(self, name) is None:
                setattr(self, name, fallback)

        if self.model_dir is None:
            # Same directory `anonymize download-model` writes to, so a fresh checkout
            # works with no config file and no environment variable at all.
            from .download import default_model_dir

            self.model_dir = default_model_dir()

        if self.device is None:
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")

        if self.selection not in SELECTION_STRATEGIES:
            raise ValueError(
                f"selection must be one of {SELECTION_STRATEGIES}, got {self.selection!r}"
            )

        self.select_top_k = int(self.select_top_k)
        if self.select_top_k < 1:
            raise ValueError(f"select_top_k must be at least 1, got {self.select_top_k}")

        self.iterations = int(self.iterations)
        self.max_attempts = int(self.max_attempts)
        self.threshold = float(self.threshold)
        self.output_sample_rate = int(self.output_sample_rate)

    @classmethod
    def from_yaml(cls, path: str, **overrides: Any) -> "AnonymizerConfig":
        """Load from a YAML file. Keyword overrides win over the file."""
        import yaml

        with open(path) as fh:
            data = yaml.safe_load(fh) or {}

        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"unknown option(s) in {path}: {', '.join(sorted(unknown))}. "
                f"Valid options: {', '.join(sorted(known))}"
            )

        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)

    @classmethod
    def resolve(cls, config_path: Optional[str] = None, **overrides: Any) -> "AnonymizerConfig":
        """Build a config from an optional YAML path plus non-None overrides."""
        if config_path:
            return cls.from_yaml(config_path, **overrides)
        return cls(**{k: v for k, v in overrides.items() if v is not None})

    def to_dict(self) -> dict:
        return asdict(self)
