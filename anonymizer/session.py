"""Lazily-loaded model state shared across anonymization calls.

Everything expensive lives here and is built on first use, never at import time:

* the XTTS model and its GPTTrainer wrapper (~2 GB of checkpoints),
* Whisper, for transcribing the target when no text is supplied,
* ECAPA2, for measuring how far the output moved from the original speaker.

Hold one session and reuse it: in a notebook or a batch run the checkpoints load once.
"""

from __future__ import annotations

import os
from typing import Optional

from .config import AnonymizerConfig

#: Speaker-verification model used to score anonymization. Downloaded on first use.
ECAPA2_REPO = "Jenthe/ECAPA2"
ECAPA2_FILE = "ecapa2.pt"

REQUIRED_CHECKPOINT_FILES = ("model.pth", "dvae.pth", "mel_stats.pth", "vocab.json", "config.json")


class ModelFilesMissing(FileNotFoundError):
    """Raised when the XTTS checkpoint directory is absent or incomplete."""


class AnonymizerSession:
    """Holds the loaded models for a given configuration.

    Nothing is loaded until the corresponding property is first accessed, so
    constructing a session is free.
    """

    def __init__(self, config: Optional[AnonymizerConfig] = None):
        self.config = config or AnonymizerConfig()
        self._tts = None
        self._model = None
        self._train_model = None
        self._gpt_config = None
        self._asr = None
        self._ecapa = None
        self._selector = None
        self._pool = None

    # -- checkpoints ------------------------------------------------------------

    def check_model_files(self) -> None:
        """Verify the XTTS checkpoint directory before trying to load 2 GB from it."""
        model_dir = self.config.model_dir
        if not os.path.isdir(model_dir):
            raise ModelFilesMissing(
                f"XTTS model directory not found: {model_dir}\n"
                f"Download the checkpoints with `anonymize download-model`, or point "
                f"$XTTS_MODEL_DIR / config model_dir at an existing copy."
            )
        missing = [f for f in REQUIRED_CHECKPOINT_FILES if not os.path.isfile(os.path.join(model_dir, f))]
        if missing:
            raise ModelFilesMissing(
                f"XTTS model directory {model_dir} is missing: {', '.join(missing)}\n"
                f"Run `anonymize download-model --dest {model_dir}` to fetch them."
            )

    def _load_xtts(self) -> None:
        if self._model is not None:
            return
        self.check_model_files()

        from .model_setup import ModelPaths, load_tts_and_trainer

        paths = ModelPaths(checkpoints_out_path=self.config.model_dir)
        tts, model, train_model, gpt_config = load_tts_and_trainer(paths, device=self.config.device)
        self._tts, self._model, self._train_model, self._gpt_config = tts, model, train_model, gpt_config

    # -- lazy properties --------------------------------------------------------

    @property
    def tts(self):
        """The high-level TTS wrapper; used for its wav writer."""
        self._load_xtts()
        return self._tts

    @property
    def model(self):
        """The `Xtts` instance carrying the anonymization forward methods."""
        self._load_xtts()
        return self._model

    @property
    def train_model(self):
        """GPTTrainer, needed only for `format_batch_on_device`."""
        self._load_xtts()
        return self._train_model

    @property
    def gpt_config(self):
        """GPTTrainerConfig, source of the min/max conditioning lengths."""
        self._load_xtts()
        return self._gpt_config

    @property
    def asr(self):
        """Whisper, for transcribing a target that came without text."""
        if self._asr is None:
            import whisper

            self._asr = whisper.load_model(self.config.whisper_model, device=self.config.device)
        return self._asr

    @property
    def ecapa(self):
        """ECAPA2 speaker encoder, for similarity scoring."""
        if self._ecapa is None:
            import torch
            from huggingface_hub import hf_hub_download

            weights = hf_hub_download(repo_id=ECAPA2_REPO, filename=ECAPA2_FILE)
            self._ecapa = torch.jit.load(weights, map_location=self.config.device)
        return self._ecapa

    @property
    def voice_pool(self):
        """The configured donor pool, loaded once. None when no pool is configured."""
        if self._pool is None:
            spec = self.config.voice_pool
            if not spec:
                return None
            from .voices import VoicePool

            self._pool = VoicePool.load(spec)
        return self._pool

    @property
    def selector(self):
        """A `VoiceSelector` bound to this session's ECAPA2, with a shared cache.

        The cache lives on the session, so a pool is embedded once per session no matter
        how many targets are anonymized against it.
        """
        if self._selector is None:
            from .selection import VoiceSelector

            self._selector = VoiceSelector(embed=self.embed_speaker)
        return self._selector

    # -- conditioning lengths ---------------------------------------------------

    @property
    def max_conditioning_length(self) -> int:
        return self.gpt_config.model_args.max_conditioning_length

    @property
    def min_conditioning_length(self) -> int:
        return self.gpt_config.model_args.min_conditioning_length

    # -- helpers ----------------------------------------------------------------

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        """Transcribe an audio file with Whisper."""
        kwargs = {"language": language} if language else {}
        return self.asr.transcribe(audio_path, **kwargs)["text"].strip()

    def embed_speaker(self, audio_path: str):
        """ECAPA2 embedding for an audio file, resampled to the 16 kHz ECAPA expects.

        Loads the audio directly rather than through XTTS, so scoring and donor
        selection work without the synthesis checkpoints being present.
        """
        from .audio import load_16k_mono

        device = next(self.ecapa.parameters()).device
        return self.ecapa(load_16k_mono(audio_path).to(device=device))

    def embed_waveform(self, wav, sample_rate: int = 24000):
        """ECAPA2 embedding for an in-memory waveform."""
        import librosa
        import torch

        from .audio import to_numpy

        wav = to_numpy(wav)
        if sample_rate != 16000:
            wav = librosa.resample(wav, orig_sr=sample_rate, target_sr=16000)
        device = next(self.ecapa.parameters()).device
        return self.ecapa(torch.from_numpy(wav).float().unsqueeze(0).to(device=device))
