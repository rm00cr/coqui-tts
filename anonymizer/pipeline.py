"""The public entry point: `Anonymizer`.

    from anonymizer import Anonymizer

    anon = Anonymizer()                                   # nothing loaded yet
    wav = anon.anonymize("my.wav", reference="donor.wav") # models load on first call
    anon.save(wav, "out.wav")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .audio import MODEL_SAMPLE_RATE, maybe_denoised, save_wav, to_numpy
from .config import AnonymizerConfig
from .modes import run_mode
from .session import AnonymizerSession
from .voices import ReferenceSpec, resolve_reference


@dataclass
class AnonymizationResult:
    """One anonymized utterance and what is known about it."""

    wav: np.ndarray
    sample_rate: int
    text: str
    reference: list
    mode: str
    info: Dict = field(default_factory=dict)
    output_path: Optional[str] = None

    def to_dict(self) -> dict:
        """Flat, CSV-friendly view. Excludes the waveform."""
        row = {
            "output_path": self.output_path,
            "mode": self.mode,
            "sample_rate": self.sample_rate,
            "text": self.text,
            "reference": ";".join(self.reference),
        }
        row.update({k: v for k, v in self.info.items() if not isinstance(v, (list, dict))})
        return row


class Anonymizer:
    """Anonymizes the speaker identity in speech while preserving the words.

    Given a *target* recording and a *reference* donor voice, the model resynthesizes
    what the target said in the donor's voice. The transcript survives; the speaker
    identity does not.
    """

    def __init__(self, config: Optional[AnonymizerConfig] = None, **config_overrides):
        if config is None:
            config = AnonymizerConfig(**config_overrides)
        elif config_overrides:
            merged = config.to_dict()
            merged.update({k: v for k, v in config_overrides.items() if v is not None})
            config = AnonymizerConfig(**merged)
        self.config = config
        self.session = AnonymizerSession(config)

    # -- main API ---------------------------------------------------------------

    def anonymize(
        self,
        target: str,
        reference: ReferenceSpec = None,
        text: Optional[str] = None,
        language: Optional[str] = None,
        mode: Optional[str] = None,
        output_path: Optional[str] = None,
        denoise: Optional[bool] = None,
        gender: Optional[str] = None,
        pool_language: Optional[str] = None,
    ) -> AnonymizationResult:
        """Anonymize one audio file.

        Args:
            target: path to the audio being anonymized. Its words are preserved.
            reference: the donor voice — a wav path, a directory of wavs, or a
                comma-separated list. Falls back to the configured reference or voice
                pool. The output will sound like this speaker.
            text: transcript of `target`. Transcribed with Whisper when omitted.
            language: two-letter code; defaults to the configured language.
            mode: "single", "refine" or "iterate"; defaults to the configured mode.
            output_path: if given, the result is written here.
            denoise: run noise reduction on the target first.
            gender: restrict pool donors to this gender before choosing one. Only used
                when a donor is chosen from `voice_pool`.
            pool_language: restrict pool donors to this recording language.

        Returns:
            An `AnonymizationResult`. Its `.wav` is a 1-D float32 array at 24 kHz.
        """
        if not os.path.isfile(target):
            raise FileNotFoundError(f"target audio not found: {target}")

        config = self.config
        mode = mode or config.mode
        language = (language or config.language).lower()
        denoise = config.denoise if denoise is None else denoise
        with maybe_denoised(target, denoise) as target_path:
            references, selection_info = self._resolve_donor(
                target_path, reference, gender=gender, pool_language=pool_language
            )
            resolved_text = text or self.session.transcribe(target_path, language)
            if not resolved_text.strip():
                raise ValueError(
                    f"No speech transcribed from {target}. Pass text= explicitly if the "
                    f"audio is valid."
                )

            wav, info = run_mode(
                mode, self.session, target_path, references, language, resolved_text
            )
            info.update(selection_info)

        result = AnonymizationResult(
            wav=to_numpy(wav),
            sample_rate=MODEL_SAMPLE_RATE,
            text=resolved_text,
            reference=references,
            mode=mode,
            info=info,
        )

        if output_path:
            result.output_path = self.save(result, output_path)
        return result

    def select_reference(
        self,
        target: str,
        gender: Optional[str] = None,
        pool_language: Optional[str] = None,
        top_k: Optional[int] = None,
    ):
        """Choose the donor furthest from `target` out of the configured `voice_pool`.

        Runs only ECAPA2 — no XTTS, no Whisper — so it is cheap enough to audit a whole
        pool before committing to a synthesis run.

        Returns:
            A `SelectionResult`: the chosen speaker, the clips to condition on, and the
            similarity scores behind the choice.
        """
        from .voices import VoiceResolutionError

        pool = self.session.voice_pool
        if pool is None:
            raise VoiceResolutionError(
                "no voice_pool configured. Set voice_pool to a directory of donor audio "
                "or a CSV manifest, or pass --voice-pool."
            )
        return self.session.selector.select(
            pool,
            self.session.embed_speaker(target),
            top_k=top_k or self.config.select_top_k,
            gender=gender,
            language=pool_language,
        )

    def _resolve_donor(self, target_path, reference, gender=None, pool_language=None):
        """Decide which donor audio to condition on, and say how it was decided.

        Precedence: an explicit `reference` argument, then a configured `reference`, then
        choosing one out of `voice_pool`, then the legacy flat `voice_pool_dir`.
        """
        config = self.config
        explicit = reference if reference is not None else config.reference
        if explicit is not None:
            return resolve_reference(explicit, config.voice_pool_dir), {}

        if config.voice_pool and config.selection != "none":
            result = self.select_reference(
                target_path, gender=gender, pool_language=pool_language
            )
            return list(result.clips), result.to_info()

        if config.voice_pool:  # selection disabled: average the whole pool, as before
            from .voices import VoicePool

            return VoicePool.load(config.voice_pool).paths, {"selection": "none"}

        return resolve_reference(None, config.voice_pool_dir), {}

    # -- convenience ------------------------------------------------------------

    def save(self, result, path: str) -> str:
        """Write a result (or a raw waveform) to `path` at the configured rate."""
        wav = result.wav if isinstance(result, AnonymizationResult) else result
        return save_wav(wav, path, self.config.output_sample_rate)

    def play(self, result):
        """Return an IPython audio widget. For notebook use."""
        from IPython.display import Audio

        wav = result.wav if isinstance(result, AnonymizationResult) else result
        return Audio(to_numpy(wav), rate=MODEL_SAMPLE_RATE)

    def score(self, result: AnonymizationResult, target: str):
        """Measure how well `result` anonymized `target`.

        Returns a `QualityReport`: WER and BLEU against the original transcript, plus
        speaker similarity to the original speaker (should be low) and to the donor
        (should be high).
        """
        import librosa

        from .scoring import build_report, speaker_similarity

        wav_16k = librosa.resample(result.wav, orig_sr=MODEL_SAMPLE_RATE, target_sr=16000)
        generated_text = self.session.asr.transcribe(
            wav_16k, language=self.config.language
        )["text"]

        synth_emb = self.session.embed_waveform(result.wav, MODEL_SAMPLE_RATE)
        target_emb = self.session.embed_speaker(target)
        reference_emb = self.session.embed_speaker(result.reference[0])

        return build_report(
            reference_text=result.text,
            generated_text=generated_text,
            target_similarity=speaker_similarity(target_emb, synth_emb),
            reference_similarity=speaker_similarity(reference_emb, synth_emb),
        )


def anonymize(target: str, reference: ReferenceSpec = None, **kwargs) -> AnonymizationResult:
    """One-shot convenience wrapper. Builds a throwaway session.

    Prefer holding an `Anonymizer` when doing more than one file — this reloads the
    checkpoints every call.
    """
    config_keys = set(AnonymizerConfig().to_dict())
    config_overrides = {k: v for k, v in kwargs.items() if k in config_keys}
    call_kwargs = {k: v for k, v in kwargs.items() if k not in config_keys}
    return Anonymizer(**config_overrides).anonymize(target, reference, **call_kwargs)
