"""The three anonymization strategies, behind one signature.

Each mode takes the same inputs and returns ``(waveform, info)`` where waveform is a
1-D float32 array at 24 kHz and info is a dict of whatever the strategy measured.

    single   one forward pass. Seconds per file. Always available, lowest quality.
    refine   segments the output, rescores each, regenerates the weak ones and
             crossfade-stitches. Minutes per file, wants a GPU.
    iterate  n whole-utterance passes, each fed the previous output; keeps the
             best-scoring pass. Minutes per file, wants a GPU.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np

from .audio import to_numpy
from .session import AnonymizerSession


def _single(
    session: AnonymizerSession,
    target: str,
    reference: List[str],
    language: str,
    text: str,
) -> Tuple[np.ndarray, Dict]:
    result = session.model.forward_from_audios_and_text(
        language,
        text,
        target,
        reference,
        session.train_model,
        session.max_conditioning_length,
        session.min_conditioning_length,
    )
    return to_numpy(result["wav"]), {"mode": "single"}


def _refine(
    session: AnonymizerSession,
    target: str,
    reference: List[str],
    language: str,
    text: str,
) -> Tuple[np.ndarray, Dict]:
    from development.utils import iterative_segment_refinement

    config = session.config
    wav = iterative_segment_refinement(
        model=session.model,
        target_speaker=target,
        ref_speaker=reference,
        max_conditioning_length=session.max_conditioning_length,
        min_conditioning_length=session.min_conditioning_length,
        train_model=session.train_model,
        asr_model=session.asr,
        ecapa_model=session.ecapa,
        lang=language,
        threshold=config.threshold,
        max_attempts=config.max_attempts,
    )
    # This returns a torch.Tensor or a numpy array depending on its `regenerate` flag;
    # to_numpy collapses both.
    return to_numpy(wav), {
        "mode": "refine",
        "threshold": config.threshold,
        "max_attempts": config.max_attempts,
    }


def _iterate(
    session: AnonymizerSession,
    target: str,
    reference: List[str],
    language: str,
    text: str,
) -> Tuple[np.ndarray, Dict]:
    config = session.config
    save_wavs, save_scores, quality_scores, audios = session.model.forward_iteration(
        lang=language,
        text=text,
        target_sample=target,
        ref_sample=reference,
        train_model=session.train_model,
        max_conditioning_length=session.max_conditioning_length,
        min_conditioning_length=session.min_conditioning_length,
        tts=session.tts,
        asr_model=session.asr,
        ecapa=session.ecapa,
        n=config.iterations,
    )

    if not audios:
        raise RuntimeError("forward_iteration produced no audio")

    best = int(np.argmax(quality_scores))
    info = {
        "mode": "iterate",
        "iterations": len(audios),
        "best_iteration": best,
        "quality_scores": [float(q) for q in quality_scores],
        "wer": float(save_scores["wer_scores"][best]),
        "bleu": float(save_scores["bleu_scores"][best]),
        "target_similarity": float(save_scores["target_cosine_similarities"][best]),
        "reference_similarity": float(save_scores["reference_cosine_similarities"][best]),
    }
    return to_numpy(audios[best]), info


MODE_FUNCTIONS: Dict[str, Callable[..., Tuple[np.ndarray, Dict]]] = {
    "single": _single,
    "refine": _refine,
    "iterate": _iterate,
}


def run_mode(
    mode: str,
    session: AnonymizerSession,
    target: str,
    reference: List[str],
    language: str,
    text: str,
) -> Tuple[np.ndarray, Dict]:
    """Dispatch to one of the strategies in `MODE_FUNCTIONS`."""
    try:
        fn = MODE_FUNCTIONS[mode]
    except KeyError:
        raise ValueError(
            f"unknown mode {mode!r}; expected one of {', '.join(MODE_FUNCTIONS)}"
        ) from None
    return fn(session, target, reference, language, text)
