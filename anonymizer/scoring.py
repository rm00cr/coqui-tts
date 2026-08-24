"""Metrics for judging an anonymization.

A good anonymization satisfies two competing goals: the words must survive (low WER,
high BLEU against the source transcript) while the speaker identity must not (low
cosine similarity to the *target* speaker, high similarity to the *reference* donor).

The weighted score here is the one used by ``XTTS.forward_iteration`` and
``development/utils.assess_segment_quality``. Weights are parameters so the alternative
formulations elsewhere in the repo can be expressed without another copy of the code.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

#: Default weights, matching the existing pipeline. Reference similarity dominates
#: because sounding like the donor is what actually hides the original speaker.
DEFAULT_WEIGHTS = {
    "wer": 0.05,
    "bleu": 0.05,
    "target_dissimilarity": 0.30,
    "reference_similarity": 0.60,
}


def normalize_text(text: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace.

    Mirrors ``TTS.tts.models.xtts.clean_text`` so scores here match the ones the model's
    own iteration loops report.
    """
    text = (text or "").lower().replace("ß", "ss")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def word_error_rate(reference: str, hypothesis: str) -> float:
    """WER between two transcripts. Returns 1.0 (worst) if it cannot be computed."""
    from jiwer import wer as _wer

    try:
        return float(_wer(reference=normalize_text(reference), hypothesis=normalize_text(hypothesis)))
    except Exception:
        return 1.0


def bleu(reference: str, hypothesis: str) -> float:
    """Sentence BLEU in [0, 1]. Returns 0.0 (worst) if it cannot be computed."""
    from sacrebleu import BLEU

    try:
        scorer = BLEU(effective_order=True)
        return float(scorer.sentence_score(normalize_text(hypothesis), [normalize_text(reference)]).score) / 100.0
    except Exception:
        return 0.0


def speaker_similarity(embedding_a, embedding_b) -> float:
    """Cosine similarity between two ECAPA2 speaker embeddings."""
    import torch

    return float(torch.cosine_similarity(embedding_a, embedding_b).mean())


@dataclass
class QualityReport:
    """Scores for one anonymized utterance."""

    wer: float
    bleu: float
    target_similarity: float
    reference_similarity: float
    overall: float

    def to_dict(self) -> dict:
        return {
            "wer": self.wer,
            "bleu": self.bleu,
            "target_similarity": self.target_similarity,
            "reference_similarity": self.reference_similarity,
            "overall_quality": self.overall,
        }


def quality_score(
    wer: float,
    bleu: float,
    target_similarity: float,
    reference_similarity: float,
    weights: Optional[dict] = None,
) -> float:
    """Combine the four metrics into a single number, higher is better.

    Args:
        wer: word error rate of the anonymized audio against the source transcript.
        bleu: sentence BLEU against the source transcript.
        target_similarity: speaker similarity to the ORIGINAL speaker. Lower is better.
        reference_similarity: speaker similarity to the DONOR voice. Higher is better.
        weights: overrides for `DEFAULT_WEIGHTS`.
    """
    w = dict(DEFAULT_WEIGHTS, **(weights or {}))
    return (
        w["wer"] * (1 - wer)
        + w["bleu"] * bleu
        + w["target_dissimilarity"] * (1 - target_similarity)
        + w["reference_similarity"] * reference_similarity
    )


def build_report(
    reference_text: str,
    generated_text: str,
    target_similarity: float,
    reference_similarity: float,
    weights: Optional[dict] = None,
) -> QualityReport:
    """Score one anonymized utterance end to end."""
    wer_score = word_error_rate(reference_text, generated_text)
    bleu_score = bleu(reference_text, generated_text)
    return QualityReport(
        wer=wer_score,
        bleu=bleu_score,
        target_similarity=target_similarity,
        reference_similarity=reference_similarity,
        overall=quality_score(
            wer_score, bleu_score, target_similarity, reference_similarity, weights
        ),
    )
