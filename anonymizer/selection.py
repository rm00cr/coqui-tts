"""Choosing the donor voice that is furthest from the target speaker.

The anonymization is only as good as the donor: a donor who already sounds like the
target hides very little. So rather than fixing one reference up front, this picks —
per target — the pool speaker whose ECAPA2 embedding is *least* similar to the target's,
and then the clips of that speaker that are least similar in turn.

    target ── embed ──┐
                      ├── cosine similarity ── rank ascending ── take the furthest
    pool clips ─ embed┘

Two stages, matching the procedure used for the cluster runs
(``run_test_hdf5.py`` searched donors, ``run_test_hdf5_test_data.py`` ranked their clips):

1. **speaker** — score every pool speaker by the mean similarity of their clips to the
   target; the lowest-scoring speaker wins.
2. **clip** — within that speaker, keep the ``top_k`` least-similar clips. XTTS averages
   the conditioning across whatever it is given, so this is the conditioning set.

The ranking functions take plain floats and injected callables, so the algorithm is
testable without model weights — see ``tests/anonymizer_tests/test_selection.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .voices import VoiceClip, VoicePool

#: How many of the furthest clips to condition on. Ten is what the cluster runs used.
DEFAULT_TOP_K = 10

#: Selection strategies accepted by the config. "none" keeps the historical behaviour of
#: averaging the whole pool.
STRATEGIES = ("most_distant", "none")


class VoiceSelectionError(ValueError):
    """Raised when no donor can be chosen — an empty pool, or filters that exclude every clip."""


@dataclass(frozen=True)
class Ranked:
    """One candidate and how similar it is to the target. Lower similarity = further away."""

    key: str
    similarity: float


@dataclass
class SelectionResult:
    """Which donor was chosen for a target, and the evidence for it."""

    speaker_id: str
    clips: List[str]
    speaker_similarity: float
    clip_similarities: List[float] = field(default_factory=list)
    ranked_speakers: List[Ranked] = field(default_factory=list)
    candidates: int = 0

    def to_info(self) -> dict:
        """The subset worth recording alongside an anonymization result."""
        return {
            "selection": "most_distant",
            "selected_speaker": self.speaker_id,
            "selected_speaker_similarity": round(self.speaker_similarity, 6),
            "selected_clips": len(self.clips),
            "pool_candidates": self.candidates,
            "clip_similarities": [round(s, 6) for s in self.clip_similarities],
        }


def rank_ascending(similarities: Mapping[str, float]) -> List[Ranked]:
    """Candidates ordered furthest-first (lowest cosine similarity first).

    Ties break on the key so a run is reproducible regardless of dict ordering.
    """
    return [
        Ranked(key=key, similarity=value)
        for key, value in sorted(similarities.items(), key=lambda kv: (kv[1], kv[0]))
    ]


def most_distant(similarities: Mapping[str, float], k: int = 1) -> List[str]:
    """The `k` keys least similar to the target, furthest first."""
    if not similarities:
        raise VoiceSelectionError("nothing to select from: no candidate similarities")
    if k < 1:
        raise VoiceSelectionError(f"k must be at least 1, got {k}")
    return [ranked.key for ranked in rank_ascending(similarities)[:k]]


def mean_similarity_per_speaker(
    clip_similarities: Mapping[str, float],
    speaker_of: Mapping[str, str],
) -> Dict[str, float]:
    """Average each speaker's clip similarities into one score per speaker."""
    totals: Dict[str, List[float]] = {}
    for clip_key, similarity in clip_similarities.items():
        totals.setdefault(speaker_of[clip_key], []).append(similarity)
    return {speaker: sum(values) / len(values) for speaker, values in totals.items()}


class VoiceSelector:
    """Applies the most-distant rule to a `VoicePool`.

    Args:
        embed: turns an audio path into a speaker embedding.
        similarity: cosine similarity between two embeddings. Defaults to the ECAPA2
            scoring used everywhere else in the package.
        cache: path -> embedding, reused across targets. A pool is embedded once no
            matter how many files you anonymize against it.
    """

    def __init__(
        self,
        embed: Callable[[str], Any],
        similarity: Optional[Callable[[Any, Any], float]] = None,
        cache: Optional[Dict[str, Any]] = None,
    ):
        self.embed = embed
        self._similarity = similarity
        self.cache: Dict[str, Any] = {} if cache is None else cache

    def similarity(self, a: Any, b: Any) -> float:
        if self._similarity is not None:
            return self._similarity(a, b)
        from .scoring import speaker_similarity

        return speaker_similarity(a, b)

    def embedding_for(self, path: str) -> Any:
        """Embed `path`, reusing the cached embedding when there is one."""
        if path not in self.cache:
            self.cache[path] = self.embed(path)
        return self.cache[path]

    def score_clips(self, clips: Sequence[VoiceClip], target_embedding: Any) -> Dict[str, float]:
        """Similarity of every clip to the target, keyed by path."""
        return {
            clip.path: float(self.similarity(self.embedding_for(clip.path), target_embedding))
            for clip in clips
        }

    def select(
        self,
        pool: VoicePool,
        target_embedding: Any,
        top_k: int = DEFAULT_TOP_K,
        gender: Optional[str] = None,
        language: Optional[str] = None,
    ) -> SelectionResult:
        """Pick the donor speaker furthest from the target, and their furthest clips.

        Args:
            pool: the candidate donors.
            target_embedding: ECAPA2 embedding of the audio being anonymized.
            top_k: how many of that speaker's clips to condition on.
            gender: restrict to donors of this gender before ranking. Requires the pool
                to carry gender metadata.
            language: restrict to donors recorded in this language.

        Raises:
            VoiceSelectionError: if the filters leave no candidates. This is deliberate —
                an empty candidate set must never fall through to conditioning on nothing.
        """
        candidates = pool.filter(gender=gender, language=language)
        if not candidates:
            applied = ", ".join(
                f"{name}={value!r}" for name, value in (("gender", gender), ("language", language)) if value
            )
            raise VoiceSelectionError(
                f"no donor voices left in the pool after filtering ({applied or 'no filters'}). "
                f"The pool holds {len(pool)} clip(s) from {len(pool.speakers)} speaker(s); "
                f"check that it carries the metadata you are filtering on."
            )

        clip_similarities = self.score_clips(candidates.clips, target_embedding)
        speaker_of = {clip.path: clip.speaker_id for clip in candidates.clips}

        speaker_scores = mean_similarity_per_speaker(clip_similarities, speaker_of)
        ranked_speakers = rank_ascending(speaker_scores)
        chosen = ranked_speakers[0]

        chosen_clips = {
            path: similarity
            for path, similarity in clip_similarities.items()
            if speaker_of[path] == chosen.key
        }
        selected = most_distant(chosen_clips, k=min(top_k, len(chosen_clips)))

        return SelectionResult(
            speaker_id=chosen.key,
            clips=selected,
            speaker_similarity=chosen.similarity,
            clip_similarities=[chosen_clips[path] for path in selected],
            ranked_speakers=ranked_speakers,
            candidates=len(candidates),
        )
