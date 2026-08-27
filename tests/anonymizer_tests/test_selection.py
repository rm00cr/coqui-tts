"""Choosing the donor voice furthest from the target.

The algorithm is exercised with injected embeddings and an injected similarity, so these
run without ECAPA2, without XTTS, and without any audio.
"""

import pytest

from anonymizer.selection import (
    Ranked,
    VoiceSelectionError,
    VoiceSelector,
    mean_similarity_per_speaker,
    most_distant,
    rank_ascending,
)
from anonymizer.voices import VoiceClip, VoicePool


def cosine(a, b):
    """Cosine similarity over plain lists, so tests need no torch."""
    dot = sum(x * y for x, y in zip(a, b))
    norm = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
    return dot / norm if norm else 0.0


@pytest.fixture
def pool():
    """Three speakers, two clips each, with gender metadata."""
    clips = [
        VoiceClip("near/a.wav", "near", gender="female"),
        VoiceClip("near/b.wav", "near", gender="female"),
        VoiceClip("mid/a.wav", "mid", gender="female"),
        VoiceClip("mid/b.wav", "mid", gender="female"),
        VoiceClip("far/a.wav", "far", gender="male"),
        VoiceClip("far/b.wav", "far", gender="male"),
    ]
    return VoicePool(clips)


@pytest.fixture
def embeddings():
    """Unit vectors laid out so "near" hugs the target and "far" opposes it."""
    return {
        "target": [1.0, 0.0],
        "near/a.wav": [1.0, 0.05],
        "near/b.wav": [1.0, 0.10],
        "mid/a.wav": [1.0, 1.0],
        "mid/b.wav": [1.0, 1.2],
        "far/a.wav": [-1.0, 0.0],
        "far/b.wav": [-1.0, 0.1],
    }


@pytest.fixture
def selector(embeddings):
    return VoiceSelector(embed=lambda path: embeddings[path], similarity=cosine)


class TestRanking:
    def test_rank_ascending_puts_the_most_distant_first(self):
        ranked = rank_ascending({"a": 0.9, "b": -0.2, "c": 0.4})
        assert [r.key for r in ranked] == ["b", "c", "a"]

    def test_rank_ascending_returns_similarities_with_the_keys(self):
        assert rank_ascending({"a": 0.25}) == [Ranked("a", 0.25)]

    def test_ties_break_on_key_so_runs_are_reproducible(self):
        assert [r.key for r in rank_ascending({"z": 0.5, "a": 0.5})] == ["a", "z"]

    def test_most_distant_takes_the_k_lowest(self):
        assert most_distant({"a": 0.9, "b": -0.2, "c": 0.4}, k=2) == ["b", "c"]

    def test_most_distant_defaults_to_one(self):
        assert most_distant({"a": 0.9, "b": -0.2}) == ["b"]

    def test_most_distant_on_empty_candidates_is_an_error(self):
        with pytest.raises(VoiceSelectionError, match="nothing to select from"):
            most_distant({})

    def test_most_distant_rejects_k_below_one(self):
        with pytest.raises(VoiceSelectionError, match="at least 1"):
            most_distant({"a": 0.1}, k=0)

    def test_speaker_score_is_the_mean_of_its_clips(self):
        scores = mean_similarity_per_speaker(
            {"x1": 0.2, "x2": 0.4, "y1": 0.9},
            {"x1": "x", "x2": "x", "y1": "y"},
        )
        assert scores == {"x": pytest.approx(0.3), "y": pytest.approx(0.9)}


class TestSelection:
    def test_picks_the_speaker_furthest_from_the_target(self, selector, pool, embeddings):
        result = selector.select(pool, embeddings["target"])
        assert result.speaker_id == "far"

    def test_conditions_only_on_clips_of_the_chosen_speaker(self, selector, pool, embeddings):
        result = selector.select(pool, embeddings["target"])
        assert all(path.startswith("far/") for path in result.clips)

    def test_clips_are_ordered_furthest_first(self, selector, pool, embeddings):
        result = selector.select(pool, embeddings["target"])
        assert result.clip_similarities == sorted(result.clip_similarities)

    def test_top_k_limits_the_conditioning_set(self, selector, pool, embeddings):
        result = selector.select(pool, embeddings["target"], top_k=1)
        assert len(result.clips) == 1

    def test_top_k_larger_than_the_speaker_is_not_padded(self, selector, pool, embeddings):
        result = selector.select(pool, embeddings["target"], top_k=50)
        assert len(result.clips) == 2

    def test_every_speaker_is_ranked_not_just_the_winner(self, selector, pool, embeddings):
        result = selector.select(pool, embeddings["target"])
        assert [r.key for r in result.ranked_speakers] == ["far", "mid", "near"]

    def test_gender_filter_restricts_the_candidates(self, selector, pool, embeddings):
        # "far" is male, so a female-only search must settle for the next best.
        result = selector.select(pool, embeddings["target"], gender="female")
        assert result.speaker_id == "mid"

    def test_gender_filter_accepts_short_forms(self, selector, pool, embeddings):
        assert selector.select(pool, embeddings["target"], gender="F").speaker_id == "mid"

    def test_a_filter_that_empties_the_pool_raises_rather_than_returning_nothing(
        self, selector, pool, embeddings
    ):
        with pytest.raises(VoiceSelectionError, match="no donor voices left"):
            selector.select(pool, embeddings["target"], gender="other")

    def test_empty_pool_raises(self, selector, embeddings):
        with pytest.raises(VoiceSelectionError, match="no donor voices left"):
            selector.select(VoicePool([]), embeddings["target"])

    def test_embeddings_are_cached_across_targets(self, pool, embeddings):
        calls = []

        def counting_embed(path):
            calls.append(path)
            return embeddings[path]

        selector = VoiceSelector(embed=counting_embed, similarity=cosine)
        selector.select(pool, embeddings["target"])
        selector.select(pool, embeddings["target"])
        assert len(calls) == len(pool)  # second pass hit the cache

    def test_result_records_the_evidence_for_the_choice(self, selector, pool, embeddings):
        info = selector.select(pool, embeddings["target"]).to_info()
        assert info["selection"] == "most_distant"
        assert info["selected_speaker"] == "far"
        assert info["pool_candidates"] == 6
        assert info["selected_speaker_similarity"] < 0
