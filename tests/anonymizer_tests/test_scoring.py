"""Metric helpers and the combined quality score."""

import pytest

from anonymizer.scoring import (
    DEFAULT_WEIGHTS,
    build_report,
    normalize_text,
    quality_score,
    word_error_rate,
)


class TestNormalizeText:
    def test_lowercases_and_strips_punctuation(self):
        assert normalize_text("Hello, World!") == "hello world"

    def test_collapses_whitespace(self):
        assert normalize_text("  a   b  ") == "a b"

    def test_expands_german_sharp_s(self):
        assert normalize_text("Straße") == "strasse"

    def test_strips_accents(self):
        assert normalize_text("café") == "cafe"

    def test_handles_empty_and_none(self):
        assert normalize_text("") == ""
        assert normalize_text(None) == ""


class TestWordErrorRate:
    def test_identical_text_scores_zero(self):
        assert word_error_rate("the quick brown fox", "the quick brown fox") == 0.0

    def test_one_substitution_in_four_words(self):
        assert word_error_rate("the quick brown fox", "the quick brown dog") == pytest.approx(0.25)

    def test_normalization_makes_punctuation_irrelevant(self):
        assert word_error_rate("Hello, world!", "hello world") == 0.0


class TestQualityScore:
    def test_perfect_anonymization_scores_one(self):
        """Words preserved, sounds nothing like the original, exactly like the donor."""
        assert quality_score(wer=0.0, bleu=1.0, target_similarity=0.0, reference_similarity=1.0) == pytest.approx(1.0)

    def test_worst_case_scores_zero(self):
        assert quality_score(wer=1.0, bleu=0.0, target_similarity=1.0, reference_similarity=0.0) == pytest.approx(0.0)

    def test_reference_similarity_dominates(self):
        """Sounding like the donor is weighted more than any other single term."""
        assert DEFAULT_WEIGHTS["reference_similarity"] > sum(
            v for k, v in DEFAULT_WEIGHTS.items() if k != "reference_similarity"
        )

    def test_lower_target_similarity_is_better(self):
        worse = quality_score(0.1, 0.9, target_similarity=0.9, reference_similarity=0.5)
        better = quality_score(0.1, 0.9, target_similarity=0.1, reference_similarity=0.5)
        assert better > worse

    def test_weights_can_be_overridden(self):
        only_wer = {"wer": 1.0, "bleu": 0.0, "target_dissimilarity": 0.0, "reference_similarity": 0.0}
        assert quality_score(0.25, 0.0, 1.0, 1.0, weights=only_wer) == pytest.approx(0.75)

    def test_matches_the_hand_computed_formula(self):
        expected = 0.05 * (1 - 0.2) + 0.05 * 0.7 + 0.30 * (1 - 0.3) + 0.60 * 0.8
        assert quality_score(0.2, 0.7, 0.3, 0.8) == pytest.approx(expected)


class TestBuildReport:
    def test_populates_every_field(self):
        report = build_report(
            reference_text="the quick brown fox",
            generated_text="the quick brown fox",
            target_similarity=0.2,
            reference_similarity=0.85,
        )
        assert report.wer == 0.0
        assert report.bleu == pytest.approx(1.0)
        assert report.target_similarity == 0.2
        assert report.reference_similarity == 0.85
        assert 0.0 <= report.overall <= 1.0

    def test_to_dict_is_flat_and_csv_friendly(self):
        report = build_report("hello world", "hello world", 0.1, 0.9)
        row = report.to_dict()
        assert set(row) == {"wer", "bleu", "target_similarity", "reference_similarity", "overall_quality"}
        assert all(isinstance(v, float) for v in row.values())
