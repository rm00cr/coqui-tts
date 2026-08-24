"""Config resolution: defaults, environment, YAML, and override precedence."""

import os

import pytest

from anonymizer.config import AnonymizerConfig


def test_defaults_are_usable_without_any_setup():
    config = AnonymizerConfig()
    assert config.mode == "single"
    assert config.language == "en"
    assert config.output_sample_rate == 24000
    assert config.device in ("cuda", "cpu")
    assert config.model_dir  # always resolves to something


def test_explicit_values_win_over_defaults():
    config = AnonymizerConfig(mode="refine", language="de", threshold=0.9)
    assert config.mode == "refine"
    assert config.language == "de"
    assert config.threshold == 0.9


def test_environment_fills_unset_fields(monkeypatch):
    monkeypatch.setenv("XTTS_MODEL_DIR", "/tmp/some-model-dir")
    monkeypatch.setenv("ANONYMIZER_MODE", "iterate")
    config = AnonymizerConfig()
    assert config.model_dir == "/tmp/some-model-dir"
    assert config.mode == "iterate"


def test_explicit_value_beats_environment(monkeypatch):
    monkeypatch.setenv("XTTS_MODEL_DIR", "/tmp/from-env")
    config = AnonymizerConfig(model_dir="/tmp/explicit")
    assert config.model_dir == "/tmp/explicit"


def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError, match="mode must be one of"):
        AnonymizerConfig(mode="nonsense")


def test_numeric_fields_are_coerced():
    config = AnonymizerConfig(iterations="7", threshold="0.42", output_sample_rate="16000")
    assert config.iterations == 7
    assert config.threshold == pytest.approx(0.42)
    assert config.output_sample_rate == 16000


def test_from_yaml_reads_file(tmp_path):
    path = tmp_path / "conf.yaml"
    path.write_text("mode: refine\nlanguage: de\nthreshold: 0.75\n")
    config = AnonymizerConfig.from_yaml(str(path))
    assert (config.mode, config.language, config.threshold) == ("refine", "de", 0.75)


def test_yaml_overrides_win_over_file(tmp_path):
    path = tmp_path / "conf.yaml"
    path.write_text("mode: refine\nlanguage: de\n")
    config = AnonymizerConfig.from_yaml(str(path), mode="single")
    assert config.mode == "single"
    assert config.language == "de"  # untouched by the override


def test_none_overrides_do_not_clobber_the_file(tmp_path):
    """An unset CLI flag arrives as None and must not overwrite the config file."""
    path = tmp_path / "conf.yaml"
    path.write_text("mode: refine\n")
    config = AnonymizerConfig.from_yaml(str(path), mode=None)
    assert config.mode == "refine"


def test_unknown_yaml_key_is_reported(tmp_path):
    path = tmp_path / "conf.yaml"
    path.write_text("mode: single\ntypoed_option: 3\n")
    with pytest.raises(ValueError, match="typoed_option"):
        AnonymizerConfig.from_yaml(str(path))


def test_resolve_without_config_path_uses_overrides():
    config = AnonymizerConfig.resolve(None, mode="iterate", language=None)
    assert config.mode == "iterate"
    assert config.language == "en"  # None override fell through to the default


def test_to_dict_roundtrips():
    config = AnonymizerConfig(mode="refine", iterations=3)
    assert AnonymizerConfig(**config.to_dict()).to_dict() == config.to_dict()
