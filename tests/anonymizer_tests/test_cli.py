"""CLI argument parsing and config wiring. No models are loaded."""

import pytest

from anonymizer.cli import _config_from_args, build_parser


def parse(*argv):
    return build_parser().parse_args(list(argv))


class TestParser:
    def test_run_requires_an_output(self):
        with pytest.raises(SystemExit):
            parse("run", "in.wav")

    def test_run_parses_core_arguments(self):
        args = parse("run", "in.wav", "-o", "out.wav", "--reference", "donor.wav")
        assert (args.command, args.input, args.output) == ("run", "in.wav", "out.wav")
        assert args.reference == "donor.wav"

    def test_batch_parses_core_arguments(self):
        args = parse("batch", "inputs/", "-o", "outputs/", "--resume")
        assert (args.command, args.input_dir, args.output_dir) == ("batch", "inputs/", "outputs/")
        assert args.resume is True

    def test_a_command_is_required(self):
        with pytest.raises(SystemExit):
            parse()

    def test_invalid_mode_is_rejected_by_the_parser(self):
        with pytest.raises(SystemExit):
            parse("run", "in.wav", "-o", "out.wav", "--mode", "nonsense")

    def test_unset_flags_are_none_so_they_fall_through(self):
        """Anything left unset must be None, or it would clobber the config file."""
        args = parse("run", "in.wav", "-o", "out.wav")
        for field in (
            "mode", "language", "device", "threshold", "denoise", "reference",
            "voice_pool", "selection", "select_top_k",
        ):
            assert getattr(args, field) is None, f"{field} should default to None"

    def test_download_and_config_subcommands_exist(self):
        assert parse("download-model").command == "download-model"
        assert parse("config", "--show").command == "config"

    def test_select_subcommand_previews_the_donor_choice(self):
        args = parse("select", "in.wav", "--voice-pool", "pool.csv")
        assert (args.command, args.input, args.voice_pool) == ("select", "in.wav", "pool.csv")

    def test_pool_flags_parse_on_run(self):
        args = parse(
            "run", "in.wav", "-o", "out.wav",
            "--voice-pool", "pool.csv", "--gender", "female", "--select-top-k", "3",
        )
        assert (args.voice_pool, args.gender, args.select_top_k) == ("pool.csv", "female", 3)

    def test_pool_flags_parse_on_batch(self):
        args = parse("batch", "in/", "-o", "out/", "--voice-pool", "voices/", "--pool-language", "de")
        assert (args.voice_pool, args.pool_language) == ("voices/", "de")

    def test_invalid_selection_is_rejected_by_the_parser(self):
        with pytest.raises(SystemExit):
            parse("run", "in.wav", "-o", "out.wav", "--selection", "closest")


class TestConfigFromArgs:
    def test_flags_reach_the_config(self):
        args = parse("run", "in.wav", "-o", "out.wav", "--mode", "refine", "--lang", "de")
        config = _config_from_args(args)
        assert config.mode == "refine"
        assert config.language == "de"

    def test_unset_flags_leave_defaults_intact(self):
        config = _config_from_args(parse("run", "in.wav", "-o", "out.wav"))
        assert config.mode == "single"
        assert config.language == "en"

    def test_flag_beats_config_file(self, tmp_path):
        conf = tmp_path / "c.yaml"
        conf.write_text("mode: refine\nlanguage: de\n")
        args = parse("run", "in.wav", "-o", "out.wav", "--config", str(conf), "--mode", "single")
        config = _config_from_args(args)
        assert config.mode == "single"  # flag wins
        assert config.language == "de"  # file survives where no flag was given

    def test_numeric_flags_are_typed(self):
        args = parse("run", "in.wav", "-o", "out.wav", "--iterations", "3", "--threshold", "0.8")
        config = _config_from_args(args)
        assert config.iterations == 3
        assert config.threshold == pytest.approx(0.8)


class TestMainErrorHandling:
    def test_missing_input_exits_with_code_2(self, capsys):
        from anonymizer.cli import main

        code = main(["run", "/nonexistent/file.wav", "-o", "/tmp/out.wav", "--reference", "/nonexistent/ref.wav"])
        assert code == 2
        assert "error:" in capsys.readouterr().err

    def test_pool_flags_reach_the_config(self):
        args = parse(
            "run", "in.wav", "-o", "out.wav",
            "--voice-pool", "pool.csv", "--selection", "none", "--select-top-k", "4",
        )
        config = _config_from_args(args)
        assert (config.voice_pool, config.selection, config.select_top_k) == ("pool.csv", "none", 4)

    def test_selection_defaults_to_most_distant(self):
        assert _config_from_args(parse("run", "in.wav", "-o", "out.wav")).selection == "most_distant"
