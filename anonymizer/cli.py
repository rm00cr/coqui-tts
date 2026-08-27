"""Command line interface: ``anonymize``.

    anonymize run   input.wav --reference donor.wav -o out.wav
    anonymize batch inputs/   --reference donor.wav -o outputs/ --resume
    anonymize download-model
    anonymize config --show
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .config import MODES, SELECTION_STRATEGIES


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    """Options that map onto AnonymizerConfig fields. All default to None so that
    unset flags fall through to the config file, then the environment, then defaults."""
    parser.add_argument("--config", help="path to a YAML config file")
    parser.add_argument(
        "--reference",
        help="donor voice: a wav file, a directory of wavs, or a comma-separated list. "
        "The output will sound like this speaker.",
    )
    parser.add_argument(
        "--voice-pool",
        dest="voice_pool",
        help="pool of donor voices to choose from: a directory (pool/<speaker>/*.wav) or "
        "a CSV manifest with an audio-path column. The donor furthest from each input is "
        "picked automatically. Env: ANONYMIZER_VOICE_POOL",
    )
    parser.add_argument(
        "--selection",
        choices=SELECTION_STRATEGIES,
        help="how to pick a donor from --voice-pool (default: most_distant)",
    )
    parser.add_argument(
        "--select-top-k",
        dest="select_top_k",
        type=int,
        help="how many of the chosen donor's clips to condition on (default: 10)",
    )
    parser.add_argument("--mode", choices=MODES, help="anonymization strategy (default: single)")
    parser.add_argument("--lang", dest="language", help="two-letter language code (default: en)")
    parser.add_argument("--model-dir", dest="model_dir", help="directory with the XTTS v2 checkpoints")
    parser.add_argument("--device", help="cuda or cpu (default: auto-detect)")
    parser.add_argument("--whisper-model", dest="whisper_model", help="Whisper size (default: base)")
    parser.add_argument("--iterations", type=int, help="passes for --mode iterate (default: 5)")
    parser.add_argument("--threshold", type=float, help="segment quality threshold for --mode refine")
    parser.add_argument("--max-attempts", dest="max_attempts", type=int, help="retries per weak segment")
    parser.add_argument("--denoise", action="store_true", default=None, help="denoise the input first")
    parser.add_argument(
        "--sample-rate", dest="output_sample_rate", type=int, help="output sample rate (default: 24000)"
    )


def _config_from_args(args) -> "object":
    from .config import AnonymizerConfig

    overrides = {
        key: getattr(args, key, None)
        for key in (
            "reference", "mode", "language", "model_dir", "device", "whisper_model",
            "iterations", "threshold", "max_attempts", "denoise", "output_sample_rate",
            "voice_pool", "selection", "select_top_k",
        )
    }
    return AnonymizerConfig.resolve(getattr(args, "config", None), **overrides)


def _add_pool_filters(parser: argparse.ArgumentParser) -> None:
    """Filters applied to the donor pool before a donor is chosen."""
    parser.add_argument("--gender", help="restrict pool donors to this gender (needs a gender column)")
    parser.add_argument(
        "--pool-language", dest="pool_language", help="restrict pool donors to this recording language"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anonymize",
        description="Anonymize the speaker identity in speech while keeping the words. "
        "Give it the audio to anonymize (the target) and a donor voice (the reference); "
        "the output says the same thing in the donor's voice.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="anonymize a single audio file")
    run.add_argument("input", help="audio file to anonymize")
    run.add_argument("-o", "--output", required=True, help="where to write the anonymized wav")
    run.add_argument("--text", help="transcript of the input; transcribed with Whisper if omitted")
    run.add_argument("--score", action="store_true", help="also report quality metrics (slower)")
    _add_common_options(run)
    _add_pool_filters(run)

    batch = sub.add_parser("batch", help="anonymize every audio file in a directory")
    batch.add_argument("input_dir", help="directory of audio files")
    batch.add_argument("-o", "--output-dir", required=True, help="directory for the anonymized wavs")
    batch.add_argument("--manifest", help="path for the results CSV (default: <output-dir>/manifest.csv)")
    batch.add_argument("--resume", action="store_true", help="skip files already completed")
    _add_common_options(batch)
    _add_pool_filters(batch)

    select = sub.add_parser(
        "select",
        help="show which donor the pool would pick for an input, without synthesizing",
    )
    select.add_argument("input", help="audio file to choose a donor for")
    select.add_argument("--top", type=int, default=5, help="how many ranked speakers to show")
    _add_common_options(select)
    _add_pool_filters(select)

    download = sub.add_parser("download-model", help="fetch the XTTS v2 checkpoint files (~2 GB)")
    download.add_argument("--dest", help="target directory (default: $XTTS_MODEL_DIR or ./XTTS_v2.0_original_model_files)")
    download.add_argument("--force", action="store_true", help="re-download files that already exist")

    config_cmd = sub.add_parser("config", help="inspect the resolved configuration")
    config_cmd.add_argument("--show", action="store_true", help="print the resolved config and exit")
    _add_common_options(config_cmd)

    return parser


def _cmd_run(args) -> int:
    from .pipeline import Anonymizer

    config = _config_from_args(args)
    anonymizer = Anonymizer(config)

    print(f" > Anonymizing {args.input} (mode={config.mode}, device={config.device})")
    result = anonymizer.anonymize(
        args.input,
        reference=config.reference,
        text=args.text,
        output_path=args.output,
        gender=getattr(args, "gender", None),
        pool_language=getattr(args, "pool_language", None),
    )
    if result.info.get("selected_speaker"):
        print(
            f" > Donor: speaker {result.info['selected_speaker']} "
            f"({result.info['selected_clips']} clip(s), similarity "
            f"{result.info['selected_speaker_similarity']:.3f} — lower is further away)"
        )
    print(f" > Transcript: {result.text}")
    print(f" > Wrote {result.output_path}")

    if args.score:
        report = anonymizer.score(result, args.input)
        print(" > Quality:")
        print(f"     WER vs original transcript : {report.wer:.3f}  (lower is better)")
        print(f"     BLEU                       : {report.bleu:.3f}  (higher is better)")
        print(f"     similarity to original spk : {report.target_similarity:.3f}  (lower is better)")
        print(f"     similarity to donor voice  : {report.reference_similarity:.3f}  (higher is better)")
        print(f"     overall                    : {report.overall:.3f}")
    return 0


def _cmd_batch(args) -> int:
    from .batch import anonymize_directory
    from .pipeline import Anonymizer

    config = _config_from_args(args)
    anonymizer = Anonymizer(config)

    def progress(path, index, total):
        print(f" > [{index}/{total}] {path}")

    summary = anonymize_directory(
        args.input_dir,
        args.output_dir,
        reference=config.reference,
        anonymizer=anonymizer,
        resume=args.resume,
        manifest_path=args.manifest,
        on_progress=progress,
        gender=getattr(args, "gender", None),
        pool_language=getattr(args, "pool_language", None),
    )

    print(f" > Completed {summary['completed']}/{summary['total']} file(s)")
    if summary["manifest"]:
        print(f" > Manifest: {summary['manifest']}")
    if summary["failures"]:
        print(f" > {len(summary['failures'])} file(s) failed:", file=sys.stderr)
        for failure in summary["failures"]:
            print(f"     {failure['input_path']}: {failure['error']}", file=sys.stderr)
        return 1
    return 0


def _cmd_select(args) -> int:
    from .pipeline import Anonymizer

    config = _config_from_args(args)
    anonymizer = Anonymizer(config)
    result = anonymizer.select_reference(
        args.input, gender=args.gender, pool_language=args.pool_language
    )

    print(f" > Pool: {anonymizer.session.voice_pool}")
    print(f" > Chose speaker {result.speaker_id} out of {result.candidates} candidate clip(s)")
    print(f" > Mean similarity to the input: {result.speaker_similarity:.4f} (lower is further away)")
    print(f" > Conditioning on {len(result.clips)} clip(s):")
    for path, similarity in zip(result.clips, result.clip_similarities):
        print(f"     {similarity:+.4f}  {path}")

    if args.top:
        print(f" > Speakers ranked furthest-first (top {args.top}):")
        for rank, ranked in enumerate(result.ranked_speakers[: args.top], start=1):
            print(f"     {rank:>2}. {ranked.key:<24} {ranked.similarity:+.4f}")
    return 0


def _cmd_download(args) -> int:
    from .download import download_model

    download_model(dest=args.dest, force=args.force)
    return 0


def _cmd_config(args) -> int:
    import json

    config = _config_from_args(args)
    print(json.dumps(config.to_dict(), indent=2))
    return 0


COMMANDS = {
    "run": _cmd_run,
    "batch": _cmd_batch,
    "select": _cmd_select,
    "download-model": _cmd_download,
    "config": _cmd_config,
}


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except (FileNotFoundError, ValueError, NotADirectoryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
