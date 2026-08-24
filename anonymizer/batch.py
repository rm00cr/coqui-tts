"""Anonymize a folder of audio files, resumably.

Writes one output wav per input plus a manifest CSV. State is kept in a JSON checkpoint
next to the outputs so an interrupted run can be resumed with ``resume=True`` — the same
file-locking approach ``run_test_hdf5.py`` uses on the cluster, so several processes can
share one output directory.
"""

from __future__ import annotations

import csv
import fcntl
import json
import os
import time
from typing import Callable, Dict, List, Optional

from .audio import AUDIO_SUFFIXES
from .pipeline import Anonymizer
from .voices import ReferenceSpec

CHECKPOINT_NAME = "anonymize_checkpoint.json"
MANIFEST_NAME = "manifest.csv"


def find_audio_files(directory: str) -> List[str]:
    """All audio files directly inside `directory`, sorted."""
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"input directory not found: {directory}")
    return sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.lower().endswith(AUDIO_SUFFIXES)
    )


def load_checkpoint(path: str) -> Dict:
    if os.path.exists(path):
        try:
            with open(path) as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            # A checkpoint truncated by a crash should not abort the whole run.
            return {"completed": [], "rows": []}
    return {"completed": [], "rows": []}


def save_checkpoint(path: str, completed: List[str], rows: List[Dict], max_retries: int = 10) -> None:
    """Write the checkpoint under an exclusive file lock."""
    lock_path = path + ".lock"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    for attempt in range(max_retries):
        try:
            with open(lock_path, "w") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with open(path, "w") as fh:
                    json.dump({"completed": completed, "rows": rows}, fh, indent=2)
                break
        except (IOError, OSError):
            if attempt == max_retries - 1:
                raise
            time.sleep(0.1 * (2**attempt))  # exponential backoff

    try:
        os.unlink(lock_path)
    except OSError:
        pass


def write_manifest(path: str, rows: List[Dict]) -> Optional[str]:
    """Write the per-file results as CSV. Returns the path, or None if there are no rows."""
    if not rows:
        return None

    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def anonymize_directory(
    input_dir: str,
    output_dir: str,
    reference: ReferenceSpec = None,
    anonymizer: Optional[Anonymizer] = None,
    resume: bool = False,
    manifest_path: Optional[str] = None,
    on_progress: Optional[Callable[[str, int, int], None]] = None,
    **anonymize_kwargs,
) -> Dict:
    """Anonymize every audio file in `input_dir` into `output_dir`.

    Args:
        reference: donor voice, forwarded to `Anonymizer.anonymize`.
        anonymizer: reuse an existing one, so the checkpoints load once.
        resume: skip inputs already recorded in the checkpoint.
        manifest_path: defaults to `output_dir/manifest.csv`.
        on_progress: called with (input_path, index, total) before each file.
        **anonymize_kwargs: passed through (text, language, mode, denoise).

    Returns:
        A summary dict with the completed rows, any failures, and the manifest path.
    """
    anonymizer = anonymizer or Anonymizer()
    inputs = find_audio_files(input_dir)
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_path = os.path.join(output_dir, CHECKPOINT_NAME)
    manifest_path = manifest_path or os.path.join(output_dir, MANIFEST_NAME)

    checkpoint = load_checkpoint(checkpoint_path) if resume else {"completed": [], "rows": []}
    completed: List[str] = list(checkpoint["completed"])
    rows: List[Dict] = list(checkpoint["rows"])
    failures: List[Dict] = []

    for index, input_path in enumerate(inputs, start=1):
        name = os.path.basename(input_path)
        if resume and name in completed:
            continue

        if on_progress:
            on_progress(input_path, index, len(inputs))

        output_path = os.path.join(output_dir, f"{os.path.splitext(name)[0]}_anonymized.wav")
        try:
            result = anonymizer.anonymize(
                input_path, reference=reference, output_path=output_path, **anonymize_kwargs
            )
        except Exception as exc:  # keep going; one bad file must not sink the batch
            failures.append({"input_path": input_path, "error": f"{type(exc).__name__}: {exc}"})
            continue

        row = {"input_path": input_path}
        row.update(result.to_dict())
        rows.append(row)
        completed.append(name)
        save_checkpoint(checkpoint_path, completed, rows)

    written_manifest = write_manifest(manifest_path, rows)

    return {
        "total": len(inputs),
        "completed": len(rows),
        "skipped": len(inputs) - len(rows) - len(failures) if resume else 0,
        "failures": failures,
        "manifest": written_manifest,
        "output_dir": output_dir,
    }
