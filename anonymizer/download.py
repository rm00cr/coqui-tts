"""Fetch the XTTS v2 checkpoint files the anonymizer runs on.

About 2 GB in total, downloaded once. The model itself is Coqui's XTTS v2; this project
adds the anonymization procedure on top of it.
"""

from __future__ import annotations

import os
from typing import List, Optional

BASE_URL = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main"

#: filename -> download URL. All five are required by `AnonymizerSession`.
CHECKPOINT_FILES = {
    "config.json": f"{BASE_URL}/config.json",
    "vocab.json": f"{BASE_URL}/vocab.json",
    "mel_stats.pth": f"{BASE_URL}/mel_stats.pth",
    "dvae.pth": f"{BASE_URL}/dvae.pth",
    "model.pth": f"{BASE_URL}/model.pth",
}

DEFAULT_DEST = "./XTTS_v2.0_original_model_files"


def missing_files(dest: str) -> List[str]:
    """Names of the checkpoint files not yet present in `dest`."""
    return [name for name in CHECKPOINT_FILES if not os.path.isfile(os.path.join(dest, name))]


def download_model(dest: Optional[str] = None, force: bool = False, progress_bar: bool = True) -> str:
    """Download any missing XTTS v2 checkpoint files into `dest`.

    Args:
        dest: target directory. Defaults to $XTTS_MODEL_DIR, else ./XTTS_v2.0_original_model_files.
        force: re-download files that already exist.
        progress_bar: show per-file download progress.

    Returns:
        The directory the files are in.
    """
    from TTS.utils.manage import ModelManager

    dest = dest or os.getenv("XTTS_MODEL_DIR") or DEFAULT_DEST
    os.makedirs(dest, exist_ok=True)

    wanted = list(CHECKPOINT_FILES) if force else missing_files(dest)
    if not wanted:
        print(f" > XTTS v2 checkpoint files already present in {dest}")
        return dest

    print(f" > Downloading {len(wanted)} XTTS v2 file(s) into {dest}: {', '.join(wanted)}")
    # One call per file so a failure names the file that failed.
    for name in wanted:
        print(f"   - {name}")
        ModelManager._download_model_files([CHECKPOINT_FILES[name]], dest, progress_bar=progress_bar)

    still_missing = missing_files(dest)
    if still_missing:
        raise RuntimeError(
            f"download finished but these files are still missing from {dest}: "
            f"{', '.join(still_missing)}"
        )

    print(f" > Done. Point the anonymizer at it with:  export XTTS_MODEL_DIR={os.path.abspath(dest)}")
    return dest
