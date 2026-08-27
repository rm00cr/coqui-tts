"""Resolving the donor ("reference") voice.

The anonymizer needs two inputs: the *target* (the audio being anonymized, whose words
are preserved) and the *reference* (the donor voice the output will sound like). This
module turns whatever the user passed for the reference into the ``list[str]`` of wav
paths that ``XTTS.prep_batch`` expects — it averages the conditioning across all of them.

Two ways to supply donors:

* **directly** — a wav, a directory of wavs, or a list. `resolve_reference` handles it.
* **as a pool** — a directory tree or a CSV manifest describing many speakers, from
  which one donor is *chosen per target*. That is `VoicePool`; the choosing is in
  `anonymizer.selection`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Union

from .audio import AUDIO_SUFFIXES

ReferenceSpec = Union[str, Sequence[str], None]

#: CSV column names understood for each field, in priority order. A manifest only has to
#: name its audio-path column something recognizable; everything else is optional.
COLUMN_ALIASES = {
    "path": ("path", "file_path", "filepath", "audio_path", "audio", "wav", "wav_path", "filename", "file"),
    "speaker_id": ("speaker_id", "speaker", "spk", "spk_id", "speaker_name", "client_id"),
    "gender": ("gender", "predicted_gender", "sex"),
    "language": ("language", "lang", "locale"),
}

_GENDER_ALIASES = {"m": "male", "male": "male", "f": "female", "female": "female"}


class VoiceResolutionError(ValueError):
    """Raised when a reference voice cannot be resolved to usable audio files."""


def _audio_files_in(directory: str) -> List[str]:
    entries = sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.lower().endswith(AUDIO_SUFFIXES)
    )
    return entries


def normalize_gender(value: Optional[str]) -> Optional[str]:
    """Map m/M/male/FEMALE/... onto "male"/"female". Unknown values pass through lowercased."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text in ("nan", "none", "null", "unknown", ""):
        return None  # genuinely missing metadata; "other" is a real value and is kept
    return _GENDER_ALIASES.get(text, text)


@dataclass(frozen=True)
class VoiceClip:
    """One donor recording and what the pool knows about it."""

    path: str
    speaker_id: str
    gender: Optional[str] = None
    language: Optional[str] = None


class VoicePool:
    """A set of donor recordings grouped by speaker.

    Built from a directory tree (``pool/<speaker>/*.wav``) or a CSV manifest. The pool
    itself does no selection — it is the candidate set that `anonymizer.selection`
    ranks against a target.
    """

    def __init__(self, clips: Sequence[VoiceClip], source: Optional[str] = None):
        self.clips: List[VoiceClip] = list(clips)
        self.source = source

    def __len__(self) -> int:
        return len(self.clips)

    def __iter__(self) -> Iterator[VoiceClip]:
        return iter(self.clips)

    def __bool__(self) -> bool:
        return bool(self.clips)

    def __repr__(self) -> str:
        return f"VoicePool({len(self.clips)} clips, {len(self.speakers)} speakers, source={self.source!r})"

    @property
    def paths(self) -> List[str]:
        return [clip.path for clip in self.clips]

    @property
    def speakers(self) -> Dict[str, List[VoiceClip]]:
        """speaker_id -> its clips, insertion-ordered."""
        grouped: Dict[str, List[VoiceClip]] = {}
        for clip in self.clips:
            grouped.setdefault(clip.speaker_id, []).append(clip)
        return grouped

    def filter(self, gender: Optional[str] = None, language: Optional[str] = None) -> "VoicePool":
        """Narrow the pool. A filter value of None means "do not filter on this".

        Clips whose metadata does not record the field are dropped when that field is
        filtered on — a pool with no gender column cannot honour a gender filter, and
        silently ignoring it is how you end up conditioning on the wrong voices.
        """
        wanted_gender = normalize_gender(gender)
        wanted_language = language.lower() if language else None

        kept = [
            clip
            for clip in self.clips
            if (wanted_gender is None or clip.gender == wanted_gender)
            and (wanted_language is None or (clip.language or "").lower() == wanted_language)
        ]
        return VoicePool(kept, source=self.source)

    def clips_for(self, speaker_id: str) -> List[VoiceClip]:
        return self.speakers.get(speaker_id, [])

    # -- constructors -----------------------------------------------------------

    @classmethod
    def from_directory(cls, directory: str) -> "VoicePool":
        """Every audio file under `directory`, recursively.

        A file in a subdirectory belongs to the speaker named by that subdirectory
        (``pool/spk42/a.wav`` -> speaker ``spk42``); a file sitting directly in the pool
        root is its own speaker, named after the file.
        """
        if not os.path.isdir(directory):
            raise VoiceResolutionError(f"voice pool directory does not exist: {directory}")

        clips: List[VoiceClip] = []
        for root, _dirs, names in os.walk(directory):
            for name in sorted(names):
                if not name.lower().endswith(AUDIO_SUFFIXES):
                    continue
                path = os.path.join(root, name)
                relative = os.path.relpath(path, directory)
                parent = os.path.dirname(relative)
                speaker = parent.split(os.sep)[0] if parent else os.path.splitext(name)[0]
                clips.append(VoiceClip(path=path, speaker_id=speaker))

        if not clips:
            raise VoiceResolutionError(
                f"voice pool {directory} contains no audio files ({', '.join(AUDIO_SUFFIXES)})"
            )
        return cls(sorted(clips, key=lambda c: c.path), source=directory)

    @classmethod
    def from_csv(cls, csv_path: str, audio_root: Optional[str] = None) -> "VoicePool":
        """Read a manifest.

        One row per recording. The audio-path column may be named any of
        `COLUMN_ALIASES["path"]`; `speaker_id`, `gender` and `language` columns are
        picked up when present and are what make speaker-level and gender-matched
        selection possible.

        Relative paths resolve against `audio_root`, defaulting to the directory holding
        the CSV, so a manifest travels with its audio.
        """
        import csv as _csv

        if not os.path.isfile(csv_path):
            raise VoiceResolutionError(f"voice pool CSV not found: {csv_path}")

        root = audio_root or os.path.dirname(os.path.abspath(csv_path))

        with open(csv_path, newline="", encoding="utf-8-sig") as fh:
            sample = fh.read(8192)
            fh.seek(0)
            try:
                dialect = _csv.Sniffer().sniff(sample, delimiters=",;\t|")
            except _csv.Error:
                dialect = _csv.excel
            rows = list(_csv.DictReader(fh, dialect=dialect))

        if not rows:
            raise VoiceResolutionError(f"voice pool CSV {csv_path} has no rows")

        columns = {name.strip().lower(): name for name in rows[0] if name}
        chosen = {
            field: next((columns[a] for a in aliases if a in columns), None)
            for field, aliases in COLUMN_ALIASES.items()
        }
        if chosen["path"] is None:
            raise VoiceResolutionError(
                f"voice pool CSV {csv_path} has no audio-path column. Expected one of: "
                f"{', '.join(COLUMN_ALIASES['path'])}. Found: {', '.join(sorted(columns))}"
            )

        clips: List[VoiceClip] = []
        missing: List[str] = []
        for row in rows:
            raw_path = (row.get(chosen["path"]) or "").strip()
            if not raw_path:
                continue
            path = raw_path if os.path.isabs(raw_path) else os.path.join(root, raw_path)
            if not os.path.isfile(path):
                missing.append(path)
                continue
            speaker = (row.get(chosen["speaker_id"]) or "").strip() if chosen["speaker_id"] else ""
            clips.append(
                VoiceClip(
                    path=path,
                    speaker_id=speaker or _speaker_from_path(path),
                    gender=normalize_gender(row.get(chosen["gender"])) if chosen["gender"] else None,
                    language=(row.get(chosen["language"]) or "").strip().lower() or None
                    if chosen["language"]
                    else None,
                )
            )

        if not clips:
            shown = ", ".join(missing[:3])
            raise VoiceResolutionError(
                f"voice pool CSV {csv_path} yielded no usable audio. "
                f"{len(missing)} path(s) did not exist, e.g. {shown}. "
                f"Paths are resolved against {root}; pass audio_root= to change that."
            )
        if missing:
            print(
                f" > warning: {len(missing)} of {len(rows)} pool entries point at files "
                f"that do not exist (e.g. {missing[0]}); they were skipped"
            )
        return cls(clips, source=csv_path)

    @classmethod
    def load(cls, spec: str, audio_root: Optional[str] = None) -> "VoicePool":
        """Build from a CSV manifest or a directory, whichever `spec` names."""
        if os.path.isdir(spec):
            return cls.from_directory(spec)
        if os.path.isfile(spec):
            return cls.from_csv(spec, audio_root=audio_root)
        raise VoiceResolutionError(
            f"voice pool not found: {spec}. Expected a directory of donor audio or a CSV manifest."
        )


def _speaker_from_path(path: str) -> str:
    """Fallback speaker id for a manifest with no speaker column: the parent folder."""
    parent = os.path.basename(os.path.dirname(path))
    return parent or os.path.splitext(os.path.basename(path))[0]


def resolve_reference(reference: ReferenceSpec, voice_pool_dir: Optional[str] = None) -> List[str]:
    """Resolve a reference spec to a non-empty list of audio file paths.

    Accepts a single file path, a directory (every audio file in it is used, and the
    conditioning is averaged), a comma-separated string, or an explicit sequence of
    paths. When ``reference`` is None, falls back to ``voice_pool_dir`` if one is
    configured — this is the hook for a bundled default voice pool.

    Raises:
        VoiceResolutionError: if nothing usable could be resolved.
    """
    if reference is None:
        if voice_pool_dir:
            if not os.path.isdir(voice_pool_dir):
                raise VoiceResolutionError(
                    f"voice_pool_dir does not exist: {voice_pool_dir}"
                )
            pool = _audio_files_in(voice_pool_dir)
            if not pool:
                raise VoiceResolutionError(
                    f"voice pool {voice_pool_dir} contains no audio files "
                    f"({', '.join(AUDIO_SUFFIXES)})"
                )
            return pool
        raise VoiceResolutionError(
            "No reference voice given. Pass --reference with a wav file or a directory "
            "of wavs (the donor voice the output should sound like), set voice_pool in "
            "your config to pick a donor automatically, or set voice_pool_dir."
        )

    if isinstance(reference, str):
        candidates = [part.strip() for part in reference.split(",") if part.strip()]
    else:
        candidates = [str(part) for part in reference]

    if not candidates:
        raise VoiceResolutionError("Reference voice resolved to an empty list.")

    resolved: List[str] = []
    for candidate in candidates:
        if os.path.isdir(candidate):
            files = _audio_files_in(candidate)
            if not files:
                raise VoiceResolutionError(
                    f"Reference directory {candidate} contains no audio files "
                    f"({', '.join(AUDIO_SUFFIXES)})"
                )
            resolved.extend(files)
        elif os.path.isfile(candidate):
            resolved.append(candidate)
        else:
            raise VoiceResolutionError(f"Reference voice not found: {candidate}")

    return resolved
