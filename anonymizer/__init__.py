"""Speaker anonymization for speech, built on XTTS v2.

Give it a recording and a donor voice; it returns the same words spoken in the donor's
voice, so the original speaker's identity is not recoverable from the audio.

    from anonymizer import Anonymizer

    anon = Anonymizer()
    result = anon.anonymize("interview.wav", reference="donor.wav")
    anon.save(result, "interview_anonymized.wav")

Nothing loads at import time; the checkpoints load on the first `anonymize` call and are
reused for the life of the object.
"""

from .config import MODES, AnonymizerConfig
from .pipeline import AnonymizationResult, Anonymizer, anonymize
from .session import AnonymizerSession
from .voices import VoiceResolutionError, resolve_reference

__all__ = [
    "Anonymizer",
    "AnonymizerConfig",
    "AnonymizerSession",
    "AnonymizationResult",
    "VoiceResolutionError",
    "anonymize",
    "resolve_reference",
    "MODES",
]

__version__ = "0.1.0"
