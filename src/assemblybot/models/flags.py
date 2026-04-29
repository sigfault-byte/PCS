from enum import IntFlag, auto


class SegmentFlag(IntFlag):
    """Bitmask flags attached to segments requiring downstream attention."""

    NONE = 0

    # General quality
    SHORT_SEGMENT = auto()
    NEEDS_REVIEW = auto()
    GIBBERISH = auto()
    NON_FRENCH = auto()

    # VAD alignment
    OUTSIDE_VAD = auto()
    PARTIAL_VAD_OVERLAP = auto()
    INSIDE_VAD_GAP = auto()

    # Whisper quality
    LOW_WHISPER_CONFIDENCE = auto()
    HIGH_NO_SPEECH_PROB = auto()
    HIGH_COMPRESSION_RATIO = auto()

    # Diarization quality
    DIARIZATION_OVERLAP = auto()
    MULTI_SPEAKER_CANDIDATE = auto()
    SPEAKER_CHANGE_NEARBY = auto()

    # Merge integrity
    ORPHAN_TRANSCRIPT = auto()
    ORPHAN_DIARIZATION = auto()


def flags_to_list(flags: int | SegmentFlag) -> list[str | None]:
    """Return active flag names; excludes NONE."""

    return [f.name for f in SegmentFlag if f & flags]


def has_flag(flags: int, flag: SegmentFlag) -> bool:
    return bool(flags & flag)
