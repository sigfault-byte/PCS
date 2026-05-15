from enum import IntFlag


class SegmentFlag(IntFlag):
    """Bitmask flags attached to segments requiring downstream attention.

    Flags describe why a transcript, diarization, or final merged segment is
    suspicious. They are diagnostic signals, not deletion decisions: downstream
    stages can combine several flags to decide whether to review, repair, merge,
    or discard a segment.
    """

    NONE = 0

    # General quality: bits 0-9
    # Text is too dense for a very short segment: word or character rate exceeds
    # the configured plausible speech threshold.
    IMPOSSIBLE_SPEECH_RATE = 1 << 0
    # UTF-8 byte rate is too high for the segment duration, often indicating
    # repeated text, symbols, or another dense transcript artifact.
    INFORMATION_RATE_TOO_HIGH = 1 << 1
    # Manual or downstream catch-all marker for a segment that should be
    # inspected even if no more specific flag applies.
    NEEDS_REVIEW = 1 << 2
    # Intended for text that is nonsensical after transcription or normalization.
    GIBBERISH = 1 << 3
    # Intended for text detected as not French in a French-focused pipeline.
    NON_FRENCH = 1 << 4
    # Audio is mostly quiet, but has a short transient event that may have caused
    # Whisper to hallucinate speech.
    MOSTLY_SILENCE_WITH_SHORT_EVENT = 1 << 5

    # VAD alignment: bits 10-19
    # Whisper segment has no overlap with any VAD speech interval.
    OUTSIDE_VAD = 1 << 10
    # Whisper segment overlaps VAD, but coverage is below the partial-coverage
    # threshold.
    PARTIAL_VAD_OVERLAP = 1 << 11
    # VAD coverage inside the Whisper segment has an internal gap longer than
    # the configured threshold.
    DISCONTIGUOUS_VAD_COVERAGE = 1 << 12
    # Long Whisper segment has too little total VAD coverage.
    LONG_WHISPER_SEGMENT_LOW_VAD_COVERAGE = 1 << 13

    # Whisper quality: bits 20-29
    # Whisper average log probability is below the configured confidence floor.
    LOW_WHISPER_CONFIDENCE = 1 << 20
    # Whisper no-speech probability is above the configured threshold.
    HIGH_NO_SPEECH_PROB = 1 << 21
    # Whisper compression ratio is above the configured threshold, a common sign
    # of repetitive or degenerate decoder output.
    HIGH_COMPRESSION_RATIO = 1 << 22
    # Segment duration is long, but the text has too few words or characters.
    LONG_DURATION_SHORT_TEXT = 1 << 23

    # Diarization quality: bits 30-39
    # Diarization segment intersects a diarization overlap region.
    DIARIZATION_OVERLAP = 1 << 30
    # Segment may contain more than one speaker. In the diarization stage this
    # means an overlap region has at least two speakers; in the Whisper audit it
    # means the Whisper segment intersects any diarization overlap region.
    MULTI_SPEAKER_CANDIDATE = 1 << 31
    # Intended for segments close to a speaker boundary where attribution may be
    # unstable.
    SPEAKER_CHANGE_NEARBY = 1 << 32

    # Merge integrity: bits 40-49
    # Intended for transcript content that could not be matched to diarization.
    ORPHAN_TRANSCRIPT = 1 << 40
    # Intended for diarization speech that could not be matched to transcript
    # content.
    ORPHAN_DIARIZATION = 1 << 41


def flags_to_list(flags: int | SegmentFlag) -> list[str | None]:
    """Return active flag names; excludes NONE."""

    return [f.name for f in SegmentFlag if f & flags]


def has_flag(flags: int, flag: SegmentFlag) -> bool:
    return bool(flags & flag)
