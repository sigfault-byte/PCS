from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AuditThresholds:
    vad_partial_coverage: float
    vad_long_segment_seconds: float
    vad_long_segment_min_coverage: float
    vad_internal_gap_seconds: float
    low_avg_logprob: float  # kinda useless with this metric
    high_no_speech_prob: float
    high_compression_ratio: float
    short_segment_seconds: float
    long_short_text_seconds: float
    long_short_text_min_words: int
    long_short_text_min_chars: int
    words_per_second: float
    chars_per_second: float
    bytes_per_second: float
    silence_event_max_seconds: float
    silence_event_median_db: float
    silence_event_db_delta_p95: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)
