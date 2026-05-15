from assemblybot.models.audit_threshold import AuditThresholds

AUDIO_AUDIT_FEATURES = (
    "frame_center_seconds",
    "rms",
    "db",
    "zcr",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_flatness",
    "db_rolling_median",
    "db_delta",
)

DEFAULT_AUDIT_THRESHOLD = AuditThresholds(
    vad_partial_coverage=0.80,
    vad_long_segment_seconds=10.0,
    vad_long_segment_min_coverage=0.60,
    vad_internal_gap_seconds=0.75,
    low_avg_logprob=-1.0,  # kinda useless threshold
    high_no_speech_prob=0.6,
    high_compression_ratio=2.8,
    short_segment_seconds=0.40,
    long_short_text_seconds=8.0,
    long_short_text_min_words=4,
    long_short_text_min_chars=25,
    words_per_second=8.0,
    chars_per_second=45.0,
    bytes_per_second=60,
    silence_event_max_seconds=3.0,
    silence_event_median_db=-55.0,
    silence_event_db_delta_p95=15.0,
)
