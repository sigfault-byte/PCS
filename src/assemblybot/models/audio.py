from dataclasses import asdict, dataclass


@dataclass
class SegmentAudioStats:
    vad_coverage: float
    diarization_overlap_seconds: float
    diarization_overlap_region_count: int
    frame_count: int
    db_mean: float | None
    db_p10: float | None
    db_p50: float | None
    db_p90: float | None
    db_delta_p95: float | None
    rms_mean: float | None
    zcr_mean: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


@dataclass
class TurnAudioQualityMetrics:
    db_mean: float | None
    db_p10: float | None
    db_p50: float | None
    db_p90: float | None

    rms_mean: float | None

    zcr_mean: float | None

    spectral_centroid_mean: float | None

    spectral_rolloff_mean: float | None

    speech_rate: float | None

    information_rate: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)
