from dataclasses import dataclass, field

from .time import TimeRange


@dataclass
class DiarizationEngine:
    name: str = "pyannote"
    model: str | None = None
    device: str | None = None


@dataclass
class DiarizationRawSegment:
    segment_id: str
    time: TimeRange
    speaker_id: str
    confidence: float | None = None


@dataclass
class DiarizationSection:
    engine: DiarizationEngine = field(default_factory=DiarizationEngine)
    speakers_count: int | None = None
    raw_segments: list[DiarizationRawSegment] = field(default_factory=list)
    speaker_embeddings: list[list[float]] = field(default_factory=list)
