from dataclasses import dataclass, field

from .collapse_diarization import CollapsedDiarizationSegment
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
    segments_count: int = 0
    raw_segments: list[DiarizationRawSegment] = field(default_factory=list)

    collapsed_segments: list[CollapsedDiarizationSegment] = field(default_factory=list)
    collapsed_segments_count: int = 0

    artifacts: dict[str, str | None] = field(default_factory=dict)
