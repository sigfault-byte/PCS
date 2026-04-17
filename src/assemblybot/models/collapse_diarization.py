from dataclasses import dataclass, field

from .time import TimeRange


@dataclass
class CollapsedDiarizationSegment:
    segment_id: str
    time: TimeRange
    speaker_id: str
    source_diarization_segment_ids: list[str] = field(default_factory=list)
    segments_count: int = 0
