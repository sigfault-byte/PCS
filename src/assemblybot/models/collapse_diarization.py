from dataclasses import dataclass, field
from typing import Any

from .time import TimeRange


@dataclass
class CollapsedDiarizationSegment:
    segment_id: str
    time: TimeRange
    speaker_id: str
    source_diarization_segment_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CollapsedDiarizationSegment":
        return cls(
            segment_id=data["segment_id"],
            time=TimeRange.from_dict(data["time"]),
            speaker_id=data["speaker_id"],
            source_diarization_segment_ids=list(
                data.get("source_diarization_segment_ids", [])
            ),
        )
