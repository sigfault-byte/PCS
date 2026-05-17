from dataclasses import dataclass, field
from typing import Any

from .ids import require_positive_int_id, require_positive_int_ids
from .time import TimeRange


@dataclass
class CollapsedDiarizationSegment:
    """Consecutive diarization intervals merged for one speaker."""

    segment_id: int
    time: TimeRange
    speaker_id: str
    source_diarization_segment_ids: list[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CollapsedDiarizationSegment":
        return cls(
            segment_id=require_positive_int_id(data["segment_id"], "segment_id"),
            time=TimeRange.from_dict(data["time"]),
            speaker_id=data["speaker_id"],
            source_diarization_segment_ids=require_positive_int_ids(
                data.get("source_diarization_segment_ids", []),
                "source_diarization_segment_ids",
            ),
        )
