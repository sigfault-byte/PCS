from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.assemblybot.models.audio import TurnAudioQualityMetrics
from src.assemblybot.models.time import TimeRange

from .flags import SegmentFlag


@dataclass
class Turn:
    turn_id: int
    audio_time: TimeRange
    text: str

    speaker_id: str | None
    speaker_confidence: float

    transcript_segment_ids: list[int]
    diarization_segment_ids: list[int]

    flags: SegmentFlag = SegmentFlag.NONE

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Turn":
        return cls(
            turn_id=data.get("turn_id", 0),
            audio_time=TimeRange.from_dict(data["audio_time"]),
            text=data.get("text", ""),
            speaker_id=data.get("speaker_id"),
            speaker_confidence=data.get("speaker_confidence", 0.0),
            transcript_segment_ids=data.get("transcript_segment_ids", []),
            diarization_segment_ids=data.get("diarization_segment_ids", []),
            flags=SegmentFlag(data.get("flags", 0)),
        )


@dataclass
class TurnAnalysis:
    turn_id: int

    keywords: list[str] = field(default_factory=list)
    persons: list[str] = field(default_factory=list)
    person_purity: float | None = None
    organizations: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)

    # maybe a link to a npz with the data instead of the json
    embedding_id: int | None = None
    audio_audit: TurnAudioQualityMetrics | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TurnAnalysis":
        audio_audit_data = data.get("audio_audit")

        return cls(
            turn_id=data.get("turn_id", 0),
            keywords=data.get("keywords", []),
            persons=data.get("persons", []),
            person_purity=data.get("person_purity", None),
            organizations=data.get("organizations", []),
            locations=data.get("locations", []),
            embedding_id=data.get("embedding_id"),
            audio_audit=(
                TurnAudioQualityMetrics(**audio_audit_data)  # shorthand unpacking
                if audio_audit_data is not None
                else None
            ),
        )


@dataclass
class TurnDocument:
    turns: list[Turn]
    turns_analysis: list[TurnAnalysis]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TurnDocument":
        return cls(
            turns=[Turn.from_dict(item) for item in data.get("turns", [])],
            turns_analysis=[
                TurnAnalysis.from_dict(item) for item in data.get("turns_analysis", [])
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
