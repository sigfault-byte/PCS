from dataclasses import dataclass, field
from typing import Any

from src.assemblybot.models.flags import SegmentFlag

from .time import TimeRange


@dataclass
class SpeakerInfo:
    """Resolved speaker metadata for a final segment."""

    speaker_id: str | None = None
    speaker_label: str | None = None
    speaker_label_source: str | None = None
    confidence: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpeakerInfo":
        return cls(
            speaker_id=data.get("speaker_id"),
            speaker_label=data.get("speaker_label"),
            speaker_label_source=data.get("speaker_label_source"),
            confidence=data.get("confidence"),
        )


@dataclass
class TextInfo:
    """Text payload for a final segment."""

    raw: str
    normalized: str | None = None
    language: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextInfo":
        return cls(
            raw=data.get("raw", ""),
            normalized=data.get("normalized"),
            language=data.get("language"),
        )


@dataclass
class Provenance:
    """Source segment ids used to build a final segment."""

    transcript_segment_ids: list[str] = field(default_factory=list)
    diarization_segment_ids: list[str] = field(default_factory=list)
    transcript_token_start_id: int | None = None
    transcript_token_end_id: int | None = None
    stage_created_by: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Provenance":
        return cls(
            transcript_segment_ids=list(data.get("transcript_segment_ids", [])),
            diarization_segment_ids=list(data.get("diarization_segment_ids", [])),
            transcript_token_start_id=data.get("transcript_token_start_id"),
            transcript_token_end_id=data.get("transcript_token_end_id"),
            stage_created_by=data.get("stage_created_by"),
        )


@dataclass
class FinalSegment:
    """Merged user-facing segment built from transcript and diarization data."""

    segment_id: str
    time: TimeRange
    speaker: SpeakerInfo
    text: TextInfo
    flags: SegmentFlag = SegmentFlag.NONE
    entities: list[dict[str, Any]] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    provenance: Provenance = field(default_factory=Provenance)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FinalSegment":
        return cls(
            segment_id=data["segment_id"],
            time=TimeRange.from_dict(data["time"]),
            speaker=SpeakerInfo.from_dict(data.get("speaker", {})),
            text=TextInfo.from_dict(data.get("text", {})),
            flags=SegmentFlag(data.get("flags", 0)),
            entities=list(data.get("entities", [])),
            keywords=list(data.get("keywords", [])),
            provenance=Provenance.from_dict(data.get("provenance", {})),
        )
