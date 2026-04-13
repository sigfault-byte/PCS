from dataclasses import dataclass, field
from typing import Any

from .time import TimeRange


@dataclass
class SpeakerInfo:
    speaker_id: str | None = None
    speaker_label: str | None = None
    speaker_label_source: str | None = None
    confidence: float | None = None


@dataclass
class TextInfo:
    raw: str
    normalized: str | None = None
    language: str | None = None


@dataclass
class Provenance:
    transcript_segment_ids: list[str] = field(default_factory=list)
    diarization_segment_ids: list[str] = field(default_factory=list)
    stage_created_by: str | None = None


@dataclass
class FinalSegment:
    segment_id: str
    time: TimeRange
    speaker: SpeakerInfo
    text: TextInfo
    flags: int = 0
    entities: list[dict[str, Any]] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    provenance: Provenance = field(default_factory=Provenance)
