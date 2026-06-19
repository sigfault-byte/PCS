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

    speaker_evidence_ratio: float = 0.0
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
            speaker_evidence_ratio=data.get("speaker_evidence_ratio", 0.0),
            flags=SegmentFlag(data.get("flags", 0)),
        )


@dataclass(frozen=True)
class PersonIdentity:
    id: str | None
    name: str
    role: str | None
    kind: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PersonIdentity":
        return cls(
            id=data.get("id"),
            name=data.get("name", ""),
            role=data.get("role"),
            kind=data.get("kind", "raw_per"),
        )


@dataclass(frozen=True)
class SpeakerIdentityEvidence:
    source: str
    eligible_for_cluster_majority: bool
    person: PersonIdentity
    source_turn_id: int
    target_turn_id: int
    source_speaker_id: str | None
    target_speaker_id: str | None
    speaker_raw: str
    speaker_normalized: str
    match_score: float
    is_known_person: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpeakerIdentityEvidence":
        return cls(
            source=data.get("source", ""),
            eligible_for_cluster_majority=bool(
                data.get("eligible_for_cluster_majority", False)
            ),
            person=PersonIdentity.from_dict(data.get("person", {})),
            source_turn_id=data.get("source_turn_id", 0),
            target_turn_id=data.get("target_turn_id", 0),
            source_speaker_id=data.get("source_speaker_id"),
            target_speaker_id=data.get("target_speaker_id"),
            speaker_raw=data.get("speaker_raw", ""),
            speaker_normalized=data.get("speaker_normalized", ""),
            match_score=float(data.get("match_score", 0.0)),
            is_known_person=bool(data.get("is_known_person", False)),
        )


@dataclass
class TurnAnalysis:
    turn_id: int

    keywords: list[str] = field(default_factory=list)
    current_speaker: PersonIdentity | None = None
    current_speaker_source: str | None = None
    current_speaker_purity: float | None = None
    speaker_identity_evidence: list[SpeakerIdentityEvidence] = field(
        default_factory=list
    )
    mentioned_persons: list[PersonIdentity] = field(default_factory=list)
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
            current_speaker=(
                PersonIdentity.from_dict(data["current_speaker"])
                if data.get("current_speaker") is not None
                else None
            ),
            current_speaker_source=data.get("current_speaker_source"),
            current_speaker_purity=data.get("current_speaker_purity", None),
            speaker_identity_evidence=[
                SpeakerIdentityEvidence.from_dict(item)
                for item in data.get("speaker_identity_evidence", [])
            ],
            mentioned_persons=[
                PersonIdentity.from_dict(item)
                for item in data.get("mentioned_persons", [])
            ],
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
