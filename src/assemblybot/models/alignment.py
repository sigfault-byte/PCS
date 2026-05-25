from dataclasses import dataclass, field
from typing import Any


@dataclass
class TranscriptDiarizationMatch:
    """Association between one transcript segment and overlapping diarization segments."""

    transcript_segment_id: int

    diarization_segment_ids: list[int] = field(default_factory=list)
    speaker_ids: list[str] = field(default_factory=list)

    probable_speaker_id: str | None = None

    # Raw overlap evidence.
    speaker_overlap_seconds: dict[str, float] = field(default_factory=dict)
    speaker_longest_overlap_seconds: dict[str, float] = field(default_factory=dict)
    speaker_overlap_segment_count: dict[str, int] = field(default_factory=dict)

    # Weighted speaker election evidence.
    speaker_evidence_score: dict[str, float] = field(default_factory=dict)

    total_overlap_seconds: float = 0.0
    winning_overlap_seconds: float = 0.0
    winning_evidence_score: float = 0.0

    speaker_confidence: float | None = None

    flags: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptDiarizationMatch":
        return cls(
            transcript_segment_id=data.get("transcript_segment_id", 0),
            diarization_segment_ids=data.get("diarization_segment_ids", []),
            speaker_ids=data.get("speaker_ids", []),
            probable_speaker_id=data.get("probable_speaker_id"),
            speaker_overlap_seconds=data.get("speaker_overlap_seconds", {}),
            speaker_longest_overlap_seconds=data.get(
                "speaker_longest_overlap_seconds", {}
            ),
            speaker_overlap_segment_count=data.get("speaker_overlap_segment_count", {}),
            speaker_evidence_score=data.get("speaker_evidence_score", {}),
            total_overlap_seconds=data.get("total_overlap_seconds", 0.0),
            winning_overlap_seconds=data.get("winning_overlap_seconds", 0.0),
            winning_evidence_score=data.get("winning_evidence_score", 0.0),
            speaker_confidence=data.get("speaker_confidence"),
            flags=data.get("flags", 0),
        )


@dataclass
class AlignmentSection:
    """Cross-modal alignment outputs stored at document.alignment."""

    transcript_diarization_matches: list[TranscriptDiarizationMatch] = field(
        default_factory=list
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AlignmentSection":
        return cls(
            transcript_diarization_matches=[
                TranscriptDiarizationMatch.from_dict(item)
                for item in data.get("transcript_diarization_matches", [])
            ],
        )
