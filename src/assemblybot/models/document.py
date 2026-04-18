from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .diarization import (
    DiarizationSection,
)
from .final_segment import FinalSegment
from .transcript import (
    TranscriptSection,
)


class StageArtifactStatus:
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SourceInfo:
    source_id: str
    input_path: str
    input_filename: str
    media_type: str = "audio"
    language_expected: str = "fr"
    duration_seconds: float | None = None
    file_sha256: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceInfo":
        return cls(
            source_id=data.get("source_id", ""),
            input_path=data.get("input_path", ""),
            input_filename=data.get("input_filename", ""),
            media_type=data.get("media_type", "audio"),
            language_expected=data.get("language_expected", "fr"),
            duration_seconds=data.get("duration_seconds"),
            file_sha256=data.get("file_sha256"),
        )


@dataclass
class StageArtifact:
    status: str = "not_started"
    output_path: str | None = None
    completed_at: str | None = None
    error_message: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StageArtifact":
        return cls(
            status=data.get("status", "not_started"),
            output_path=data.get("output_path"),
            completed_at=data.get("completed_at"),
            error_message=data.get("error_message"),
        )


@dataclass
class StageOutputs:
    transcription: StageArtifact = field(default_factory=StageArtifact)
    diarization: StageArtifact = field(default_factory=StageArtifact)
    merge: StageArtifact = field(default_factory=StageArtifact)
    enrichment: StageArtifact = field(default_factory=StageArtifact)
    export: StageArtifact = field(default_factory=StageArtifact)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StageOutputs":
        return cls(
            transcription=StageArtifact.from_dict(data.get("transcription", {})),
            diarization=StageArtifact.from_dict(data.get("diarization", {})),
            merge=StageArtifact.from_dict(data.get("merge", {})),
            enrichment=StageArtifact.from_dict(data.get("enrichment", {})),
            export=StageArtifact.from_dict(data.get("export", {})),
        )


@dataclass
class PipelineInfo:
    created_at: str
    updated_at: str
    stage_outputs: StageOutputs = field(default_factory=StageOutputs)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineInfo":
        return cls(
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            stage_outputs=StageOutputs.from_dict(data.get("stage_outputs", {})),
        )


@dataclass
class CanonicalDocument:
    schema_version: str
    source: SourceInfo
    pipeline: PipelineInfo
    transcript: TranscriptSection = field(default_factory=TranscriptSection)
    diarization: DiarizationSection = field(default_factory=DiarizationSection)
    segments: list[FinalSegment] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CanonicalDocument":
        return cls(
            schema_version=data.get("schema_version", "0.1.0"),
            source=SourceInfo.from_dict(data.get("source", {})),
            pipeline=PipelineInfo.from_dict(data.get("pipeline", {})),
            transcript=TranscriptSection.from_dict(data.get("transcript", {})),
            diarization=DiarizationSection.from_dict(data.get("diarization", {})),
            segments=[
                FinalSegment.from_dict(item) for item in data.get("segments", [])
            ],
        )
