from dataclasses import asdict, dataclass, field
from typing import Any

from .diarization import DiarizationSection
from .flags import flags_to_list
from .segments import FinalSegment
from .transcript import TranscriptSection


@dataclass
class SourceInfo:
    input_path: str
    input_filename: str
    media_type: str = "audio"
    language_expected: str = "fr"
    duration_seconds: float | None = None


@dataclass
class StageOutputs:
    transcription: str | None = None
    diarization: str | None = None
    merge: str | None = None
    enrichment: str | None = None
    export: str | None = None


@dataclass
class PipelineInfo:
    created_at: str
    updated_at: str
    stages_completed: list[str] = field(default_factory=list)
    stage_outputs: StageOutputs = field(default_factory=StageOutputs)


@dataclass
class CanonicalDocument:
    schema_version: str
    source: SourceInfo
    pipeline: PipelineInfo
    transcript: TranscriptSection = field(default_factory=TranscriptSection)
    diarization: DiarizationSection = field(default_factory=DiarizationSection)
    segments: list[FinalSegment] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)

        # future-proof hook
        data["schema_version"] = self.schema_version

        # enrich for debug / readability
        for seg in data.get("segments", []):
            flags = seg.get("flags", 0)
            seg["flags_readable"] = flags_to_list(flags)

        return data
