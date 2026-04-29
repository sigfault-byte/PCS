from dataclasses import dataclass, field
from typing import Any

from .collapse_diarization import CollapsedDiarizationSegment
from .flags import SegmentFlag
from .time import TimeRange


@dataclass
class DiarizationEngine:
    name: str = "pyannote"
    model: str | None = None
    device: str | None = None
    version: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiarizationEngine":
        return cls(
            name=data.get("name", "pyannote"),
            model=data.get("model"),
            device=data.get("device"),
            version=data.get("version"),
        )


@dataclass
class DiarizationRawSegment:
    segment_id: str
    time: TimeRange
    speaker_id: str
    flags: SegmentFlag = SegmentFlag.NONE

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiarizationRawSegment":
        return cls(
            segment_id=data["segment_id"],
            time=TimeRange.from_dict(data["time"]),
            speaker_id=data["speaker_id"],
        )


@dataclass
class DiarizationArtifacts:
    raw_txt_path: str | None = None
    collapsed_txt_path: str | None = None
    embeddings_npy_path: str | None = None
    centroids_npy_path: str | None = None
    embedding_model: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiarizationArtifacts":
        return cls(
            raw_txt_path=data.get("raw_txt_path"),
            collapsed_txt_path=data.get("collapsed_txt_path"),
            embeddings_npy_path=data.get("embeddings_npy_path"),
            centroids_npy_path=data.get("centroids_npy_path"),
            embedding_model=data.get("embedding_model"),
        )


@dataclass
class DiarizationSection:
    engine: DiarizationEngine = field(default_factory=DiarizationEngine)
    raw_segments: list[DiarizationRawSegment] = field(default_factory=list)
    collapsed_segments: list[CollapsedDiarizationSegment] = field(default_factory=list)
    speakers_count: int | None = None
    artifacts: DiarizationArtifacts = field(default_factory=DiarizationArtifacts)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiarizationSection":
        return cls(
            engine=DiarizationEngine.from_dict(data.get("engine", {})),
            raw_segments=[
                DiarizationRawSegment.from_dict(item)
                for item in data.get("raw_segments", [])
            ],
            collapsed_segments=[
                CollapsedDiarizationSegment.from_dict(item)
                for item in data.get("collapsed_segments", [])
            ],
            speakers_count=data.get("speakers_count"),
            artifacts=DiarizationArtifacts.from_dict(data.get("artifacts", {})),
        )
