from dataclasses import dataclass, field
from typing import Any

from .collapse_diarization import CollapsedDiarizationSegment
from .flags import SegmentFlag
from .ids import require_positive_int_id
from .time import TimeRange


@dataclass
class DiarizationEngine:
    """Runtime settings used by the diarization stage."""

    name: str = "pyannote"
    model: str | None = None
    device: str | None = None
    version: str | None = None
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiarizationEngine":
        return cls(
            name=data.get("name", "pyannote"),
            model=data.get("model"),
            device=data.get("device"),
            version=data.get("version"),
            options=dict(data["options"]),
        )


@dataclass
class DiarizationRawSegment:
    """One speaker-labeled interval emitted by diarization."""

    segment_id: int
    time: TimeRange
    speaker_id: str
    flags: SegmentFlag = SegmentFlag.NONE
    overlap_speaker_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiarizationRawSegment":
        return cls(
            segment_id=require_positive_int_id(data["segment_id"], "segment_id"),
            time=TimeRange.from_dict(data["time"]),
            speaker_id=data["speaker_id"],
            flags=SegmentFlag(data.get("flags", 0)),
            overlap_speaker_ids=list(data.get("overlap_speaker_ids", [])),
        )


@dataclass
class DiarizationOverlapRegion:
    """Time interval where two or more speakers are active."""

    region_id: int
    time: TimeRange
    speaker_ids: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiarizationOverlapRegion":
        return cls(
            region_id=require_positive_int_id(data["region_id"], "region_id"),
            time=TimeRange.from_dict(data["time"]),
            speaker_ids=list(data.get("speaker_ids", [])),
        )


@dataclass
class DiarizationArtifacts:
    """Paths to files produced by the diarization stage."""

    raw_txt_path: str | None = None
    collapsed_txt_path: str | None = None
    embeddings_npz_path: str | None = None
    centroids_npz_path: str | None = None
    embedding_model: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiarizationArtifacts":
        return cls(
            raw_txt_path=data.get("raw_txt_path"),
            collapsed_txt_path=data.get("collapsed_txt_path"),
            embeddings_npz_path=(
                data.get("embeddings_npz_path") or data.get("embeddings_npy_path")
            ),
            centroids_npz_path=(
                data.get("centroids_npz_path") or data.get("centroids_npy_path")
            ),
            embedding_model=data.get("embedding_model"),
        )


@dataclass
class DiarizationSection:
    """Canonical diarization output stored at document.diarization."""

    engine: DiarizationEngine = field(default_factory=DiarizationEngine)
    raw_segments: list[DiarizationRawSegment] = field(default_factory=list)
    overlap_regions: list[DiarizationOverlapRegion] = field(default_factory=list)
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
            overlap_regions=[
                DiarizationOverlapRegion.from_dict(item)
                for item in data.get("overlap_regions", [])
            ],
            collapsed_segments=[
                CollapsedDiarizationSegment.from_dict(item)
                for item in data.get("collapsed_segments", [])
            ],
            speakers_count=data.get("speakers_count"),
            artifacts=DiarizationArtifacts.from_dict(data.get("artifacts", {})),
        )
