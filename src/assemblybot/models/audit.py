from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass
class AudioAuditSource:
    audio_path: str
    audio_filename: str
    duration_seconds: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AudioAuditSource":
        return cls(
            audio_path=data["audio_path"],
            audio_filename=data["audio_filename"],
            duration_seconds=float(data["duration_seconds"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AudioAuditParameters:
    target_sample_rate: int
    sample_rate: int | float
    frame_length: int
    hop_length: int
    frame_duration_seconds: float
    hop_duration_seconds: float
    time_reference: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AudioAuditParameters":
        return cls(
            target_sample_rate=int(data["target_sample_rate"]),
            sample_rate=int(data["sample_rate"]),
            frame_length=int(data["frame_length"]),
            hop_length=int(data["hop_length"]),
            frame_duration_seconds=float(data["frame_duration_seconds"]),
            hop_duration_seconds=float(data["hop_duration_seconds"]),
            time_reference=data["time_reference"],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureSummary:
    mean: float
    std: float
    min: float
    p05: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float
    max: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureSummary":
        return cls(
            mean=float(data["mean"]),
            std=float(data["std"]),
            min=float(data["min"]),
            p05=float(data["p05"]),
            p10=float(data["p10"]),
            p25=float(data["p25"]),
            p50=float(data["p50"]),
            p75=float(data["p75"]),
            p90=float(data["p90"]),
            p95=float(data["p95"]),
            p99=float(data["p99"]),
            max=float(data["max"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AudioAuditSummary:
    rows: int
    features: dict[str, FeatureSummary]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AudioAuditSummary":
        return cls(
            rows=int(data["rows"]),
            features={
                name: FeatureSummary.from_dict(values)
                for name, values in data["features"].items()
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AudioAuditBuildResult:
    source: AudioAuditSource
    parameters: AudioAuditParameters
    summary: AudioAuditSummary
    features: dict[str, np.ndarray]
    sample_rate: int | float
    rows: int
