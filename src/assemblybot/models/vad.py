from dataclasses import asdict, dataclass, field
from typing import Any

from .ids import require_positive_int_id
from .time import TimeRange


@dataclass
class VadEngine:
    """Runtime settings used by the voice activity detector."""

    name: str = "silero-vad"
    model: str | None = None
    threshold: float | None = None
    min_speech_duration_ms: int | None = None
    min_silence_duration_ms: int | None = None
    speech_pad_ms: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VadEngine":
        return cls(
            name=data.get("name", "silero-vad"),
            model=data.get("model"),
            threshold=data.get("threshold"),
            min_speech_duration_ms=data.get("min_speech_duration_ms"),
            min_silence_duration_ms=data.get("min_silence_duration_ms"),
            speech_pad_ms=data.get("speech_pad_ms"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VadSegment:
    """One detected speech interval."""

    segment_id: int
    time: TimeRange
    confidence: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VadSegment":
        return cls(
            segment_id=require_positive_int_id(data["segment_id"], "segment_id"),
            time=TimeRange.from_dict(data["time"]),
            confidence=data.get("confidence"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "time": self.time.to_dict(),
            "confidence": self.confidence,
        }


@dataclass
class VadSection:
    """Canonical VAD output stored at document.vad."""

    engine: VadEngine = field(default_factory=VadEngine)
    segments: list[VadSegment] = field(default_factory=list)
    speech_seconds_total: float = 0.0
    non_speech_seconds_total: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VadSection":
        return cls(
            engine=VadEngine.from_dict(data.get("engine", {})),
            segments=[
                VadSegment.from_dict(segment) for segment in data.get("segments", [])
            ],
            speech_seconds_total=data.get("speech_seconds_total", 0.0),
            non_speech_seconds_total=data.get("non_speech_seconds_total", 0.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine.to_dict(),
            "segments": [segment.to_dict() for segment in self.segments],
            "speech_seconds_total": self.speech_seconds_total,
            "non_speech_seconds_total": self.non_speech_seconds_total,
        }
