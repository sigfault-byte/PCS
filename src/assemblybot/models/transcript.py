from dataclasses import dataclass, field
from typing import Any

from .time import TimeRange


@dataclass
class TranscriptEngine:
    name: str = "faster-whisper"
    model: str | None = None
    device: str | None = None
    compute_type: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptEngine":
        return cls(
            name=data.get("name", "faster-whisper"),
            model=data.get("model"),
            device=data.get("device"),
            compute_type=data.get("compute_type"),
        )


@dataclass
class TranscriptRawToken:
    token_id: int
    start_seconds: float
    end_seconds: float
    raw_token: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptRawToken":
        return cls(
            token_id=data["token_id"],
            start_seconds=data["start_seconds"],
            end_seconds=data["end_seconds"],
            raw_token=data["raw_token"],
        )


@dataclass
class TranscriptRawSegment:
    segment_id: str
    start_token_id: int | None
    end_token_id: int | None
    time: TimeRange
    raw_text: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptRawSegment":
        return cls(
            segment_id=data["segment_id"],
            start_token_id=data.get("start_token_id"),
            end_token_id=data.get("end_token_id"),
            time=TimeRange.from_dict(data["time"]),
            raw_text=data.get("raw_text", ""),
        )


@dataclass
class TranscriptSection:
    engine: TranscriptEngine = field(default_factory=TranscriptEngine)
    language_detected: str | None = None
    language_probability: float | None = None
    raw_tokens: list[TranscriptRawToken] = field(default_factory=list)
    raw_segments: list[TranscriptRawSegment] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptSection":
        return cls(
            engine=TranscriptEngine.from_dict(data.get("engine", {})),
            language_detected=data.get("language_detected"),
            language_probability=data.get("language_probability"),
            raw_tokens=[
                TranscriptRawToken.from_dict(item)
                for item in data.get("raw_tokens", [])
            ],
            raw_segments=[
                TranscriptRawSegment.from_dict(item)
                for item in data.get("raw_segments", [])
            ],
        )
