from dataclasses import dataclass, field

from .time import TimeRange


@dataclass
class TranscriptEngine:
    name: str = "faster-whisper"
    model: str | None = None
    device: str | None = None
    compute_type: str | None = None


@dataclass
class TranscriptRawSegment:
    segment_id: str
    time: TimeRange
    text: str


@dataclass
class TranscriptSection:
    engine: TranscriptEngine = field(default_factory=TranscriptEngine)
    language_detected: str | None = None
    language_probability: float | None = None
    segments_count: int = 0
    raw_segments: list[TranscriptRawSegment] = field(default_factory=list)
