from dataclasses import dataclass, field

from .time import TimeRange


@dataclass
class TranscriptEngine:
    name: str = "faster-whisper"
    model: str | None = None
    device: str | None = None
    compute_type: str | None = None


# @dataclass
# class TranscriptRawSegment:
#     segment_id: str
#     time: TimeRange
#     text: str


# @dataclass
# class TranscriptSection:
#     engine: TranscriptEngine = field(default_factory=TranscriptEngine)
#     language_detected: str | None = None
#     language_probability: float | None = None
#     segments_count: int = 0
#     raw_segments: list[TranscriptRawSegment] = field(default_factory=list)


@dataclass
class TranscriptRawToken:
    token_id: int
    start_seconds: float
    end_seconds: float
    raw_token: str


@dataclass
class TranscriptRawSegment:
    segment_id: str
    start_token_id: int
    end_token_id: int
    time: TimeRange
    text: str


@dataclass
class TranscriptSection:
    engine: TranscriptEngine = field(default_factory=TranscriptEngine)
    language_detected: str | None = None
    language_probability: float | None = None
    tokens_count: int = 0
    segments_count: int = 0
    raw_tokens: list[TranscriptRawToken] = field(default_factory=list)
    raw_segments: list[TranscriptRawSegment] = field(default_factory=list)
