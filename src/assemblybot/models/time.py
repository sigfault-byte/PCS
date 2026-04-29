from dataclasses import dataclass
from datetime import datetime, timezone


def seconds_to_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.ss."""

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"


def now_utc_iso() -> str:
    """Current UTC time as an ISO-8601 string."""

    return datetime.now(timezone.utc).isoformat()


@dataclass
class TimeRange:
    """Shared interval representation used by every timed model."""

    start_seconds: float
    end_seconds: float
    duration_seconds: float
    start_ts: str
    end_ts: str

    @classmethod
    def from_seconds(cls, start_seconds: float, end_seconds: float) -> "TimeRange":
        return cls(
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            duration_seconds=max(0.0, end_seconds - start_seconds),
            start_ts=seconds_to_timestamp(start_seconds),
            end_ts=seconds_to_timestamp(end_seconds),
        )

    @classmethod
    def from_dict(cls, data: dict) -> "TimeRange":
        return cls.from_seconds(
            start_seconds=data["start_seconds"],
            end_seconds=data["end_seconds"],
        )

    def to_dict(self) -> dict[str, float | str]:
        return {
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "duration_seconds": self.duration_seconds,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
        }
