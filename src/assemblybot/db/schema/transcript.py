from __future__ import annotations

from sqlalchemy import BigInteger, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from assemblybot.db.base import Base


class TranscriptSegmentRecord(Base):
    """One raw transcription segment emitted by the transcription pipeline run."""

    __tablename__ = "transcript_segment"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    pipeline_run_id: Mapped[int] = mapped_column(
        ForeignKey("pipeline_run.id"),
        nullable=False,
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    start_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    end_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    flags: Mapped[int] = mapped_column(BigInteger, nullable=False)
    avg_log_prob: Mapped[float | None] = mapped_column(Float, nullable=True)
    no_speech_prob: Mapped[float | None] = mapped_column(Float, nullable=True)
    compression_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
