from __future__ import annotations

from typing import Any

from sqlalchemy import BigInteger, Float, ForeignKey, Integer, JSON
from sqlalchemy.orm import Mapped, mapped_column

from assemblybot.db.base import Base


class DiarizationSegmentRecord(Base):
    """One raw diarization segment emitted by the diarization pipeline run."""

    __tablename__ = "diarization_segment"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    pipeline_run_id: Mapped[int] = mapped_column(
        ForeignKey("pipeline_run.id"),
        nullable=False,
    )
    speaker_cluster_id: Mapped[int] = mapped_column(
        ForeignKey("speaker_cluster.id"),
        nullable=False,
    )
    start_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    end_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    flags: Mapped[int] = mapped_column(BigInteger, nullable=False)
    overlap_speaker_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
