from __future__ import annotations

from typing import Any

from sqlalchemy import BigInteger, Float, ForeignKey, Integer, JSON, LargeBinary, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from assemblybot.db.base import Base


class TurnRecord(Base):
    """One logical merged turn in a session."""

    __tablename__ = "turn"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("session.id"),
        nullable=False,
    )
    speaker_cluster_id: Mapped[int] = mapped_column(
        ForeignKey("speaker_cluster.id"),
        nullable=False,
    )
    speaker_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    start_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    end_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    flags: Mapped[int] = mapped_column(BigInteger, nullable=False)


class TurnAnalysisRecord(Base):
    """Analysis and enrichment attached to one logical turn."""

    __tablename__ = "turn_analysis"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    turn_id: Mapped[int] = mapped_column(ForeignKey("turn.id"), nullable=False)
    current_person_id: Mapped[int | None] = mapped_column(
        ForeignKey("person.id"),
        nullable=True,
    )
    current_person_source: Mapped[str | None] = mapped_column(String, nullable=True)
    current_person_purity: Mapped[float | None] = mapped_column(Float, nullable=True)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    keywords_json: Mapped[list[Any]] = mapped_column(JSON, nullable=False)
    organizations_json: Mapped[list[Any]] = mapped_column(JSON, nullable=False)
    mentioned_persons_json: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON,
        nullable=False,
    )
    speaker_identity_evidence_json: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON,
        nullable=False,
    )


class TurnTranscriptSegmentRecord(Base):
    """Many-to-many link between turns and raw transcript segments."""

    __tablename__ = "turn_transcript_segment"

    turn_id: Mapped[int] = mapped_column(
        ForeignKey("turn.id"),
        primary_key=True,
    )
    transcript_segment_id: Mapped[int] = mapped_column(
        ForeignKey("transcript_segment.id"),
        primary_key=True,
    )


class TurnDiarizationSegmentRecord(Base):
    """Many-to-many link between turns and raw diarization segments."""

    __tablename__ = "turn_diarization_segment"

    turn_id: Mapped[int] = mapped_column(
        ForeignKey("turn.id"),
        primary_key=True,
    )
    diarization_segment_id: Mapped[int] = mapped_column(
        ForeignKey("diarization_segment.id"),
        primary_key=True,
    )
