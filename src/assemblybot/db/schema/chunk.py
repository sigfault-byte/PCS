from __future__ import annotations

from sqlalchemy import (
    Boolean,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from assemblybot.db.base import Base


class EmbeddingRecord(Base):
    """One stored dense vector produced by an embedding pipeline run."""

    __tablename__ = "embedding"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("session.id"),
        nullable=False,
    )
    pipeline_run_id: Mapped[int] = mapped_column(
        ForeignKey("pipeline_run.id"),
        nullable=False,
    )
    model_name: Mapped[str] = mapped_column(String, nullable=False)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    dtype: Mapped[str] = mapped_column(String, nullable=False)
    normalized: Mapped[bool] = mapped_column(Boolean, nullable=False)


class SemanticChunkRecord(Base):
    """A deterministic semantic text chunk belonging to one turn."""

    __tablename__ = "semantic_chunk"
    __table_args__ = (
        UniqueConstraint("turn_id", "chunk_index", name="uq_semantic_chunk_turn_index"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pipeline_run_id: Mapped[int] = mapped_column(
        ForeignKey("pipeline_run.id"),
        nullable=False,
    )
    turn_id: Mapped[int] = mapped_column(ForeignKey("turn.id"), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_sentence_index: Mapped[int] = mapped_column(Integer, nullable=False)
    end_sentence_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_id: Mapped[int] = mapped_column(
        ForeignKey("embedding.id"),
        nullable=False,
    )


class TurnEmbeddingRecord(Base):
    """Link between a turn and its full-turn embedding vector."""

    __tablename__ = "turn_embedding"

    turn_id: Mapped[int] = mapped_column(ForeignKey("turn.id"), primary_key=True)
    embedding_id: Mapped[int] = mapped_column(
        ForeignKey("embedding.id"),
        primary_key=True,
    )
