from __future__ import annotations

from typing import Any

from sqlalchemy import ForeignKey, Integer, JSON, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from assemblybot.db.base import Base


class PipelineRunRecord(Base):
    """Runtime metadata for one executed pipeline stage."""

    __tablename__ = "pipeline_run"
    __table_args__ = (
        UniqueConstraint("session_id", "stage", name="uq_pipeline_run_session_stage"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    schema_ver: Mapped[str] = mapped_column(String, nullable=False)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("session.id"),
        nullable=False,
    )
    stage: Mapped[str] = mapped_column(String, nullable=False)
    engine_name: Mapped[str] = mapped_column(String, nullable=False)
    model: Mapped[str | None] = mapped_column(String, nullable=True)
    device: Mapped[str | None] = mapped_column(String, nullable=True)
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
