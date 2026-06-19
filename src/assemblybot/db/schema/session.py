from __future__ import annotations

from datetime import date

from sqlalchemy import Date, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from assemblybot.db.base import Base


class SessionRecord(Base):
    """One source audio/video session imported into the shared SQLite DB."""

    __tablename__ = "session"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    slug: Mapped[str | None] = mapped_column(String, unique=True, nullable=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    source_url: Mapped[str | None] = mapped_column(String, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    vad_duration: Mapped[float | None] = mapped_column(Float, nullable=True)
    audio_file_hash: Mapped[str] = mapped_column(String(64), nullable=False)
