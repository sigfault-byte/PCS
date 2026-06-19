from __future__ import annotations

from sqlalchemy import Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from assemblybot.db.base import Base


class SpeakerClusterRecord(Base):
    """One Pyannote speaker cluster label within a session."""

    __tablename__ = "speaker_cluster"
    __table_args__ = (
        UniqueConstraint("session_id", "label", name="uq_speaker_cluster_session_label"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("session.id"),
        nullable=False,
    )
    label: Mapped[str] = mapped_column(String, nullable=False)
    total_detected_speech: Mapped[float] = mapped_column(Float, nullable=False)
    majority_person_id: Mapped[int | None] = mapped_column(
        ForeignKey("person.id"),
        nullable=True,
    )
    evidence_purity: Mapped[float | None] = mapped_column(Float, nullable=True)
    absolute_purity: Mapped[float | None] = mapped_column(Float, nullable=True)
