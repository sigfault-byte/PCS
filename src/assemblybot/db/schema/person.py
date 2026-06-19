from __future__ import annotations

from sqlalchemy import LargeBinary, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from assemblybot.db.base import Base


class PersonRecord(Base):
    """A real-world or best-known speaker identity."""

    __tablename__ = "person"
    __table_args__ = (
        UniqueConstraint("normalized_name", "kind", name="uq_person_name_kind"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    normalized_name: Mapped[str] = mapped_column(String, nullable=False)
    kind: Mapped[str] = mapped_column(String, nullable=False)
    external_id: Mapped[str | None] = mapped_column(String, nullable=True)
    party: Mapped[str | None] = mapped_column(String, nullable=True)
    role: Mapped[str | None] = mapped_column(String, nullable=True)
    canonical_voice_centroid: Mapped[bytes | None] = mapped_column(
        LargeBinary,
        nullable=True,
    )
