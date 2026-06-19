from __future__ import annotations

import hashlib
import re
from datetime import date
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from assemblybot.db.schema.session import SessionRecord
from assemblybot.helper.document import load_document
from assemblybot.models.document import CanonicalDocument


FRENCH_MONTHS = {
    "janvier": 1,
    "fevrier": 2,
    "février": 2,
    "mars": 3,
    "avril": 4,
    "mai": 5,
    "juin": 6,
    "juillet": 7,
    "aout": 8,
    "août": 8,
    "septembre": 9,
    "octobre": 10,
    "novembre": 11,
    "decembre": 12,
    "décembre": 12,
}

DATE_SUFFIX_RE = re.compile(
    r"-(?P<day>\d{1,2})-(?P<month>[a-zéû]+)-(?P<year>\d{4})$",
    re.IGNORECASE,
)


class DuplicateSessionError(ValueError):
    """Raised when a session slug already exists in the target database."""


def parse_session_date(slug: str) -> date:
    """Parse the trailing French date from a session slug."""
    match = DATE_SUFFIX_RE.search(slug)
    if match is None:
        raise ValueError(f"Could not parse trailing French date from slug: {slug}")

    month_name = match.group("month").lower()
    month = FRENCH_MONTHS.get(month_name)
    if month is None:
        raise ValueError(f"Unknown French month in slug: {month_name}")

    return date(
        int(match.group("year")),
        month,
        int(match.group("day")),
    )


def derive_session_title(slug: str) -> str:
    """Build a readable title from a source slug, excluding its date suffix."""
    match = DATE_SUFFIX_RE.search(slug)
    if match is None:
        raise ValueError(f"Could not parse trailing French date from slug: {slug}")

    title_slug = slug[: match.start()].strip("-")
    title = title_slug.replace("--", " - ").replace("-", " ")
    title = " ".join(title.split())
    if not title:
        raise ValueError(f"Could not derive a title from slug: {slug}")

    return title[0].upper() + title[1:]


def hash_file_sha256(path: str | Path) -> str:
    """Return the SHA-256 hex digest for a local file."""
    path = Path(path)
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def build_session_record(
    document: CanonicalDocument,
    audio_path: str | Path,
) -> SessionRecord:
    """Build a session ORM row from the canonical document and source audio."""
    slug = document.source.source_id or None
    if slug is None:
        raise ValueError("Document source.source_id is required to load a session")

    return SessionRecord(
        slug=slug,
        title=derive_session_title(slug),
        date=parse_session_date(slug),
        source_url=None,
        duration_seconds=document.source.duration_seconds,
        vad_duration=document.vad.speech_seconds_total,
        audio_file_hash=hash_file_sha256(audio_path),
    )


def load_session_record(
    db_session: Session,
    json_path: str | Path,
    audio_path: str | Path,
) -> SessionRecord:
    """Load one canonical JSON document into the `session` table."""
    document = load_document(Path(json_path))
    record = build_session_record(document, audio_path)

    existing_id = db_session.scalar(
        select(SessionRecord.id).where(SessionRecord.slug == record.slug)
    )
    if existing_id is not None:
        raise DuplicateSessionError(
            f"Session already exists for slug {record.slug!r} (id={existing_id})"
        )

    db_session.add(record)
    db_session.flush()
    return record
