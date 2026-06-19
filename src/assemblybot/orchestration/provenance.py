from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


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


@dataclass(frozen=True)
class DetectedDate:
    value: str | None
    matched_text: str | None


@dataclass(frozen=True)
class Provenance:
    original_filename: str
    file_stem: str
    detected_date: str | None
    detected_title_session_slug: str
    file_size_bytes: int
    sha256: str
    ingestion_timestamp: str

    @property
    def sha8(self) -> str:
        return self.sha256[:8]


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_date_from_stem(stem: str) -> DetectedDate:
    iso_match = re.search(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)", stem)
    if iso_match:
        return DetectedDate(
            value=f"{iso_match.group(1)}-{iso_match.group(2)}-{iso_match.group(3)}",
            matched_text=iso_match.group(0),
        )

    compact_match = re.search(r"(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)", stem)
    if compact_match:
        return DetectedDate(
            value=(
                f"{compact_match.group(1)}-"
                f"{compact_match.group(2)}-"
                f"{compact_match.group(3)}"
            ),
            matched_text=compact_match.group(0),
        )

    month_names = "|".join(sorted(FRENCH_MONTHS, key=len, reverse=True))
    french_match = re.search(
        rf"(?<!\d)(\d{{1,2}})[-_ ]+({month_names})[-_ ]+(\d{{4}})(?!\d)",
        stem,
        flags=re.IGNORECASE,
    )
    if french_match:
        day = int(french_match.group(1))
        month = FRENCH_MONTHS[french_match.group(2).lower()]
        year = int(french_match.group(3))
        return DetectedDate(
            value=f"{year:04d}-{month:02d}-{day:02d}",
            matched_text=french_match.group(0),
        )

    return DetectedDate(value=None, matched_text=None)


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_value.lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug or "session"


def slug_from_stem(stem: str, matched_date_text: str | None) -> str:
    title = stem
    if matched_date_text:
        title = title.replace(matched_date_text, " ")
    return slugify(title)


def build_provenance(path: Path, *, ingestion_timestamp: str | None = None) -> Provenance:
    detected_date = detect_date_from_stem(path.stem)
    return Provenance(
        original_filename=path.name,
        file_stem=path.stem,
        detected_date=detected_date.value,
        detected_title_session_slug=slug_from_stem(path.stem, detected_date.matched_text),
        file_size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
        ingestion_timestamp=ingestion_timestamp or now_utc_iso(),
    )
