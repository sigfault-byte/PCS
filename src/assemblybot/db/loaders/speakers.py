from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from assemblybot.db.schema.person import PersonRecord
from assemblybot.db.schema.session import SessionRecord
from assemblybot.db.schema.speaker import SpeakerClusterRecord
from assemblybot.helper.document import load_document
from assemblybot.models.turn_document import (
    PersonIdentity,
    SpeakerIdentityEvidence,
    TurnDocument,
)
from assemblybot.stages.per_identity import normalize_name


class DuplicateSpeakerClusterError(ValueError):
    """Raised when speaker clusters already exist for a session."""


class SpeakerMajorityAmbiguityError(ValueError):
    """Raised when a speaker cluster has no duration-weighted majority person."""


class CanonicalTurnAlignmentError(ValueError):
    """Raised when turns and turns_analysis violate index alignment."""


class PersonExternalIdConflictError(ValueError):
    """Raised when one normalized identity has conflicting external ids."""


@dataclass(frozen=True)
class SpeakerClusterStats:
    total_detected_speech: float


@dataclass(frozen=True)
class SpeakerClusterMajority:
    person_id: int | None
    evidence_purity: float | None
    absolute_purity: float | None


def load_turn_document(json_path: str | Path) -> TurnDocument:
    with Path(json_path).open("r", encoding="utf-8") as file:
        return TurnDocument.from_dict(json.load(file))


def person_identity_key(identity: PersonIdentity) -> tuple[str, str]:
    return normalize_name(identity.name), identity.kind


def ensure_turn_analysis_alignment(document: TurnDocument) -> None:
    """
    Enforce canonical turn alignment.

    Invariant:
    - turns[i] and turns_analysis[i] describe the same logical turn.
    - turn_id values are expected to match.
    - array ordering is part of the canonical format.
    """
    if len(document.turns) != len(document.turns_analysis):
        raise CanonicalTurnAlignmentError(
            "PER extraction JSON has different turns and turns_analysis lengths: "
            f"{len(document.turns)} != {len(document.turns_analysis)}"
        )

    for index, (turn, analysis) in enumerate(
        zip(document.turns, document.turns_analysis, strict=True)
    ):
        if turn.turn_id != analysis.turn_id:
            raise CanonicalTurnAlignmentError(
                "PER extraction JSON violates turn alignment at index "
                f"{index}: turns.turn_id={turn.turn_id}, "
                f"turns_analysis.turn_id={analysis.turn_id}"
            )


def get_diarization_speaker_labels(alignment_json_path: str | Path) -> list[str]:
    document = load_document(Path(alignment_json_path))
    labels = sorted({segment.speaker_id for segment in document.diarization.raw_segments})
    speakers_count = document.diarization.speakers_count

    if speakers_count is not None and len(labels) != speakers_count:
        raise ValueError(
            "Diarization speaker label count does not match speakers_count: "
            f"{len(labels)} != {speakers_count}"
        )

    return labels


def compute_speaker_cluster_stats(
    document: TurnDocument,
) -> dict[str, SpeakerClusterStats]:
    durations_by_speaker: dict[str, float] = defaultdict(float)

    for turn in document.turns:
        if turn.speaker_id is None:
            continue

        durations_by_speaker[turn.speaker_id] += turn.audio_time.duration_seconds

    stats: dict[str, SpeakerClusterStats] = {}
    for speaker_id, total_duration in durations_by_speaker.items():
        stats[speaker_id] = SpeakerClusterStats(
            total_detected_speech=total_duration,
        )

    return stats


def iter_person_identities(document: TurnDocument):
    for analysis in document.turns_analysis:
        if analysis.current_speaker is not None:
            yield analysis.current_speaker

        for evidence in analysis.speaker_identity_evidence:
            yield evidence.person

        yield from analysis.mentioned_persons


def build_person_records(document: TurnDocument) -> list[PersonRecord]:
    records_by_key: dict[tuple[str, str], PersonRecord] = {}

    for identity in iter_person_identities(document):
        normalized_name, kind = person_identity_key(identity)
        existing = records_by_key.get((normalized_name, kind))
        if existing is not None:
            if (
                existing.external_id is not None
                and identity.id is not None
                and existing.external_id != identity.id
            ):
                raise PersonExternalIdConflictError(
                    "Conflicting external_id values for person "
                    f"{normalized_name!r}/{kind!r}: "
                    f"{existing.external_id!r} != {identity.id!r}"
                )
            if existing.external_id is None and identity.id is not None:
                existing.external_id = identity.id
            continue

        records_by_key[(normalized_name, kind)] = PersonRecord(
            name=identity.name,
            normalized_name=normalized_name,
            kind=kind,
            external_id=identity.id,
            party=None,
            role=identity.role,
            canonical_voice_centroid=None,
        )

    return list(records_by_key.values())


def get_or_create_person_records(
    db_session: Session,
    document: TurnDocument,
) -> dict[tuple[str, str], PersonRecord]:
    built_records = build_person_records(document)
    records_by_key: dict[tuple[str, str], PersonRecord] = {}

    for record in built_records:
        key = (record.normalized_name, record.kind)
        existing = db_session.scalar(
            select(PersonRecord).where(
                PersonRecord.normalized_name == record.normalized_name,
                PersonRecord.kind == record.kind,
            )
        )

        if existing is not None:
            if (
                existing.external_id is not None
                and record.external_id is not None
                and existing.external_id != record.external_id
            ):
                raise PersonExternalIdConflictError(
                    "Conflicting external_id values for person "
                    f"{record.normalized_name!r}/{record.kind!r}: "
                    f"{existing.external_id!r} != {record.external_id!r}"
                )
            if existing.external_id is None and record.external_id is not None:
                existing.external_id = record.external_id
            records_by_key[key] = existing
            continue

        db_session.add(record)
        records_by_key[key] = record

    db_session.flush()
    return records_by_key


def compute_speaker_cluster_majorities(
    turn_document: TurnDocument,
    people_by_key: dict[tuple[str, str], PersonRecord],
) -> dict[str, SpeakerClusterMajority]:
    turns_by_id = {turn.turn_id: turn for turn in turn_document.turns}
    stats_by_speaker = compute_speaker_cluster_stats(turn_document)
    evidence_duration_by_speaker: dict[str, dict[tuple[str, str], float]] = defaultdict(
        lambda: defaultdict(float)
    )

    for analysis in turn_document.turns_analysis:
        for evidence in analysis.speaker_identity_evidence:
            if not evidence.eligible_for_cluster_majority:
                continue

            target_turn = turns_by_id.get(evidence.target_turn_id)
            if target_turn is None or target_turn.speaker_id is None:
                continue

            key = person_identity_key(evidence.person)
            evidence_duration_by_speaker[target_turn.speaker_id][
                key
            ] += target_turn.audio_time.duration_seconds

    majorities: dict[str, SpeakerClusterMajority] = {}
    for speaker_id, evidence_by_person in evidence_duration_by_speaker.items():
        total_evidence_duration = sum(evidence_by_person.values())
        if total_evidence_duration <= 0:
            continue

        max_duration = max(evidence_by_person.values())
        winning_keys = [
            key
            for key, duration in evidence_by_person.items()
            if duration == max_duration
        ]
        if len(winning_keys) > 1:
            raise SpeakerMajorityAmbiguityError(
                f"Speaker {speaker_id!r} has tied majority evidence: "
                f"{sorted(winning_keys)}"
            )

        winning_key = winning_keys[0]
        person = people_by_key[winning_key]
        if person.id is None:
            raise ValueError(f"Person for speaker {speaker_id!r} has no database id")

        total_cluster_duration = stats_by_speaker.get(
            speaker_id,
            SpeakerClusterStats(total_detected_speech=0.0),
        ).total_detected_speech
        majorities[speaker_id] = SpeakerClusterMajority(
            person_id=person.id,
            evidence_purity=max_duration / total_evidence_duration,
            absolute_purity=(
                max_duration / total_cluster_duration
                if total_cluster_duration > 0
                else None
            ),
        )

    return majorities


def build_speaker_cluster_records(
    labels: list[str],
    turn_document: TurnDocument,
    session_id: int,
    majorities_by_label: dict[str, SpeakerClusterMajority],
) -> list[SpeakerClusterRecord]:
    stats_by_speaker = compute_speaker_cluster_stats(turn_document)
    records: list[SpeakerClusterRecord] = []

    for label in labels:
        stats = stats_by_speaker.get(
            label,
            SpeakerClusterStats(total_detected_speech=0.0),
        )
        majority = majorities_by_label.get(
            label,
            SpeakerClusterMajority(
                person_id=None,
                evidence_purity=None,
                absolute_purity=None,
            ),
        )
        records.append(
            SpeakerClusterRecord(
                session_id=session_id,
                label=label,
                total_detected_speech=stats.total_detected_speech,
                majority_person_id=majority.person_id,
                evidence_purity=majority.evidence_purity,
                absolute_purity=majority.absolute_purity,
            )
        )

    return records


def load_speaker_cluster_records(
    db_session: Session,
    alignment_json_path: str | Path,
    per_extraction_json_path: str | Path,
    session_id: int,
) -> list[SpeakerClusterRecord]:
    parent = db_session.get(SessionRecord, session_id)
    if parent is None:
        raise ValueError(f"Session does not exist for session_id={session_id}")

    existing_cluster_id = db_session.scalar(
        select(SpeakerClusterRecord.id).where(
            SpeakerClusterRecord.session_id == session_id
        )
    )
    if existing_cluster_id is not None:
        raise DuplicateSpeakerClusterError(
            f"Speaker clusters already exist for session_id={session_id}"
        )

    labels = get_diarization_speaker_labels(alignment_json_path)
    turn_document = load_turn_document(per_extraction_json_path)
    ensure_turn_analysis_alignment(turn_document)

    people_by_key = get_or_create_person_records(db_session, turn_document)
    majorities_by_label = compute_speaker_cluster_majorities(
        turn_document,
        people_by_key,
    )
    records = build_speaker_cluster_records(
        labels,
        turn_document,
        session_id,
        majorities_by_label,
    )

    db_session.add_all(records)
    db_session.flush()
    return records
