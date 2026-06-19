from __future__ import annotations

from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from assemblybot.db.schema.diarization import DiarizationSegmentRecord
from assemblybot.db.schema.person import PersonRecord
from assemblybot.db.schema.session import SessionRecord
from assemblybot.db.schema.speaker import SpeakerClusterRecord
from assemblybot.db.schema.transcript import TranscriptSegmentRecord
from assemblybot.db.schema.turn import (
    TurnAnalysisRecord,
    TurnDiarizationSegmentRecord,
    TurnRecord,
    TurnTranscriptSegmentRecord,
)
from assemblybot.db.loaders.speakers import (
    ensure_turn_analysis_alignment,
    load_turn_document,
    person_identity_key,
)
from assemblybot.models.turn_document import (
    PersonIdentity,
    SpeakerIdentityEvidence,
    TurnDocument,
)


class DuplicateTurnLoadError(ValueError):
    """Raised when turn rows already exist for a session."""


class MissingTurnDependencyError(ValueError):
    """Raised when a turn loader dependency has not been loaded."""


LEGACY_PROPAGATED_SOURCES = {
    "inferred_from_next_speaker",
    "inferred_from_previous_speaker",
}
PROPAGATED_SOURCE = "propagated_from_speaker_cluster"


def build_speaker_cluster_lookup(
    db_session: Session,
    session_id: int,
) -> dict[str, int]:
    rows = db_session.execute(
        select(SpeakerClusterRecord.label, SpeakerClusterRecord.id).where(
            SpeakerClusterRecord.session_id == session_id
        )
    ).all()
    return {label: speaker_cluster_id for label, speaker_cluster_id in rows}


def build_person_lookup(db_session: Session) -> dict[tuple[str, str], int]:
    rows = db_session.execute(
        select(PersonRecord.normalized_name, PersonRecord.kind, PersonRecord.id)
    ).all()
    return {
        (normalized_name, kind): person_id
        for normalized_name, kind, person_id in rows
    }


def resolve_current_person_id(
    identity: PersonIdentity | None,
    people_by_key: dict[tuple[str, str], int],
) -> int | None:
    if identity is None:
        return None

    key = person_identity_key(identity)
    person_id = people_by_key.get(key)
    if person_id is None:
        raise MissingTurnDependencyError(
            f"Missing person row for current speaker {key!r}"
        )

    return person_id


def mentioned_persons_to_json(mentioned_persons: list[PersonIdentity]) -> list[dict[str, str | None]]:
    return [
        {
            "id": identity.id,
            "name": identity.name,
            "role": identity.role,
            "kind": identity.kind,
        }
        for identity in mentioned_persons
    ]


def speaker_identity_evidence_to_json(
    evidence: list[SpeakerIdentityEvidence],
) -> list[dict[str, object]]:
    return [
        {
            "source": item.source,
            "eligible_for_cluster_majority": item.eligible_for_cluster_majority,
            "person": {
                "id": item.person.id,
                "name": item.person.name,
                "role": item.person.role,
                "kind": item.person.kind,
            },
            "source_turn_id": item.source_turn_id,
            "target_turn_id": item.target_turn_id,
            "source_speaker_id": item.source_speaker_id,
            "target_speaker_id": item.target_speaker_id,
            "speaker_raw": item.speaker_raw,
            "speaker_normalized": item.speaker_normalized,
            "match_score": item.match_score,
            "is_known_person": item.is_known_person,
        }
        for item in evidence
    ]


def normalized_current_person_source(analysis) -> str | None:
    if (
        not analysis.speaker_identity_evidence
        and analysis.current_speaker_source in LEGACY_PROPAGATED_SOURCES
    ):
        return PROPAGATED_SOURCE

    return analysis.current_speaker_source


def build_turn_records(
    document: TurnDocument,
    session_id: int,
    speaker_clusters_by_label: dict[str, int],
) -> list[TurnRecord]:
    records: list[TurnRecord] = []

    for turn in document.turns:
        if turn.speaker_id is None:
            raise MissingTurnDependencyError(
                f"Turn {turn.turn_id} has no speaker_id"
            )

        speaker_cluster_id = speaker_clusters_by_label.get(turn.speaker_id)
        if speaker_cluster_id is None:
            raise MissingTurnDependencyError(
                f"Missing speaker_cluster for turn {turn.turn_id}: {turn.speaker_id!r}"
            )

        records.append(
            TurnRecord(
                id=turn.turn_id,
                session_id=session_id,
                speaker_cluster_id=speaker_cluster_id,
                speaker_confidence=turn.speaker_confidence,
                speaker_evidence_ratio=turn.speaker_evidence_ratio,
                text=turn.text,
                start_seconds=turn.audio_time.start_seconds,
                end_seconds=turn.audio_time.end_seconds,
                flags=int(turn.flags),
            )
        )

    return records


def build_turn_analysis_records(
    document: TurnDocument,
    people_by_key: dict[tuple[str, str], int],
) -> list[TurnAnalysisRecord]:
    records: list[TurnAnalysisRecord] = []

    for analysis in document.turns_analysis:
        records.append(
            TurnAnalysisRecord(
                id=analysis.turn_id,
                turn_id=analysis.turn_id,
                current_person_id=resolve_current_person_id(
                    analysis.current_speaker,
                    people_by_key,
                ),
                current_person_source=normalized_current_person_source(analysis),
                current_person_purity=analysis.current_speaker_purity,
                embedding=None,
                keywords_json=list(analysis.keywords),
                organizations_json=list(analysis.organizations),
                mentioned_persons_json=mentioned_persons_to_json(
                    analysis.mentioned_persons
                ),
                speaker_identity_evidence_json=speaker_identity_evidence_to_json(
                    analysis.speaker_identity_evidence
                ),
            )
        )

    return records


def validate_segment_references(
    db_session: Session,
    document: TurnDocument,
) -> None:
    transcript_ids = {
        item
        for turn in document.turns
        for item in turn.transcript_segment_ids
    }
    diarization_ids = {
        item
        for turn in document.turns
        for item in turn.diarization_segment_ids
    }

    existing_transcript_ids = set(
        db_session.scalars(
            select(TranscriptSegmentRecord.id).where(
                TranscriptSegmentRecord.id.in_(transcript_ids)
            )
        ).all()
    )
    missing_transcript_ids = transcript_ids - existing_transcript_ids
    if missing_transcript_ids:
        raise MissingTurnDependencyError(
            "Missing transcript_segment rows: "
            f"{sorted(missing_transcript_ids)[:10]}"
        )

    existing_diarization_ids = set(
        db_session.scalars(
            select(DiarizationSegmentRecord.id).where(
                DiarizationSegmentRecord.id.in_(diarization_ids)
            )
        ).all()
    )
    missing_diarization_ids = diarization_ids - existing_diarization_ids
    if missing_diarization_ids:
        raise MissingTurnDependencyError(
            "Missing diarization_segment rows: "
            f"{sorted(missing_diarization_ids)[:10]}"
        )


def build_turn_transcript_segment_records(
    document: TurnDocument,
) -> list[TurnTranscriptSegmentRecord]:
    return [
        TurnTranscriptSegmentRecord(
            turn_id=turn.turn_id,
            transcript_segment_id=transcript_segment_id,
        )
        for turn in document.turns
        for transcript_segment_id in turn.transcript_segment_ids
    ]


def build_turn_diarization_segment_records(
    document: TurnDocument,
) -> list[TurnDiarizationSegmentRecord]:
    return [
        TurnDiarizationSegmentRecord(
            turn_id=turn.turn_id,
            diarization_segment_id=diarization_segment_id,
        )
        for turn in document.turns
        for diarization_segment_id in turn.diarization_segment_ids
    ]


def load_turn_records(
    db_session: Session,
    per_extraction_json_path: str | Path,
    session_id: int,
) -> tuple[
    list[TurnRecord],
    list[TurnAnalysisRecord],
    list[TurnTranscriptSegmentRecord],
    list[TurnDiarizationSegmentRecord],
]:
    parent = db_session.get(SessionRecord, session_id)
    if parent is None:
        raise ValueError(f"Session does not exist for session_id={session_id}")

    existing_turn_id = db_session.scalar(
        select(TurnRecord.id).where(TurnRecord.session_id == session_id)
    )
    if existing_turn_id is not None:
        raise DuplicateTurnLoadError(f"Turns already exist for session_id={session_id}")

    document = load_turn_document(per_extraction_json_path)
    ensure_turn_analysis_alignment(document)

    speaker_clusters_by_label = build_speaker_cluster_lookup(db_session, session_id)
    people_by_key = build_person_lookup(db_session)
    validate_segment_references(db_session, document)

    turn_records = build_turn_records(
        document,
        session_id,
        speaker_clusters_by_label,
    )
    analysis_records = build_turn_analysis_records(document, people_by_key)
    transcript_link_records = build_turn_transcript_segment_records(document)
    diarization_link_records = build_turn_diarization_segment_records(document)

    db_session.add_all(turn_records)
    db_session.add_all(analysis_records)
    db_session.add_all(transcript_link_records)
    db_session.add_all(diarization_link_records)
    db_session.flush()

    return (
        turn_records,
        analysis_records,
        transcript_link_records,
        diarization_link_records,
    )
