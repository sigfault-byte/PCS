from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from sqlalchemy import select

from assemblybot.db.loaders.session import load_session_record
from assemblybot.db.loaders.speakers import (
    CanonicalTurnAlignmentError,
    DuplicateSpeakerClusterError,
    PersonExternalIdConflictError,
    SpeakerMajorityAmbiguityError,
    build_person_records,
    build_speaker_cluster_records,
    compute_speaker_cluster_majorities,
    ensure_turn_analysis_alignment,
    get_or_create_person_records,
    load_speaker_cluster_records,
    load_turn_document,
)
from assemblybot.db.schema.person import PersonRecord
from assemblybot.db.schema.speaker import SpeakerClusterRecord
from assemblybot.db.session import (
    create_all_tables,
    create_session_factory,
    create_sqlite_engine,
    session_scope,
)
from assemblybot.models.time import TimeRange
from assemblybot.models.turn_document import (
    PersonIdentity,
    SpeakerIdentityEvidence,
    Turn,
    TurnAnalysis,
    TurnDocument,
)


ALIGNMENT_JSON_PATH = Path(
    "data/interim/"
    "1ere-seance--questions-au-gouvernement--simplification-de-la-vie-economique-cmp--renforcer-la-s-14-avril-2026_03_alignment.json"
)
PER_EXTRACTION_JSON_PATH = Path(
    "data/interim/"
    "1ere-seance--questions-au-gouvernement--simplification-de-la-vie-economique-cmp--renforcer-la-s-14-avril-2026_02_per_extraction.json"
)


def make_turn_document() -> TurnDocument:
    alice = PersonIdentity(id="person:a", name="Alice Dupont", role="Député", kind="deputy")
    raw = PersonIdentity(id=None, name="elisabeth lucot", role=None, kind="raw_per")
    turns = [
        Turn(1, TimeRange.from_seconds(0, 3), "hello", "SPEAKER_01", 0.5, [], []),
        Turn(2, TimeRange.from_seconds(3, 5), "again", "SPEAKER_01", 1.0, [], []),
        Turn(3, TimeRange.from_seconds(5, 7), "raw", "SPEAKER_02", 0.75, [], []),
    ]
    analyses = [
        TurnAnalysis(1, current_speaker=alice),
        TurnAnalysis(2, current_speaker=alice),
        TurnAnalysis(3, current_speaker=raw),
    ]
    return TurnDocument(turns=turns, turns_analysis=analyses)


def evidence(
    person: PersonIdentity,
    target_turn: Turn,
    *,
    source: str = "chair_next_speaker_call",
    eligible: bool = True,
) -> SpeakerIdentityEvidence:
    return SpeakerIdentityEvidence(
        source=source,
        eligible_for_cluster_majority=eligible,
        person=person,
        source_turn_id=target_turn.turn_id,
        target_turn_id=target_turn.turn_id,
        source_speaker_id=target_turn.speaker_id,
        target_speaker_id=target_turn.speaker_id,
        speaker_raw=person.name,
        speaker_normalized=person.name.lower(),
        match_score=100.0,
        is_known_person=person.id is not None,
    )


class SpeakerDatabaseTest(unittest.TestCase):
    def test_build_person_records_from_current_speakers(self) -> None:
        document = make_turn_document()
        evidence_only = PersonIdentity(
            id="person:e",
            name="Evidence Person",
            role="Député",
            kind="deputy",
        )
        mentioned_only = PersonIdentity(
            id=None,
            name="mentioned person",
            role=None,
            kind="raw_per",
        )
        document.turns_analysis[0].speaker_identity_evidence = [
            evidence(evidence_only, document.turns[0])
        ]
        document.turns_analysis[0].mentioned_persons = [mentioned_only]

        records = build_person_records(document)
        by_key = {(record.normalized_name, record.kind): record for record in records}

        alice = by_key[("alice dupont", "deputy")]
        raw = by_key[("elisabeth lucot", "raw_per")]
        evidence_person = by_key[("evidence person", "deputy")]
        mentioned_person = by_key[("mentioned person", "raw_per")]

        self.assertEqual(len(records), 4)
        self.assertEqual(alice.name, "Alice Dupont")
        self.assertEqual(alice.external_id, "person:a")
        self.assertIsNone(alice.party)
        self.assertEqual(alice.role, "Député")
        self.assertIsNone(alice.canonical_voice_centroid)
        self.assertIsNone(raw.external_id)
        self.assertEqual(evidence_person.external_id, "person:e")
        self.assertIsNone(mentioned_person.external_id)

    def test_person_external_id_conflict_raises(self) -> None:
        document = make_turn_document()
        document.turns_analysis[1].current_speaker = PersonIdentity(
            id="person:b",
            name="Alice Dupont",
            role="Député",
            kind="deputy",
        )

        with self.assertRaises(PersonExternalIdConflictError):
            build_person_records(document)

    def test_existing_person_external_id_conflict_raises(self) -> None:
        document = make_turn_document()

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = create_sqlite_engine(Path(tmpdir) / "assemblybot.sqlite")
            create_all_tables(engine)
            session_factory = create_session_factory(engine)

            with session_scope(session_factory) as db_session:
                db_session.add(
                    PersonRecord(
                        name="Alice Dupont",
                        normalized_name="alice dupont",
                        kind="deputy",
                        external_id="person:other",
                        party=None,
                        role="Député",
                        canonical_voice_centroid=None,
                    )
                )

            with session_scope(session_factory) as db_session:
                with self.assertRaises(PersonExternalIdConflictError):
                    get_or_create_person_records(db_session, document)

    def test_turn_alignment_mismatch_raises(self) -> None:
        document = make_turn_document()
        document.turns_analysis[1].turn_id = 99

        with self.assertRaises(CanonicalTurnAlignmentError):
            ensure_turn_analysis_alignment(document)

    def test_tied_majority_evidence_raises(self) -> None:
        document = make_turn_document()
        bob = PersonIdentity(
            id="person:b",
            name="Bob Martin",
            role="Député",
            kind="deputy",
        )
        document.turns[0].audio_time = TimeRange.from_seconds(0, 2)
        document.turns[1].audio_time = TimeRange.from_seconds(2, 4)
        document.turns_analysis[0].speaker_identity_evidence = [
            evidence(document.turns_analysis[0].current_speaker, document.turns[0])
        ]
        document.turns_analysis[1].speaker_identity_evidence = [
            evidence(bob, document.turns[1])
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = create_sqlite_engine(Path(tmpdir) / "assemblybot.sqlite")
            create_all_tables(engine)
            session_factory = create_session_factory(engine)

            with session_scope(session_factory) as db_session:
                people_by_key = get_or_create_person_records(db_session, document)
                with self.assertRaises(SpeakerMajorityAmbiguityError):
                    compute_speaker_cluster_majorities(document, people_by_key)

    def test_build_speaker_cluster_records(self) -> None:
        document = make_turn_document()
        alice = document.turns_analysis[0].current_speaker
        assert alice is not None
        document.turns_analysis[0].speaker_identity_evidence = [
            evidence(alice, document.turns[0])
        ]
        majorities = {
            "SPEAKER_01": type(
                "Majority",
                (),
                {
                    "person_id": 11,
                    "evidence_purity": 1.0,
                    "absolute_purity": 3.0 / 5.0,
                },
            )()
        }
        records = build_speaker_cluster_records(
            ["SPEAKER_01", "SPEAKER_02", "SPEAKER_03"],
            document,
            session_id=7,
            majorities_by_label=majorities,
        )
        by_label = {record.label: record for record in records}

        self.assertEqual(by_label["SPEAKER_01"].total_detected_speech, 5.0)
        self.assertEqual(by_label["SPEAKER_01"].majority_person_id, 11)
        self.assertEqual(by_label["SPEAKER_01"].evidence_purity, 1.0)
        self.assertEqual(by_label["SPEAKER_01"].absolute_purity, 3.0 / 5.0)
        self.assertEqual(by_label["SPEAKER_03"].total_detected_speech, 0.0)
        self.assertIsNone(by_label["SPEAKER_03"].majority_person_id)
        self.assertIsNone(by_label["SPEAKER_03"].evidence_purity)
        self.assertIsNone(by_label["SPEAKER_03"].absolute_purity)

    def test_duration_weighted_majority_metrics(self) -> None:
        document = make_turn_document()
        alice = document.turns_analysis[0].current_speaker
        bob = PersonIdentity(id="person:b", name="Bob Martin", role="Député", kind="deputy")
        assert alice is not None
        document.turns = [
            Turn(1, TimeRange.from_seconds(0, 10), "a", "SPEAKER_01", 1.0, [], []),
            Turn(2, TimeRange.from_seconds(10, 15), "b", "SPEAKER_01", 1.0, [], []),
            Turn(3, TimeRange.from_seconds(15, 40), "none", "SPEAKER_01", 1.0, [], []),
        ]
        document.turns_analysis = [
            TurnAnalysis(
                1,
                speaker_identity_evidence=[evidence(alice, document.turns[0])],
            ),
            TurnAnalysis(
                2,
                speaker_identity_evidence=[evidence(bob, document.turns[1])],
            ),
            TurnAnalysis(3),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = create_sqlite_engine(Path(tmpdir) / "assemblybot.sqlite")
            create_all_tables(engine)
            session_factory = create_session_factory(engine)

            with session_scope(session_factory) as db_session:
                people_by_key = get_or_create_person_records(db_session, document)
                majority = compute_speaker_cluster_majorities(
                    document,
                    people_by_key,
                )["SPEAKER_01"]

                self.assertAlmostEqual(majority.evidence_purity or 0.0, 10 / 15)
                self.assertAlmostEqual(majority.absolute_purity or 0.0, 10 / 40)

    def test_load_speaker_clusters_from_current_data(self) -> None:
        turn_document = load_turn_document(PER_EXTRACTION_JSON_PATH)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            audio_path = tmpdir_path / "audio.mp3"
            audio_path.write_bytes(b"known test audio bytes")

            engine = create_sqlite_engine(tmpdir_path / "assemblybot.sqlite")
            create_all_tables(engine)
            session_factory = create_session_factory(engine)

            with session_scope(session_factory) as db_session:
                session_record = load_session_record(
                    db_session,
                    ALIGNMENT_JSON_PATH,
                    audio_path,
                )
                records = load_speaker_cluster_records(
                    db_session,
                    ALIGNMENT_JSON_PATH,
                    PER_EXTRACTION_JSON_PATH,
                    session_record.id,
                )

                self.assertEqual(len(records), 64)

            with session_scope(session_factory) as db_session:
                speaker_40 = db_session.scalar(
                    select(SpeakerClusterRecord).where(
                        SpeakerClusterRecord.label == "SPEAKER_40"
                    )
                )
                speaker_36 = db_session.scalar(
                    select(SpeakerClusterRecord).where(
                        SpeakerClusterRecord.label == "SPEAKER_36"
                    )
                )
                unresolved = db_session.scalar(
                    select(SpeakerClusterRecord).where(
                        SpeakerClusterRecord.label == "SPEAKER_00"
                    )
                )
                chair = db_session.scalar(
                    select(PersonRecord).where(
                        PersonRecord.normalized_name == "yael braun-pivet",
                        PersonRecord.kind == "assembly_chair",
                    )
                )
                raw_per = db_session.scalar(
                    select(PersonRecord).where(
                        PersonRecord.normalized_name == "elisabeth lucot",
                        PersonRecord.kind == "raw_per",
                    )
                )

                self.assertIsNotNone(speaker_40)
                self.assertIsNotNone(speaker_36)
                self.assertIsNotNone(unresolved)
                self.assertIsNotNone(chair)
                self.assertIsNotNone(raw_per)
                assert speaker_40 is not None
                assert speaker_36 is not None
                assert unresolved is not None
                assert chair is not None
                assert raw_per is not None

                self.assertAlmostEqual(
                    speaker_40.total_detected_speech,
                    1046.6043750000035,
                )
                self.assertAlmostEqual(
                    speaker_36.total_detected_speech,
                    1055.5143750000152,
                )
                self.assertEqual(speaker_40.majority_person_id, chair.id)
                self.assertAlmostEqual(speaker_40.evidence_purity, 1.0)
                self.assertAlmostEqual(
                    speaker_40.absolute_purity,
                    0.9649634801115751,
                )
                self.assertEqual(speaker_36.majority_person_id, chair.id)
                self.assertAlmostEqual(speaker_36.evidence_purity, 1.0)
                self.assertAlmostEqual(
                    speaker_36.absolute_purity,
                    0.45546691393947863,
                )
                self.assertIsNone(unresolved.majority_person_id)
                self.assertIsNone(raw_per.external_id)
                self.assertIsNone(raw_per.canonical_voice_centroid)

                stats = {
                    turn.speaker_id: sum(
                        item.audio_time.duration_seconds
                        for item in turn_document.turns
                        if item.speaker_id == turn.speaker_id
                    )
                    for turn in turn_document.turns
                }
                self.assertAlmostEqual(stats["SPEAKER_40"], speaker_40.total_detected_speech)

    def test_missing_session_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = create_sqlite_engine(Path(tmpdir) / "assemblybot.sqlite")
            create_all_tables(engine)
            session_factory = create_session_factory(engine)

            with session_scope(session_factory) as db_session:
                with self.assertRaises(ValueError):
                    load_speaker_cluster_records(
                        db_session,
                        ALIGNMENT_JSON_PATH,
                        PER_EXTRACTION_JSON_PATH,
                        session_id=999,
                    )

    def test_duplicate_speaker_clusters_raise(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            audio_path = tmpdir_path / "audio.mp3"
            audio_path.write_bytes(b"known test audio bytes")

            engine = create_sqlite_engine(tmpdir_path / "assemblybot.sqlite")
            create_all_tables(engine)
            session_factory = create_session_factory(engine)

            with session_scope(session_factory) as db_session:
                session_record = load_session_record(
                    db_session,
                    ALIGNMENT_JSON_PATH,
                    audio_path,
                )
                load_speaker_cluster_records(
                    db_session,
                    ALIGNMENT_JSON_PATH,
                    PER_EXTRACTION_JSON_PATH,
                    session_record.id,
                )

            with session_scope(session_factory) as db_session:
                with self.assertRaises(DuplicateSpeakerClusterError):
                    load_speaker_cluster_records(
                        db_session,
                        ALIGNMENT_JSON_PATH,
                        PER_EXTRACTION_JSON_PATH,
                        session_id=1,
                    )


if __name__ == "__main__":
    unittest.main()
