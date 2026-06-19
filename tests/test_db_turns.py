from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from sqlalchemy import func, select

from assemblybot.db.loaders.pipeline_run import load_pipeline_run_records
from assemblybot.db.loaders.segments import load_segment_records
from assemblybot.db.loaders.session import load_session_record
from assemblybot.db.loaders.speakers import (
    CanonicalTurnAlignmentError,
    load_speaker_cluster_records,
    load_turn_document,
)
from assemblybot.db.loaders.turns import (
    DuplicateTurnLoadError,
    MissingTurnDependencyError,
    build_turn_analysis_records,
    build_turn_diarization_segment_records,
    build_turn_records,
    build_turn_transcript_segment_records,
    load_turn_records,
    validate_segment_references,
)
from assemblybot.db.schema.diarization import DiarizationSegmentRecord
from assemblybot.db.schema.person import PersonRecord
from assemblybot.db.schema.speaker import SpeakerClusterRecord
from assemblybot.db.schema.transcript import TranscriptSegmentRecord
from assemblybot.db.schema.turn import (
    TurnAnalysisRecord,
    TurnDiarizationSegmentRecord,
    TurnRecord,
    TurnTranscriptSegmentRecord,
)
from assemblybot.db.session import (
    create_all_tables,
    create_session_factory,
    create_sqlite_engine,
    session_scope,
)
from assemblybot.models.turn_document import SpeakerIdentityEvidence
from assemblybot.stages.per_identity import normalize_name


ALIGNMENT_JSON_PATH = Path(
    "data/interim/"
    "1ere-seance--questions-au-gouvernement--simplification-de-la-vie-economique-cmp--renforcer-la-s-14-avril-2026_03_alignment.json"
)
PER_EXTRACTION_JSON_PATH = Path(
    "data/interim/"
    "1ere-seance--questions-au-gouvernement--simplification-de-la-vie-economique-cmp--renforcer-la-s-14-avril-2026_02_per_extraction.json"
)


def load_all_turn_dependencies(db_session, audio_path: Path) -> int:
    session_record = load_session_record(
        db_session,
        ALIGNMENT_JSON_PATH,
        audio_path,
    )
    load_pipeline_run_records(db_session, ALIGNMENT_JSON_PATH, session_record.id)
    load_speaker_cluster_records(
        db_session,
        ALIGNMENT_JSON_PATH,
        PER_EXTRACTION_JSON_PATH,
        session_record.id,
    )
    load_segment_records(db_session, ALIGNMENT_JSON_PATH, session_record.id)
    return session_record.id


class TurnDatabaseTest(unittest.TestCase):
    def test_build_turn_records(self) -> None:
        document = load_turn_document(PER_EXTRACTION_JSON_PATH)
        speaker_lookup = {
            speaker_id: index
            for index, speaker_id in enumerate(
                sorted({turn.speaker_id for turn in document.turns if turn.speaker_id}),
                start=1,
            )
        }

        records = build_turn_records(
            document,
            session_id=1,
            speaker_clusters_by_label=speaker_lookup,
        )
        first = records[0]

        self.assertEqual(len(records), 213)
        self.assertEqual(first.id, 1)
        self.assertEqual(first.session_id, 1)
        self.assertEqual(first.speaker_cluster_id, speaker_lookup["SPEAKER_40"])
        self.assertEqual(first.speaker_confidence, 1.0)
        self.assertTrue(first.text.startswith("Bonjour à tous"))
        self.assertEqual(first.start_seconds, 913.47471875)
        self.assertEqual(first.end_seconds, 933.6909687500001)
        self.assertEqual(first.flags, 6144)

    def test_build_turn_analysis_records(self) -> None:
        document = load_turn_document(PER_EXTRACTION_JSON_PATH)
        chair = document.turns_analysis[0].current_speaker
        assert chair is not None
        document.turns_analysis[0].speaker_identity_evidence = [
            SpeakerIdentityEvidence(
                source="hardcoded_assembly_chair",
                eligible_for_cluster_majority=True,
                person=chair,
                source_turn_id=1,
                target_turn_id=1,
                source_speaker_id="SPEAKER_40",
                target_speaker_id="SPEAKER_40",
                speaker_raw=chair.name,
                speaker_normalized="yael braun-pivet",
                match_score=100.0,
                is_known_person=True,
            )
        ]
        people_lookup = {
            (normalize_name(analysis.current_speaker.name), analysis.current_speaker.kind): index
            for index, analysis in enumerate(
                [
                    item
                    for item in document.turns_analysis
                    if item.current_speaker is not None
                ],
                start=1,
            )
        }

        records = build_turn_analysis_records(
            document,
            people_by_key=people_lookup,
        )
        first = records[0]
        second = records[1]

        self.assertEqual(len(records), 213)
        self.assertEqual(first.id, 1)
        self.assertEqual(first.turn_id, 1)
        self.assertEqual(
            first.current_person_id,
            people_lookup[("yael braun-pivet", "assembly_chair")],
        )
        self.assertEqual(first.current_person_source, "hardcoded_assembly_chair")
        self.assertEqual(first.current_person_purity, 1.0)
        self.assertIsNone(first.embedding)
        self.assertEqual(first.keywords_json, [])
        self.assertEqual(first.organizations_json, [])
        self.assertEqual(
            first.speaker_identity_evidence_json[0]["source"],
            "hardcoded_assembly_chair",
        )
        self.assertTrue(
            first.speaker_identity_evidence_json[0][
                "eligible_for_cluster_majority"
            ]
        )
        self.assertEqual(len(first.mentioned_persons_json), 5)
        self.assertEqual(first.mentioned_persons_json[0]["name"], "cecile collere")
        self.assertIsNone(second.current_person_id)

    def test_build_turn_segment_links(self) -> None:
        document = load_turn_document(PER_EXTRACTION_JSON_PATH)

        transcript_links = build_turn_transcript_segment_records(document)
        diarization_links = build_turn_diarization_segment_records(document)

        self.assertEqual(len(transcript_links), 3287)
        self.assertEqual(len(diarization_links), 1382)
        self.assertEqual(
            [
                link.transcript_segment_id
                for link in transcript_links
                if link.turn_id == 1
            ],
            [1, 2, 3],
        )
        self.assertEqual(
            [
                link.diarization_segment_id
                for link in diarization_links
                if link.turn_id == 1
            ],
            [1, 2, 3, 4, 5],
        )

    def test_load_turn_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            audio_path = tmpdir_path / "audio.mp3"
            audio_path.write_bytes(b"known test audio bytes")

            engine = create_sqlite_engine(tmpdir_path / "assemblybot.sqlite")
            create_all_tables(engine)
            session_factory = create_session_factory(engine)

            with session_scope(session_factory) as db_session:
                session_id = load_all_turn_dependencies(db_session, audio_path)
                turns, analyses, transcript_links, diarization_links = load_turn_records(
                    db_session,
                    PER_EXTRACTION_JSON_PATH,
                    session_id,
                )

                self.assertEqual(len(turns), 213)
                self.assertEqual(len(analyses), 213)
                self.assertEqual(len(transcript_links), 3287)
                self.assertEqual(len(diarization_links), 1382)

            with session_scope(session_factory) as db_session:
                self.assertEqual(
                    db_session.scalar(select(func.count(TurnRecord.id))),
                    213,
                )
                self.assertEqual(
                    db_session.scalar(select(func.count(TurnAnalysisRecord.id))),
                    213,
                )
                self.assertEqual(
                    db_session.scalar(select(func.count(TurnTranscriptSegmentRecord.turn_id))),
                    3287,
                )
                self.assertEqual(
                    db_session.scalar(select(func.count(TurnDiarizationSegmentRecord.turn_id))),
                    1382,
                )

                first_turn = db_session.get(TurnRecord, 1)
                first_analysis = db_session.get(TurnAnalysisRecord, 1)
                second_analysis = db_session.get(TurnAnalysisRecord, 2)
                propagated_analysis = db_session.get(TurnAnalysisRecord, 6)
                chair = db_session.scalar(
                    select(PersonRecord).where(
                        PersonRecord.normalized_name == "yael braun-pivet",
                        PersonRecord.kind == "assembly_chair",
                    )
                )

                self.assertIsNotNone(first_turn)
                self.assertIsNotNone(first_analysis)
                self.assertIsNotNone(second_analysis)
                self.assertIsNotNone(propagated_analysis)
                self.assertIsNotNone(chair)
                assert first_turn is not None
                assert first_analysis is not None
                assert second_analysis is not None
                assert propagated_analysis is not None
                assert chair is not None

                self.assertEqual(first_turn.flags, 6144)
                self.assertEqual(first_analysis.current_person_id, chair.id)
                self.assertIsNone(first_analysis.embedding)
                self.assertEqual(first_analysis.mentioned_persons_json[0]["kind"], "raw_per")
                self.assertIsNone(second_analysis.current_person_id)
                self.assertEqual(
                    propagated_analysis.current_person_source,
                    "chair_next_speaker_call",
                )
                self.assertEqual(
                    propagated_analysis.speaker_identity_evidence_json[0]["source"],
                    "chair_next_speaker_call",
                )

    def test_missing_session_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = create_sqlite_engine(Path(tmpdir) / "assemblybot.sqlite")
            create_all_tables(engine)
            session_factory = create_session_factory(engine)

            with session_scope(session_factory) as db_session:
                with self.assertRaises(ValueError):
                    load_turn_records(db_session, PER_EXTRACTION_JSON_PATH, 999)

    def test_missing_speaker_cluster_raises(self) -> None:
        document = load_turn_document(PER_EXTRACTION_JSON_PATH)

        with self.assertRaises(MissingTurnDependencyError):
            build_turn_records(document, session_id=1, speaker_clusters_by_label={})

    def test_missing_current_person_raises(self) -> None:
        document = load_turn_document(PER_EXTRACTION_JSON_PATH)

        with self.assertRaises(MissingTurnDependencyError):
            build_turn_analysis_records(document, people_by_key={})

    def test_missing_segment_reference_raises(self) -> None:
        document = load_turn_document(PER_EXTRACTION_JSON_PATH)

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = create_sqlite_engine(Path(tmpdir) / "assemblybot.sqlite")
            create_all_tables(engine)
            session_factory = create_session_factory(engine)

            with session_scope(session_factory) as db_session:
                with self.assertRaises(MissingTurnDependencyError):
                    validate_segment_references(db_session, document)

    def test_duplicate_turns_raise(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            audio_path = tmpdir_path / "audio.mp3"
            audio_path.write_bytes(b"known test audio bytes")

            engine = create_sqlite_engine(tmpdir_path / "assemblybot.sqlite")
            create_all_tables(engine)
            session_factory = create_session_factory(engine)

            with session_scope(session_factory) as db_session:
                session_id = load_all_turn_dependencies(db_session, audio_path)
                load_turn_records(db_session, PER_EXTRACTION_JSON_PATH, session_id)

            with session_scope(session_factory) as db_session:
                with self.assertRaises(DuplicateTurnLoadError):
                    load_turn_records(db_session, PER_EXTRACTION_JSON_PATH, 1)

    def test_turn_analysis_mismatch_raises(self) -> None:
        document = load_turn_document(PER_EXTRACTION_JSON_PATH)
        broken = copy.deepcopy(document)
        broken.turns_analysis[0].turn_id = 999

        from assemblybot.db.loaders.speakers import ensure_turn_analysis_alignment

        with self.assertRaises(CanonicalTurnAlignmentError):
            ensure_turn_analysis_alignment(broken)


if __name__ == "__main__":
    unittest.main()
