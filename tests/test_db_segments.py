from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sqlalchemy import func, select

from assemblybot.db.loaders.pipeline_run import load_pipeline_run_records
from assemblybot.db.loaders.segments import (
    DuplicateSegmentLoadError,
    MissingPipelineRunError,
    MissingSpeakerClusterError,
    build_diarization_segment_records,
    build_speaker_cluster_lookup,
    build_transcript_segment_records,
    load_segment_records,
)
from assemblybot.db.loaders.session import load_session_record
from assemblybot.db.loaders.speakers import load_speaker_cluster_records
from assemblybot.db.schema.diarization import DiarizationSegmentRecord
from assemblybot.db.schema.speaker import SpeakerClusterRecord
from assemblybot.db.schema.transcript import TranscriptSegmentRecord
from assemblybot.db.session import (
    create_all_tables,
    create_session_factory,
    create_sqlite_engine,
    session_scope,
)
from assemblybot.helper.document import load_document
from assemblybot.models.flags import SegmentFlag


ALIGNMENT_JSON_PATH = Path(
    "data/interim/"
    "1ere-seance--questions-au-gouvernement--simplification-de-la-vie-economique-cmp--renforcer-la-s-14-avril-2026_03_alignment.json"
)
PER_EXTRACTION_JSON_PATH = Path(
    "data/interim/"
    "1ere-seance--questions-au-gouvernement--simplification-de-la-vie-economique-cmp--renforcer-la-s-14-avril-2026_02_per_extraction.json"
)


def load_required_parent_rows(db_session, audio_path: Path) -> int:
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
    return session_record.id


class SegmentDatabaseTest(unittest.TestCase):
    def test_build_transcript_segment_records(self) -> None:
        document = load_document(ALIGNMENT_JSON_PATH)

        records = build_transcript_segment_records(document, pipeline_run_id=2)
        first = records[0]

        self.assertEqual(len(records), 3287)
        self.assertEqual(first.id, 1)
        self.assertEqual(first.pipeline_run_id, 2)
        self.assertEqual(first.text, "Bonjour à tous, la séance est ouverte.")
        self.assertEqual(first.start_seconds, 913.26)
        self.assertEqual(first.end_seconds, 915.5)
        self.assertEqual(first.flags, 0)
        self.assertAlmostEqual(first.avg_log_prob or 0.0, -0.10845170481638475)
        self.assertAlmostEqual(first.no_speech_prob or 0.0, 0.01445770263671875)
        self.assertAlmostEqual(first.compression_ratio or 0.0, 1.4705882352941178)

    def test_build_diarization_segment_records(self) -> None:
        document = load_document(ALIGNMENT_JSON_PATH)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            audio_path = tmpdir_path / "audio.mp3"
            audio_path.write_bytes(b"known test audio bytes")

            engine = create_sqlite_engine(tmpdir_path / "assemblybot.sqlite")
            create_all_tables(engine)
            session_factory = create_session_factory(engine)

            with session_scope(session_factory) as db_session:
                session_id = load_required_parent_rows(db_session, audio_path)
                lookup = build_speaker_cluster_lookup(db_session, session_id)
                records = build_diarization_segment_records(
                    document,
                    pipeline_run_id=3,
                    speaker_clusters_by_label=lookup,
                )

                first = records[0]
                segment_62 = records[61]

                self.assertEqual(len(records), 1378)
                self.assertEqual(first.id, 1)
                self.assertEqual(first.pipeline_run_id, 3)
                self.assertEqual(first.start_seconds, 913.47471875)
                self.assertEqual(first.end_seconds, 915.9722187500001)
                self.assertEqual(first.flags, 0)
                self.assertEqual(first.overlap_speaker_ids, [])
                self.assertEqual(first.speaker_cluster_id, lookup["SPEAKER_40"])
                self.assertEqual(segment_62.id, 62)
                self.assertEqual(segment_62.overlap_speaker_ids, ["SPEAKER_40"])
                self.assertEqual(
                    segment_62.flags,
                    int(
                        SegmentFlag.DIARIZATION_OVERLAP
                        | SegmentFlag.MULTI_SPEAKER_CANDIDATE
                    ),
                )

    def test_load_segment_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            audio_path = tmpdir_path / "audio.mp3"
            audio_path.write_bytes(b"known test audio bytes")

            engine = create_sqlite_engine(tmpdir_path / "assemblybot.sqlite")
            create_all_tables(engine)
            session_factory = create_session_factory(engine)

            with session_scope(session_factory) as db_session:
                session_id = load_required_parent_rows(db_session, audio_path)
                transcript_records, diarization_records = load_segment_records(
                    db_session,
                    ALIGNMENT_JSON_PATH,
                    session_id,
                )

                self.assertEqual(len(transcript_records), 3287)
                self.assertEqual(len(diarization_records), 1378)

            with session_scope(session_factory) as db_session:
                transcript_count = db_session.scalar(
                    select(func.count(TranscriptSegmentRecord.id))
                )
                diarization_count = db_session.scalar(
                    select(func.count(DiarizationSegmentRecord.id))
                )
                transcript_first = db_session.get(TranscriptSegmentRecord, 1)
                diarization_62 = db_session.get(DiarizationSegmentRecord, 62)

                self.assertEqual(transcript_count, 3287)
                self.assertEqual(diarization_count, 1378)
                self.assertIsNotNone(transcript_first)
                self.assertIsNotNone(diarization_62)
                assert transcript_first is not None
                assert diarization_62 is not None
                self.assertEqual(transcript_first.pipeline_run_id, 2)
                self.assertEqual(diarization_62.pipeline_run_id, 3)
                self.assertEqual(diarization_62.overlap_speaker_ids, ["SPEAKER_40"])
                self.assertGreaterEqual(
                    diarization_62.flags,
                    int(SegmentFlag.MULTI_SPEAKER_CANDIDATE),
                )

    def test_missing_session_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = create_sqlite_engine(Path(tmpdir) / "assemblybot.sqlite")
            create_all_tables(engine)
            session_factory = create_session_factory(engine)

            with session_scope(session_factory) as db_session:
                with self.assertRaises(ValueError):
                    load_segment_records(db_session, ALIGNMENT_JSON_PATH, session_id=999)

    def test_missing_pipeline_run_raises(self) -> None:
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
                with self.assertRaises(MissingPipelineRunError):
                    load_segment_records(db_session, ALIGNMENT_JSON_PATH, session_record.id)

    def test_missing_speaker_cluster_raises(self) -> None:
        document = load_document(ALIGNMENT_JSON_PATH)

        with self.assertRaises(MissingSpeakerClusterError):
            build_diarization_segment_records(
                document,
                pipeline_run_id=3,
                speaker_clusters_by_label={},
            )

    def test_duplicate_segments_raise(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            audio_path = tmpdir_path / "audio.mp3"
            audio_path.write_bytes(b"known test audio bytes")

            engine = create_sqlite_engine(tmpdir_path / "assemblybot.sqlite")
            create_all_tables(engine)
            session_factory = create_session_factory(engine)

            with session_scope(session_factory) as db_session:
                session_id = load_required_parent_rows(db_session, audio_path)
                load_segment_records(db_session, ALIGNMENT_JSON_PATH, session_id)

            with session_scope(session_factory) as db_session:
                with self.assertRaises(DuplicateSegmentLoadError):
                    load_segment_records(db_session, ALIGNMENT_JSON_PATH, session_id=1)


if __name__ == "__main__":
    unittest.main()
