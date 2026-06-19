from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sqlalchemy import inspect, select

from assemblybot.db.loaders.pipeline_run import (
    DuplicatePipelineRunError,
    build_pipeline_run_records,
    load_pipeline_run_records,
)
from assemblybot.db.loaders.session import load_session_record
from assemblybot.db.schema.pipeline_run import PipelineRunRecord
from assemblybot.db.session import (
    create_all_tables,
    create_session_factory,
    create_sqlite_engine,
    session_scope,
)
from assemblybot.helper.document import load_document


CANONICAL_JSON_PATH = Path(
    "data/interim/"
    "1ere-seance--questions-au-gouvernement--simplification-de-la-vie-economique-cmp--renforcer-la-s-14-avril-2026_03_alignment.json"
)


class PipelineRunDatabaseTest(unittest.TestCase):
    def test_build_pipeline_run_records_from_canonical_json(self) -> None:
        document = load_document(CANONICAL_JSON_PATH)

        records = build_pipeline_run_records(document, session_id=7)
        by_stage = {record.stage: record for record in records}

        self.assertEqual(set(by_stage), {"vad", "transcription", "diarization"})
        self.assertTrue(all(record.schema_ver == "0.1.0" for record in records))
        self.assertTrue(all(record.session_id == 7 for record in records))

        self.assertEqual(by_stage["vad"].engine_name, "silero-vad")
        self.assertEqual(by_stage["vad"].model, "silero-vad")
        self.assertIsNone(by_stage["vad"].device)
        self.assertEqual(by_stage["vad"].config_json["threshold"], 0.5)

        self.assertEqual(by_stage["transcription"].engine_name, "faster-whisper")
        self.assertEqual(by_stage["transcription"].model, "large-v3")
        self.assertEqual(by_stage["transcription"].device, "cuda")
        self.assertEqual(by_stage["transcription"].config_json["compute_type"], "float16")
        self.assertEqual(
            by_stage["transcription"].config_json["options"]["beam_size"],
            5,
        )

        self.assertEqual(by_stage["diarization"].engine_name, "pyannote")
        self.assertEqual(
            by_stage["diarization"].model,
            "pyannote/speaker-diarization-3.1",
        )
        self.assertEqual(by_stage["diarization"].device, "cuda")
        self.assertEqual(
            by_stage["diarization"].config_json["options"]["embedding_model"],
            "pyannote/embedding",
        )

    def test_load_pipeline_run_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            audio_path = tmpdir_path / "audio.mp3"
            audio_path.write_bytes(b"known test audio bytes")

            engine = create_sqlite_engine(tmpdir_path / "assemblybot.sqlite")
            create_all_tables(engine)

            inspector = inspect(engine)
            self.assertIn("session", inspector.get_table_names())
            self.assertIn("pipeline_run", inspector.get_table_names())

            session_factory = create_session_factory(engine)
            with session_scope(session_factory) as db_session:
                session_record = load_session_record(
                    db_session,
                    CANONICAL_JSON_PATH,
                    audio_path,
                )
                records = load_pipeline_run_records(
                    db_session,
                    CANONICAL_JSON_PATH,
                    session_record.id,
                )

                self.assertEqual(len(records), 3)
                self.assertTrue(all(record.id is not None for record in records))
                self.assertTrue(
                    all(record.session_id == session_record.id for record in records)
                )

            with session_scope(session_factory) as db_session:
                rows = db_session.scalars(
                    select(PipelineRunRecord).order_by(PipelineRunRecord.stage)
                ).all()

                self.assertEqual(
                    [row.stage for row in rows],
                    ["diarization", "transcription", "vad"],
                )

    def test_missing_parent_session_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = create_sqlite_engine(Path(tmpdir) / "assemblybot.sqlite")
            create_all_tables(engine)
            session_factory = create_session_factory(engine)

            with session_scope(session_factory) as db_session:
                with self.assertRaises(ValueError):
                    load_pipeline_run_records(db_session, CANONICAL_JSON_PATH, 999)

    def test_duplicate_pipeline_run_raises(self) -> None:
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
                    CANONICAL_JSON_PATH,
                    audio_path,
                )
                load_pipeline_run_records(
                    db_session,
                    CANONICAL_JSON_PATH,
                    session_record.id,
                )

            with session_scope(session_factory) as db_session:
                with self.assertRaises(DuplicatePipelineRunError):
                    load_pipeline_run_records(db_session, CANONICAL_JSON_PATH, 1)


if __name__ == "__main__":
    unittest.main()
