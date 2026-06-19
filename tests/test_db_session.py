from __future__ import annotations

import hashlib
import tempfile
import unittest
from datetime import date
from pathlib import Path

from sqlalchemy import inspect

from assemblybot.db.loaders.session import (
    DuplicateSessionError,
    derive_session_title,
    load_session_record,
    parse_session_date,
)
from assemblybot.db.schema.session import SessionRecord
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
CANONICAL_SLUG = (
    "1ere-seance--questions-au-gouvernement--simplification-de-la-vie-economique-cmp--renforcer-la-s-14-avril-2026"
)


class SessionDatabaseTest(unittest.TestCase):
    def test_parse_session_metadata_from_slug(self) -> None:
        self.assertEqual(parse_session_date(CANONICAL_SLUG), date(2026, 4, 14))

        title = derive_session_title(CANONICAL_SLUG)

        self.assertTrue(title)
        self.assertNotIn("14 avril 2026", title.lower())
        self.assertIn("questions au gouvernement", title.lower())

    def test_malformed_session_date_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_session_date("session-without-date")

        with self.assertRaises(ValueError):
            derive_session_title("session-without-date")

    def test_load_session_record_from_canonical_json(self) -> None:
        document = load_document(CANONICAL_JSON_PATH)
        audio_bytes = b"known test audio bytes"
        expected_hash = hashlib.sha256(audio_bytes).hexdigest()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            audio_path = tmpdir_path / "audio.mp3"
            audio_path.write_bytes(audio_bytes)

            engine = create_sqlite_engine(tmpdir_path / "assemblybot.sqlite")
            create_all_tables(engine)

            inspector = inspect(engine)
            self.assertIn("session", inspector.get_table_names())

            session_factory = create_session_factory(engine)
            with session_scope(session_factory) as db_session:
                record = load_session_record(
                    db_session,
                    CANONICAL_JSON_PATH,
                    audio_path,
                )

                self.assertIsNotNone(record.id)
                self.assertEqual(record.slug, document.source.source_id)
                self.assertEqual(record.date, date(2026, 4, 14))
                self.assertEqual(
                    record.duration_seconds,
                    document.source.duration_seconds,
                )
                self.assertEqual(
                    record.vad_duration,
                    document.vad.speech_seconds_total,
                )
                self.assertEqual(record.audio_file_hash, expected_hash)
                self.assertIsNone(record.source_url)

            with session_scope(session_factory) as db_session:
                stored = db_session.get(SessionRecord, 1)

                self.assertIsNotNone(stored)
                assert stored is not None
                self.assertEqual(stored.slug, document.source.source_id)
                self.assertEqual(stored.audio_file_hash, expected_hash)

    def test_duplicate_session_slug_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            audio_path = tmpdir_path / "audio.mp3"
            audio_path.write_bytes(b"known test audio bytes")

            engine = create_sqlite_engine(tmpdir_path / "assemblybot.sqlite")
            create_all_tables(engine)
            session_factory = create_session_factory(engine)

            with session_scope(session_factory) as db_session:
                load_session_record(db_session, CANONICAL_JSON_PATH, audio_path)

            with session_scope(session_factory) as db_session:
                with self.assertRaises(DuplicateSessionError):
                    load_session_record(db_session, CANONICAL_JSON_PATH, audio_path)


if __name__ == "__main__":
    unittest.main()
