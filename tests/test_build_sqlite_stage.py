from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from assemblybot.stages.build_sqlite import (
    ExistingDatabaseError,
    build_sqlite_database,
    parse_args,
)
from tests.test_db_chunks import write_embedding_artifacts
from tests.test_db_segments import ALIGNMENT_JSON_PATH
from tests.test_db_turns import PER_EXTRACTION_JSON_PATH


def write_full_fixture_embedding_artifacts(tmpdir_path: Path) -> tuple[Path, Path, Path]:
    turn_ids = np.arange(1, 214, dtype=np.int32)
    turn_embeddings = np.column_stack(
        [
            np.ones(len(turn_ids), dtype=np.float32),
            np.zeros(len(turn_ids), dtype=np.float32),
        ]
    )
    chunk_turn_ids = np.asarray([1, 1, 2], dtype=np.int32)
    chunk_embeddings = np.asarray(
        [
            [0.6, 0.8],
            [0.8, 0.6],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    return write_embedding_artifacts(
        tmpdir_path,
        turn_ids=turn_ids,
        turn_embeddings=turn_embeddings,
        chunk_turn_ids=chunk_turn_ids,
        chunk_indices=np.asarray([0, 1, 0], dtype=np.int16),
        start_sentence_indices=np.asarray([0, 1, 0], dtype=np.int16),
        end_sentence_indices=np.asarray([1, 2, 1], dtype=np.int16),
        chunk_embeddings=chunk_embeddings,
    )


class BuildSqliteStageTest(unittest.TestCase):
    def test_parse_args(self) -> None:
        args = parse_args(
            [
                "--alignment-json",
                "alignment.json",
                "--per-json",
                "per.json",
                "--audio-path",
                "audio.mp3",
                "--turn-embeddings-npz",
                "turns.npz",
                "--semantic-chunks-npz",
                "chunks.npz",
                "--embedding-metadata-json",
                "metadata.json",
                "--output-db",
                "assemblybot.sqlite",
                "--replace",
            ]
        )

        self.assertEqual(args.alignment_json, "alignment.json")
        self.assertTrue(args.replace)

    def test_build_sqlite_database(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            audio_path = tmpdir_path / "audio.mp3"
            audio_path.write_bytes(b"known test audio bytes")
            output_db_path = tmpdir_path / "assemblybot.sqlite"
            artifacts = write_full_fixture_embedding_artifacts(tmpdir_path)

            result = build_sqlite_database(
                output_db_path=output_db_path,
                alignment_json_path=ALIGNMENT_JSON_PATH,
                per_json_path=PER_EXTRACTION_JSON_PATH,
                audio_path=audio_path,
                turn_embeddings_npz_path=artifacts[0],
                semantic_chunks_npz_path=artifacts[1],
                embedding_metadata_json_path=artifacts[2],
            )

            self.assertEqual(result.db_path, output_db_path.resolve())
            self.assertEqual(result.session_id, 1)
            self.assertEqual(result.counts["session"], 1)
            self.assertEqual(result.counts["pipeline_run"], 4)
            self.assertEqual(result.counts["turn"], 213)
            self.assertEqual(result.counts["turn_embedding"], 213)
            self.assertEqual(result.counts["semantic_chunk"], 3)
            self.assertEqual(result.counts["embedding"], 216)

    def test_existing_db_requires_replace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            output_db_path = tmpdir_path / "assemblybot.sqlite"
            output_db_path.write_bytes(b"existing")
            audio_path = tmpdir_path / "audio.mp3"
            audio_path.write_bytes(b"known test audio bytes")
            artifacts = write_full_fixture_embedding_artifacts(tmpdir_path)

            with self.assertRaises(ExistingDatabaseError):
                build_sqlite_database(
                    output_db_path=output_db_path,
                    alignment_json_path=ALIGNMENT_JSON_PATH,
                    per_json_path=PER_EXTRACTION_JSON_PATH,
                    audio_path=audio_path,
                    turn_embeddings_npz_path=artifacts[0],
                    semantic_chunks_npz_path=artifacts[1],
                    embedding_metadata_json_path=artifacts[2],
                )

    def test_replace_rebuilds_existing_db(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            output_db_path = tmpdir_path / "assemblybot.sqlite"
            output_db_path.write_bytes(b"existing")
            audio_path = tmpdir_path / "audio.mp3"
            audio_path.write_bytes(b"known test audio bytes")
            artifacts = write_full_fixture_embedding_artifacts(tmpdir_path)

            result = build_sqlite_database(
                output_db_path=output_db_path,
                alignment_json_path=ALIGNMENT_JSON_PATH,
                per_json_path=PER_EXTRACTION_JSON_PATH,
                audio_path=audio_path,
                turn_embeddings_npz_path=artifacts[0],
                semantic_chunks_npz_path=artifacts[1],
                embedding_metadata_json_path=artifacts[2],
                replace=True,
            )

            self.assertEqual(result.counts["turn_embedding"], 213)

    def test_missing_artifact_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            audio_path = tmpdir_path / "audio.mp3"
            audio_path.write_bytes(b"known test audio bytes")
            artifacts = write_full_fixture_embedding_artifacts(tmpdir_path)

            with self.assertRaises(FileNotFoundError):
                build_sqlite_database(
                    output_db_path=tmpdir_path / "assemblybot.sqlite",
                    alignment_json_path=ALIGNMENT_JSON_PATH,
                    per_json_path=PER_EXTRACTION_JSON_PATH,
                    audio_path=audio_path,
                    turn_embeddings_npz_path=tmpdir_path / "missing.npz",
                    semantic_chunks_npz_path=artifacts[1],
                    embedding_metadata_json_path=artifacts[2],
                )


if __name__ == "__main__":
    unittest.main()
