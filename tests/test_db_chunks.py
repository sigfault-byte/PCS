from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sqlalchemy import inspect, select

from assemblybot.db.loaders.chunks import (
    DuplicateEmbeddingLoadError,
    InvalidEmbeddingArtifactError,
    MissingEmbeddingDependencyError,
    load_embedding_records,
)
from assemblybot.db.loaders.turns import load_turn_records
from assemblybot.db.schema.chunk import (
    EmbeddingRecord,
    SemanticChunkRecord,
    TurnEmbeddingRecord,
)
from assemblybot.db.schema.pipeline_run import PipelineRunRecord
from assemblybot.db.session import (
    create_all_tables,
    create_session_factory,
    create_sqlite_engine,
    session_scope,
)
from tests.test_db_turns import (
    PER_EXTRACTION_JSON_PATH,
    load_all_turn_dependencies,
)


def write_embedding_artifacts(
    tmpdir_path: Path,
    *,
    turn_ids: np.ndarray | None = None,
    turn_embeddings: np.ndarray | None = None,
    chunk_turn_ids: np.ndarray | None = None,
    chunk_indices: np.ndarray | None = None,
    start_sentence_indices: np.ndarray | None = None,
    end_sentence_indices: np.ndarray | None = None,
    chunk_embeddings: np.ndarray | None = None,
) -> tuple[Path, Path, Path]:
    turn_embeddings_path = tmpdir_path / "turn_embeddings.npz"
    semantic_chunks_path = tmpdir_path / "semantic_chunks.npz"
    metadata_path = tmpdir_path / "semantic_chunk_metadata.json"

    np.savez_compressed(
        turn_embeddings_path,
        turn_ids=(
            turn_ids
            if turn_ids is not None
            else np.asarray([1, 2], dtype=np.int32)
        ),
        embeddings=(
            turn_embeddings
            if turn_embeddings is not None
            else np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        ),
    )
    np.savez_compressed(
        semantic_chunks_path,
        turn_ids=(
            chunk_turn_ids
            if chunk_turn_ids is not None
            else np.asarray([1, 1], dtype=np.int32)
        ),
        chunk_indices=(
            chunk_indices
            if chunk_indices is not None
            else np.asarray([0, 1], dtype=np.int16)
        ),
        start_sentence_indices=(
            start_sentence_indices
            if start_sentence_indices is not None
            else np.asarray([0, 1], dtype=np.int16)
        ),
        end_sentence_indices=(
            end_sentence_indices
            if end_sentence_indices is not None
            else np.asarray([1, 2], dtype=np.int16)
        ),
        embeddings=(
            chunk_embeddings
            if chunk_embeddings is not None
            else np.asarray([[0.6, 0.8], [0.8, 0.6]], dtype=np.float32)
        ),
    )
    metadata_path.write_text(
        json.dumps(
            {
                "model_name": "h4c5/sts-camembert-base",
                "dtype": "float32",
                "normalize_embeddings": True,
                "chunking_method": "semantic_delta_v1",
                "delta_threshold": -0.1,
                "min_words": 8,
            }
        ),
        encoding="utf-8",
    )

    return turn_embeddings_path, semantic_chunks_path, metadata_path


def load_turn_database(tmpdir_path: Path):
    audio_path = tmpdir_path / "audio.mp3"
    audio_path.write_bytes(b"known test audio bytes")

    engine = create_sqlite_engine(tmpdir_path / "assemblybot.sqlite")
    create_all_tables(engine)
    session_factory = create_session_factory(engine)

    with session_scope(session_factory) as db_session:
        session_id = load_all_turn_dependencies(db_session, audio_path)
        load_turn_records(db_session, PER_EXTRACTION_JSON_PATH, session_id)

    return session_factory, session_id


class ChunkDatabaseTest(unittest.TestCase):
    def test_create_all_tables_includes_embedding_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = create_sqlite_engine(Path(tmpdir) / "assemblybot.sqlite")
            create_all_tables(engine)
            inspector = inspect(engine)

            self.assertIn("embedding", inspector.get_table_names())
            self.assertIn("semantic_chunk", inspector.get_table_names())
            self.assertIn("turn_embedding", inspector.get_table_names())

            embedding_columns = {
                column["name"] for column in inspector.get_columns("embedding")
            }
            chunk_columns = {
                column["name"] for column in inspector.get_columns("semantic_chunk")
            }
            self.assertIn("pipeline_run_id", embedding_columns)
            self.assertIn("pipeline_run_id", chunk_columns)

    def test_load_embedding_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            session_factory, session_id = load_turn_database(tmpdir_path)
            artifacts = write_embedding_artifacts(tmpdir_path)

            with session_scope(session_factory) as db_session:
                result = load_embedding_records(db_session, session_id, *artifacts)

                self.assertEqual(result.pipeline_run.stage, "embedding")
                self.assertEqual(len(result.embeddings), 4)
                self.assertEqual(len(result.turn_embeddings), 2)
                self.assertEqual(len(result.semantic_chunks), 2)

            with session_scope(session_factory) as db_session:
                pipeline_run = db_session.scalar(
                    select(PipelineRunRecord).where(
                        PipelineRunRecord.session_id == session_id,
                        PipelineRunRecord.stage == "embedding",
                    )
                )
                self.assertIsNotNone(pipeline_run)
                assert pipeline_run is not None

                embeddings = db_session.scalars(select(EmbeddingRecord)).all()
                turn_links = db_session.scalars(select(TurnEmbeddingRecord)).all()
                chunks = db_session.scalars(
                    select(SemanticChunkRecord).order_by(SemanticChunkRecord.chunk_index)
                ).all()

                self.assertEqual(len(embeddings), 4)
                self.assertTrue(
                    all(item.pipeline_run_id == pipeline_run.id for item in embeddings)
                )
                self.assertEqual(embeddings[0].model_name, "h4c5/sts-camembert-base")
                self.assertEqual(embeddings[0].dimension, 2)
                self.assertEqual(embeddings[0].dtype, "float32")
                self.assertTrue(embeddings[0].normalized)
                np.testing.assert_allclose(
                    np.frombuffer(embeddings[0].vector, dtype=np.float32),
                    [1.0, 0.0],
                )

                self.assertEqual(
                    sorted((link.turn_id, link.embedding_id) for link in turn_links),
                    [(1, embeddings[0].id), (2, embeddings[1].id)],
                )
                self.assertEqual(len(chunks), 2)
                self.assertEqual(chunks[0].pipeline_run_id, pipeline_run.id)
                self.assertEqual(chunks[0].turn_id, 1)
                self.assertEqual(chunks[0].chunk_index, 0)
                self.assertEqual(chunks[0].start_sentence_index, 0)
                self.assertEqual(chunks[0].end_sentence_index, 1)
                self.assertTrue(chunks[0].text.startswith("Bonjour à tous"))
                self.assertEqual(chunks[0].embedding_id, embeddings[2].id)

    def test_missing_session_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            engine = create_sqlite_engine(tmpdir_path / "assemblybot.sqlite")
            create_all_tables(engine)
            session_factory = create_session_factory(engine)
            artifacts = write_embedding_artifacts(tmpdir_path)

            with session_scope(session_factory) as db_session:
                with self.assertRaises(ValueError):
                    load_embedding_records(db_session, 999, *artifacts)

    def test_missing_turn_id_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            session_factory, session_id = load_turn_database(tmpdir_path)
            artifacts = write_embedding_artifacts(
                tmpdir_path,
                turn_ids=np.asarray([999], dtype=np.int32),
                turn_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
            )

            with session_scope(session_factory) as db_session:
                with self.assertRaises(MissingEmbeddingDependencyError):
                    load_embedding_records(db_session, session_id, *artifacts)

    def test_mismatched_array_lengths_raise(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            session_factory, session_id = load_turn_database(tmpdir_path)
            artifacts = write_embedding_artifacts(
                tmpdir_path,
                chunk_indices=np.asarray([0], dtype=np.int16),
            )

            with session_scope(session_factory) as db_session:
                with self.assertRaises(InvalidEmbeddingArtifactError):
                    load_embedding_records(db_session, session_id, *artifacts)

    def test_non_float32_embeddings_raise(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            session_factory, session_id = load_turn_database(tmpdir_path)
            artifacts = write_embedding_artifacts(
                tmpdir_path,
                turn_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float64),
                turn_ids=np.asarray([1], dtype=np.int32),
            )

            with session_scope(session_factory) as db_session:
                with self.assertRaises(InvalidEmbeddingArtifactError):
                    load_embedding_records(db_session, session_id, *artifacts)

    def test_duplicate_embedding_load_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            session_factory, session_id = load_turn_database(tmpdir_path)
            artifacts = write_embedding_artifacts(tmpdir_path)

            with session_scope(session_factory) as db_session:
                load_embedding_records(db_session, session_id, *artifacts)

            with session_scope(session_factory) as db_session:
                with self.assertRaises(DuplicateEmbeddingLoadError):
                    load_embedding_records(db_session, session_id, *artifacts)


if __name__ == "__main__":
    unittest.main()
