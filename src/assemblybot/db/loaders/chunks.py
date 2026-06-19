from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from assemblybot.db.schema.chunk import (
    EmbeddingRecord,
    SemanticChunkRecord,
    TurnEmbeddingRecord,
)
from assemblybot.db.schema.pipeline_run import PipelineRunRecord
from assemblybot.db.schema.session import SessionRecord
from assemblybot.db.schema.turn import TurnRecord
from assemblybot.helper.semantic_chunking import split_sentences


EMBEDDING_STAGE = "embedding"
EMBEDDING_SCHEMA_VERSION = "semantic_chunk_artifact_v1"
EMBEDDING_ENGINE_NAME = "sentence-transformers"


class DuplicateEmbeddingLoadError(ValueError):
    """Raised when embedding artifacts were already loaded for a session."""


class MissingEmbeddingDependencyError(ValueError):
    """Raised when the embedding loader depends on missing DB rows."""


class InvalidEmbeddingArtifactError(ValueError):
    """Raised when NPZ or metadata artifacts do not match the expected schema."""


@dataclass(frozen=True)
class EmbeddingLoadResult:
    pipeline_run: PipelineRunRecord
    embeddings: list[EmbeddingRecord]
    turn_embeddings: list[TurnEmbeddingRecord]
    semantic_chunks: list[SemanticChunkRecord]


def load_metadata(metadata_path: str | Path) -> dict[str, Any]:
    path = Path(metadata_path)
    with path.open("r", encoding="utf-8") as input_file:
        metadata = json.load(input_file)

    if not isinstance(metadata, dict):
        raise InvalidEmbeddingArtifactError("Embedding metadata must be a JSON object")
    return metadata


def validate_embedding_matrix(name: str, embeddings: np.ndarray) -> np.ndarray:
    if embeddings.dtype != np.float32:
        raise InvalidEmbeddingArtifactError(
            f"{name} embeddings must have dtype float32, got {embeddings.dtype}"
        )
    if embeddings.ndim != 2:
        raise InvalidEmbeddingArtifactError(
            f"{name} embeddings must be a 2D array, got shape {embeddings.shape}"
        )
    return embeddings


def ensure_1d_length(name: str, array: np.ndarray, expected_length: int) -> None:
    if array.ndim != 1:
        raise InvalidEmbeddingArtifactError(
            f"{name} must be a 1D array, got shape {array.shape}"
        )
    if len(array) != expected_length:
        raise InvalidEmbeddingArtifactError(
            f"{name} length {len(array)} does not match expected {expected_length}"
        )


def load_turn_embedding_artifact(
    turn_embeddings_npz_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    artifact = np.load(Path(turn_embeddings_npz_path))
    turn_ids = artifact["turn_ids"]
    embeddings = validate_embedding_matrix("turn", artifact["embeddings"])
    ensure_1d_length("turn_ids", turn_ids, embeddings.shape[0])
    return turn_ids.astype(np.int64, copy=False), embeddings


def load_semantic_chunk_artifact(
    semantic_chunks_npz_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    artifact = np.load(Path(semantic_chunks_npz_path))
    turn_ids = artifact["turn_ids"]
    chunk_indices = artifact["chunk_indices"]
    start_sentence_indices = artifact["start_sentence_indices"]
    end_sentence_indices = artifact["end_sentence_indices"]
    embeddings = validate_embedding_matrix("semantic chunk", artifact["embeddings"])

    expected_length = embeddings.shape[0]
    ensure_1d_length("chunk turn_ids", turn_ids, expected_length)
    ensure_1d_length("chunk_indices", chunk_indices, expected_length)
    ensure_1d_length("start_sentence_indices", start_sentence_indices, expected_length)
    ensure_1d_length("end_sentence_indices", end_sentence_indices, expected_length)

    return (
        turn_ids.astype(np.int64, copy=False),
        chunk_indices.astype(np.int64, copy=False),
        start_sentence_indices.astype(np.int64, copy=False),
        end_sentence_indices.astype(np.int64, copy=False),
        embeddings,
    )


def build_turn_lookup(
    db_session: Session,
    session_id: int,
) -> dict[int, TurnRecord]:
    rows = db_session.scalars(
        select(TurnRecord).where(TurnRecord.session_id == session_id)
    ).all()
    return {row.id: row for row in rows}


def ensure_turn_ids_exist(
    artifact_name: str,
    turn_ids: np.ndarray,
    turns_by_id: dict[int, TurnRecord],
) -> None:
    missing = sorted({int(turn_id) for turn_id in turn_ids} - set(turns_by_id))
    if missing:
        raise MissingEmbeddingDependencyError(
            f"{artifact_name} references missing turn ids: {missing[:10]}"
        )


def ensure_no_embedding_duplicates(db_session: Session, session_id: int) -> None:
    existing_run_id = db_session.scalar(
        select(PipelineRunRecord.id).where(
            PipelineRunRecord.session_id == session_id,
            PipelineRunRecord.stage == EMBEDDING_STAGE,
        )
    )
    if existing_run_id is not None:
        raise DuplicateEmbeddingLoadError(
            f"Embedding pipeline_run already exists for session_id={session_id}"
        )

    existing_turn_embedding_id = db_session.scalar(
        select(TurnEmbeddingRecord.turn_id)
        .join(TurnRecord, TurnEmbeddingRecord.turn_id == TurnRecord.id)
        .where(TurnRecord.session_id == session_id)
    )
    if existing_turn_embedding_id is not None:
        raise DuplicateEmbeddingLoadError(
            f"Turn embeddings already exist for session_id={session_id}"
        )

    existing_chunk_id = db_session.scalar(
        select(SemanticChunkRecord.id)
        .join(TurnRecord, SemanticChunkRecord.turn_id == TurnRecord.id)
        .where(TurnRecord.session_id == session_id)
    )
    if existing_chunk_id is not None:
        raise DuplicateEmbeddingLoadError(
            f"Semantic chunks already exist for session_id={session_id}"
        )


def create_embedding_pipeline_run(
    metadata: dict[str, Any],
    session_id: int,
) -> PipelineRunRecord:
    model_name = metadata.get("model_name")
    if not model_name:
        raise InvalidEmbeddingArtifactError("Embedding metadata is missing model_name")

    return PipelineRunRecord(
        schema_ver=EMBEDDING_SCHEMA_VERSION,
        session_id=session_id,
        stage=EMBEDDING_STAGE,
        engine_name=EMBEDDING_ENGINE_NAME,
        model=str(model_name),
        device=None,
        config_json=dict(metadata),
    )


def vector_to_bytes(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def build_embedding_record(
    *,
    session_id: int,
    pipeline_run_id: int,
    model_name: str,
    vector: np.ndarray,
    normalized: bool,
) -> EmbeddingRecord:
    return EmbeddingRecord(
        session_id=session_id,
        pipeline_run_id=pipeline_run_id,
        model_name=model_name,
        dimension=int(vector.shape[0]),
        vector=vector_to_bytes(vector),
        dtype="float32",
        normalized=normalized,
    )


def reconstruct_chunk_text(
    turn: TurnRecord,
    start_sentence_index: int,
    end_sentence_index: int,
) -> str:
    sentences = split_sentences(turn.text)
    if (
        start_sentence_index < 0
        or end_sentence_index <= start_sentence_index
        or end_sentence_index > len(sentences)
    ):
        raise InvalidEmbeddingArtifactError(
            "Invalid semantic chunk sentence range "
            f"[{start_sentence_index}:{end_sentence_index}] for turn_id={turn.id}"
        )

    return " ".join(sentences[start_sentence_index:end_sentence_index])


def load_embedding_records(
    db_session: Session,
    session_id: int,
    turn_embeddings_npz_path: str | Path,
    semantic_chunks_npz_path: str | Path,
    metadata_path: str | Path,
) -> EmbeddingLoadResult:
    parent = db_session.get(SessionRecord, session_id)
    if parent is None:
        raise ValueError(f"Session does not exist for session_id={session_id}")

    turns_by_id = build_turn_lookup(db_session, session_id)
    if not turns_by_id:
        raise MissingEmbeddingDependencyError(
            f"No turns exist for session_id={session_id}"
        )

    ensure_no_embedding_duplicates(db_session, session_id)

    metadata = load_metadata(metadata_path)
    model_name = str(metadata.get("model_name") or "")
    normalized = bool(metadata.get("normalize_embeddings", False))

    turn_ids, turn_embeddings = load_turn_embedding_artifact(turn_embeddings_npz_path)
    (
        chunk_turn_ids,
        chunk_indices,
        start_sentence_indices,
        end_sentence_indices,
        chunk_embeddings,
    ) = load_semantic_chunk_artifact(semantic_chunks_npz_path)

    if turn_embeddings.shape[1] != chunk_embeddings.shape[1]:
        raise InvalidEmbeddingArtifactError(
            "Turn and semantic chunk embeddings must have the same dimension: "
            f"{turn_embeddings.shape[1]} != {chunk_embeddings.shape[1]}"
        )

    ensure_turn_ids_exist("turn_embeddings", turn_ids, turns_by_id)
    ensure_turn_ids_exist("semantic_chunks", chunk_turn_ids, turns_by_id)

    pipeline_run = create_embedding_pipeline_run(metadata, session_id)
    db_session.add(pipeline_run)
    db_session.flush()

    embedding_records: list[EmbeddingRecord] = []
    turn_embedding_records: list[TurnEmbeddingRecord] = []
    for turn_id, vector in zip(turn_ids, turn_embeddings):
        embedding = build_embedding_record(
            session_id=session_id,
            pipeline_run_id=pipeline_run.id,
            model_name=model_name,
            vector=vector,
            normalized=normalized,
        )
        embedding_records.append(embedding)
        db_session.add(embedding)
        db_session.flush()
        turn_embedding_records.append(
            TurnEmbeddingRecord(
                turn_id=int(turn_id),
                embedding_id=embedding.id,
            )
        )

    semantic_chunk_records: list[SemanticChunkRecord] = []
    for turn_id, chunk_index, start_index, end_index, vector in zip(
        chunk_turn_ids,
        chunk_indices,
        start_sentence_indices,
        end_sentence_indices,
        chunk_embeddings,
    ):
        embedding = build_embedding_record(
            session_id=session_id,
            pipeline_run_id=pipeline_run.id,
            model_name=model_name,
            vector=vector,
            normalized=normalized,
        )
        embedding_records.append(embedding)
        db_session.add(embedding)
        db_session.flush()

        turn = turns_by_id[int(turn_id)]
        semantic_chunk_records.append(
            SemanticChunkRecord(
                pipeline_run_id=pipeline_run.id,
                turn_id=int(turn_id),
                chunk_index=int(chunk_index),
                start_sentence_index=int(start_index),
                end_sentence_index=int(end_index),
                text=reconstruct_chunk_text(
                    turn,
                    int(start_index),
                    int(end_index),
                ),
                embedding_id=embedding.id,
            )
        )

    db_session.add_all(turn_embedding_records)
    db_session.add_all(semantic_chunk_records)
    db_session.flush()

    return EmbeddingLoadResult(
        pipeline_run=pipeline_run,
        embeddings=embedding_records,
        turn_embeddings=turn_embedding_records,
        semantic_chunks=semantic_chunk_records,
    )
