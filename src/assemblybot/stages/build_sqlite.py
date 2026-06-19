from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import func, select

from assemblybot.db.loaders.chunks import load_embedding_records
from assemblybot.db.loaders.pipeline_run import load_pipeline_run_records
from assemblybot.db.loaders.segments import load_segment_records
from assemblybot.db.loaders.session import load_session_record
from assemblybot.db.loaders.speakers import load_speaker_cluster_records
from assemblybot.db.loaders.turns import load_turn_records
from assemblybot.db.schema.chunk import (
    EmbeddingRecord,
    SemanticChunkRecord,
    TurnEmbeddingRecord,
)
from assemblybot.db.schema.diarization import DiarizationSegmentRecord
from assemblybot.db.schema.person import PersonRecord
from assemblybot.db.schema.pipeline_run import PipelineRunRecord
from assemblybot.db.schema.session import SessionRecord
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


@dataclass(frozen=True)
class BuildSqliteResult:
    db_path: Path
    session_id: int
    counts: dict[str, int]


class ExistingDatabaseError(ValueError):
    """Raised when the output database exists and replacement is not enabled."""


def require_existing_file(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def prepare_output_db_path(output_db_path: str | Path, *, replace: bool) -> Path:
    resolved = Path(output_db_path).expanduser().resolve()
    if resolved.exists():
        if not replace:
            raise ExistingDatabaseError(
                f"Output DB already exists: {resolved}. Pass --replace to rebuild it."
            )
        resolved.unlink()

    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def collect_counts(session) -> dict[str, int]:
    return {
        "session": session.scalar(select(func.count(SessionRecord.id))) or 0,
        "pipeline_run": session.scalar(select(func.count(PipelineRunRecord.id))) or 0,
        "person": session.scalar(select(func.count(PersonRecord.id))) or 0,
        "speaker_cluster": session.scalar(select(func.count(SpeakerClusterRecord.id))) or 0,
        "transcript_segment": session.scalar(select(func.count(TranscriptSegmentRecord.id))) or 0,
        "diarization_segment": session.scalar(select(func.count(DiarizationSegmentRecord.id))) or 0,
        "turn": session.scalar(select(func.count(TurnRecord.id))) or 0,
        "turn_analysis": session.scalar(select(func.count(TurnAnalysisRecord.id))) or 0,
        "turn_transcript_segment": (
            session.scalar(select(func.count(TurnTranscriptSegmentRecord.turn_id))) or 0
        ),
        "turn_diarization_segment": (
            session.scalar(select(func.count(TurnDiarizationSegmentRecord.turn_id))) or 0
        ),
        "embedding": session.scalar(select(func.count(EmbeddingRecord.id))) or 0,
        "turn_embedding": (
            session.scalar(select(func.count(TurnEmbeddingRecord.turn_id))) or 0
        ),
        "semantic_chunk": session.scalar(select(func.count(SemanticChunkRecord.id))) or 0,
    }


def build_sqlite_database(
    *,
    output_db_path: str | Path,
    alignment_json_path: str | Path,
    per_json_path: str | Path,
    audio_path: str | Path,
    turn_embeddings_npz_path: str | Path,
    semantic_chunks_npz_path: str | Path,
    embedding_metadata_json_path: str | Path,
    replace: bool = False,
) -> BuildSqliteResult:
    alignment_json_path = require_existing_file(alignment_json_path, "Alignment JSON")
    per_json_path = require_existing_file(per_json_path, "PER JSON")
    audio_path = require_existing_file(audio_path, "Audio file")
    turn_embeddings_npz_path = require_existing_file(
        turn_embeddings_npz_path,
        "Turn embeddings NPZ",
    )
    semantic_chunks_npz_path = require_existing_file(
        semantic_chunks_npz_path,
        "Semantic chunks NPZ",
    )
    embedding_metadata_json_path = require_existing_file(
        embedding_metadata_json_path,
        "Embedding metadata JSON",
    )
    output_db_path = prepare_output_db_path(output_db_path, replace=replace)

    engine = create_sqlite_engine(output_db_path)
    create_all_tables(engine)
    session_factory = create_session_factory(engine)

    with session_scope(session_factory) as db_session:
        session_record = load_session_record(
            db_session,
            alignment_json_path,
            audio_path,
        )
        load_pipeline_run_records(db_session, alignment_json_path, session_record.id)
        load_speaker_cluster_records(
            db_session,
            alignment_json_path,
            per_json_path,
            session_record.id,
        )
        load_segment_records(db_session, alignment_json_path, session_record.id)
        load_turn_records(db_session, per_json_path, session_record.id)
        load_embedding_records(
            db_session,
            session_record.id,
            turn_embeddings_npz_path,
            semantic_chunks_npz_path,
            embedding_metadata_json_path,
        )
        counts = collect_counts(db_session)

    return BuildSqliteResult(
        db_path=output_db_path,
        session_id=session_record.id,
        counts=counts,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fresh AssemblyBot SQLite database from pipeline artifacts."
    )
    parser.add_argument("--alignment-json", required=True)
    parser.add_argument("--per-json", required=True)
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--turn-embeddings-npz", required=True)
    parser.add_argument("--semantic-chunks-npz", required=True)
    parser.add_argument("--embedding-metadata-json", required=True)
    parser.add_argument("--output-db", required=True)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace the output DB if it already exists.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = build_sqlite_database(
        output_db_path=args.output_db,
        alignment_json_path=args.alignment_json,
        per_json_path=args.per_json,
        audio_path=args.audio_path,
        turn_embeddings_npz_path=args.turn_embeddings_npz,
        semantic_chunks_npz_path=args.semantic_chunks_npz,
        embedding_metadata_json_path=args.embedding_metadata_json,
        replace=args.replace,
    )

    print(f"Built SQLite DB: {result.db_path}")
    print(f"Session id: {result.session_id}")
    for table_name, count in result.counts.items():
        print(f"{table_name}: {count}")


if __name__ == "__main__":
    main()
