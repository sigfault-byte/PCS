from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from assemblybot.config import DEFAULT_SQLITE_DB_PATH, EMBEDDING_DIR
from assemblybot.semantic_chunk_config import DEFAULT_SEMANTIC_CHUNK_CONFIG


@dataclass(frozen=True)
class BuildSqliteConfig:
    """Default paths and safety settings for building the SQLite database."""

    output_db_path: Path = DEFAULT_SQLITE_DB_PATH
    turn_embeddings_npz_path: Path = (
        EMBEDDING_DIR / DEFAULT_SEMANTIC_CHUNK_CONFIG.turn_embeddings_filename
    )
    semantic_chunks_npz_path: Path = (
        EMBEDDING_DIR / DEFAULT_SEMANTIC_CHUNK_CONFIG.semantic_chunks_filename
    )
    embedding_metadata_json_path: Path = (
        EMBEDDING_DIR / DEFAULT_SEMANTIC_CHUNK_CONFIG.metadata_filename
    )
    replace_existing_db: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "BuildSqliteConfig":
        return cls(
            output_db_path=Path(args.output_db),
            turn_embeddings_npz_path=Path(args.turn_embeddings_npz),
            semantic_chunks_npz_path=Path(args.semantic_chunks_npz),
            embedding_metadata_json_path=Path(args.embedding_metadata_json),
            replace_existing_db=args.replace,
        )


DEFAULT_BUILD_SQLITE_CONFIG = BuildSqliteConfig()


def add_build_sqlite_arguments(parser: argparse.ArgumentParser) -> None:
    """Register build-sqlite artifact defaults on a stage parser."""
    parser.add_argument(
        "--turn-embeddings-npz",
        default=str(DEFAULT_BUILD_SQLITE_CONFIG.turn_embeddings_npz_path),
        help="Turn embeddings NPZ path",
    )
    parser.add_argument(
        "--semantic-chunks-npz",
        default=str(DEFAULT_BUILD_SQLITE_CONFIG.semantic_chunks_npz_path),
        help="Semantic chunks NPZ path",
    )
    parser.add_argument(
        "--embedding-metadata-json",
        default=str(DEFAULT_BUILD_SQLITE_CONFIG.embedding_metadata_json_path),
        help="Semantic chunk embedding metadata JSON path",
    )
    parser.add_argument(
        "--output-db",
        default=str(DEFAULT_BUILD_SQLITE_CONFIG.output_db_path),
        help="SQLite output path",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        default=DEFAULT_BUILD_SQLITE_CONFIG.replace_existing_db,
        help="Replace the output DB if it already exists.",
    )
