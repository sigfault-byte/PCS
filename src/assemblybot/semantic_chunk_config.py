from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SemanticChunkConfig:
    """Runtime and artifact settings for semantic chunk embeddings."""

    model_name: str = "h4c5/sts-camembert-base"
    normalize_embeddings: bool = True
    embedding_dtype: type[np.floating] = np.float32
    delta_threshold: float = -0.1
    min_words: int = 8
    chunking_method: str = "semantic_delta_v1"
    turn_embeddings_filename: str = "turn_embeddings.npz"
    semantic_chunks_filename: str = "semantic_chunks.npz"
    metadata_filename: str = "semantic_chunk_metadata.json"
    metadata_txt_filename: str = "semantic_chunk_metadata.txt"

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "SemanticChunkConfig":
        return cls()


DEFAULT_SEMANTIC_CHUNK_CONFIG = SemanticChunkConfig()


def add_semantic_chunk_arguments(parser: argparse.ArgumentParser) -> None:
    """Register semantic chunking CLI options on a stage parser."""
