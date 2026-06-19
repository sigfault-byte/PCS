from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PyannoteDiarizationConfig:
    """Runtime settings for the pyannote diarization stage."""

    diarization_model_name: str = "pyannote/speaker-diarization-3.1"
    embedding_model_name: str = "pyannote/embedding"
    extract_embeddings: bool = True
    device: str = "auto"
    min_embedding_duration_seconds: float = 0.80
    skip_embeddings_below_seconds: float = 0.08
    embedding_dtype: type[np.floating] = np.float32

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "PyannoteDiarizationConfig":
        return cls(
            diarization_model_name=args.diarization_model,
            embedding_model_name=args.embedding_model,
            extract_embeddings=not args.no_embeddings,
            device=args.device,
        )


DEFAULT_PYANNOTE_DIARIZATION_CONFIG = PyannoteDiarizationConfig()


def add_pyannote_diarization_arguments(parser: argparse.ArgumentParser) -> None:
    """Register pyannote-specific CLI options on a stage parser."""
    parser.add_argument(
        "--diarization-model",
        default=DEFAULT_PYANNOTE_DIARIZATION_CONFIG.diarization_model_name,
        help="Pyannote diarization model name",
    )

    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_PYANNOTE_DIARIZATION_CONFIG.embedding_model_name,
        help="Pyannote embedding model name",
    )

    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        default=not DEFAULT_PYANNOTE_DIARIZATION_CONFIG.extract_embeddings,
        help="Skip embedding extraction, centroid computation, and NPZ artifacts",
    )

    parser.add_argument(
        "--device",
        default=DEFAULT_PYANNOTE_DIARIZATION_CONFIG.device,
        help="Device to use: auto, cpu, cuda",
    )
