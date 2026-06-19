from __future__ import annotations

import re
from typing import Any

import numpy as np


SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
WORD_RE = re.compile(r"\S+")


def split_sentences(text: str) -> list[str]:
    """Split text into deterministic punctuation-delimited sentence candidates."""
    return [sentence.strip() for sentence in SENTENCE_SPLIT_RE.split(text) if sentence.strip()]


def count_words(text: str) -> int:
    """Count whitespace-delimited word-like tokens deterministically."""
    return len(WORD_RE.findall(text))


def compute_adjacent_cosines(embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine scores between consecutive normalized embeddings."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.shape[0] < 2:
        return np.empty((0,), dtype=np.float32)

    return np.sum(embeddings[:-1] * embeddings[1:], axis=1, dtype=np.float32).astype(
        np.float32,
        copy=False,
    )


def compute_cosine_deltas(cosines: np.ndarray) -> np.ndarray:
    """Compute consecutive cosine differences."""
    cosines = np.asarray(cosines, dtype=np.float32)
    if cosines.shape[0] < 2:
        return np.empty((0,), dtype=np.float32)

    return np.diff(cosines).astype(np.float32, copy=False)


def find_semantic_breakpoints(
    sentence_embeddings: np.ndarray,
    threshold: float = -0.1,
) -> list[int]:
    """Return sentence indices where a new semantic chunk should start."""
    cosines = compute_adjacent_cosines(sentence_embeddings)
    deltas = compute_cosine_deltas(cosines)
    return [index + 2 for index, delta in enumerate(deltas) if float(delta) < threshold]


def build_chunks_from_breakpoints(
    sentences: list[str],
    breakpoints: list[int],
) -> list[dict[str, Any]]:
    """Build start-inclusive/end-exclusive chunk ranges from sentence breakpoints."""
    if not sentences:
        return []

    chunks: list[dict[str, Any]] = []
    start = 0

    for breakpoint in breakpoints:
        if breakpoint <= start or breakpoint >= len(sentences):
            continue

        chunks.append(
            {
                "chunk_index": len(chunks),
                "start_sentence_index": start,
                "end_sentence_index": breakpoint,
                "text": " ".join(sentences[start:breakpoint]),
            }
        )
        start = breakpoint

    chunks.append(
        {
            "chunk_index": len(chunks),
            "start_sentence_index": start,
            "end_sentence_index": len(sentences),
            "text": " ".join(sentences[start:]),
        }
    )
    return chunks
