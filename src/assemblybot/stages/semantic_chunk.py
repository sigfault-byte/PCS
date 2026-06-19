from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from assemblybot.config import EMBEDDING_DIR
from assemblybot.helper.artifact import save_npz
from assemblybot.helper.semantic_chunking import (
    build_chunks_from_breakpoints,
    count_words,
    find_semantic_breakpoints,
    split_sentences,
)
from assemblybot.models.turn_document import TurnDocument
from assemblybot.semantic_chunk_config import (
    DEFAULT_SEMANTIC_CHUNK_CONFIG,
    SemanticChunkConfig,
    add_semantic_chunk_arguments,
)

MODEL_NAME = DEFAULT_SEMANTIC_CHUNK_CONFIG.model_name
DELTA_THRESHOLD = DEFAULT_SEMANTIC_CHUNK_CONFIG.delta_threshold
MIN_WORDS = DEFAULT_SEMANTIC_CHUNK_CONFIG.min_words
CHUNKING_METHOD = DEFAULT_SEMANTIC_CHUNK_CONFIG.chunking_method

TURN_EMBEDDINGS_FILENAME = DEFAULT_SEMANTIC_CHUNK_CONFIG.turn_embeddings_filename
SEMANTIC_CHUNKS_FILENAME = DEFAULT_SEMANTIC_CHUNK_CONFIG.semantic_chunks_filename
METADATA_FILENAME = DEFAULT_SEMANTIC_CHUNK_CONFIG.metadata_filename
METADATA_TXT_FILENAME = DEFAULT_SEMANTIC_CHUNK_CONFIG.metadata_txt_filename


class EmbeddingModel(Protocol):
    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class SemanticChunk:
    turn_id: int
    chunk_index: int
    start_sentence_index: int
    end_sentence_index: int
    text: str


@dataclass(frozen=True)
class SemanticEmbeddingArtifacts:
    turn_embeddings_path: Path
    semantic_chunks_path: Path
    metadata_path: Path
    metadata_txt_path: Path


def load_turn_document(json_path: Path) -> TurnDocument:
    with json_path.open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)
    return TurnDocument.from_dict(data)


def encode_texts(
    model: EmbeddingModel,
    texts: list[str],
    config: SemanticChunkConfig = DEFAULT_SEMANTIC_CHUNK_CONFIG,
) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=config.embedding_dtype)

    embeddings = model.encode(
        texts,
        normalize_embeddings=config.normalize_embeddings,
    )
    return np.asarray(embeddings, dtype=config.embedding_dtype)


def semantic_breakpoints_for_turn(
    sentences: list[str],
    model: EmbeddingModel,
    *,
    config: SemanticChunkConfig = DEFAULT_SEMANTIC_CHUNK_CONFIG,
    threshold: float | None = None,
    min_words: int | None = None,
) -> list[int]:
    threshold = config.delta_threshold if threshold is None else threshold
    min_words = config.min_words if min_words is None else min_words

    if len(sentences) < 2:
        return []

    usable = [
        (index, sentence)
        for index, sentence in enumerate(sentences)
        if min_words <= 0 or count_words(sentence) >= min_words
    ]
    if len(usable) < 2:
        return []

    usable_indices = [index for index, _sentence in usable]
    usable_texts = [sentence for _index, sentence in usable]
    sentence_embeddings = encode_texts(model, usable_texts, config)
    usable_breakpoints = find_semantic_breakpoints(sentence_embeddings, threshold)

    return [
        usable_indices[breakpoint]
        for breakpoint in usable_breakpoints
        if 0 < breakpoint < len(usable_indices)
    ]


def build_semantic_chunks_for_turn(
    *,
    turn_id: int,
    text: str,
    model: EmbeddingModel,
    config: SemanticChunkConfig = DEFAULT_SEMANTIC_CHUNK_CONFIG,
    threshold: float | None = None,
    min_words: int | None = None,
) -> list[SemanticChunk]:
    sentences = split_sentences(text)
    if not sentences:
        return []

    breakpoints = semantic_breakpoints_for_turn(
        sentences,
        model,
        config=config,
        threshold=threshold,
        min_words=min_words,
    )
    raw_chunks = build_chunks_from_breakpoints(sentences, breakpoints)

    return [
        SemanticChunk(
            turn_id=turn_id,
            chunk_index=int(chunk["chunk_index"]),
            start_sentence_index=int(chunk["start_sentence_index"]),
            end_sentence_index=int(chunk["end_sentence_index"]),
            text=str(chunk["text"]),
        )
        for chunk in raw_chunks
    ]


def build_semantic_chunks(
    document: TurnDocument,
    model: EmbeddingModel,
    *,
    config: SemanticChunkConfig = DEFAULT_SEMANTIC_CHUNK_CONFIG,
    threshold: float | None = None,
    min_words: int | None = None,
) -> list[SemanticChunk]:
    chunks: list[SemanticChunk] = []
    for turn in document.turns:
        chunks.extend(
            build_semantic_chunks_for_turn(
                turn_id=turn.turn_id,
                text=turn.text,
                model=model,
                config=config,
                threshold=threshold,
                min_words=min_words,
            )
        )
    return chunks


def save_turn_embeddings(
    document: TurnDocument,
    model: EmbeddingModel,
    output_path: Path,
    config: SemanticChunkConfig = DEFAULT_SEMANTIC_CHUNK_CONFIG,
) -> None:
    turn_ids = np.asarray([turn.turn_id for turn in document.turns], dtype=np.int32)
    turn_texts = [turn.text for turn in document.turns]
    embeddings = encode_texts(model, turn_texts, config)
    save_npz(output_path, turn_ids=turn_ids, embeddings=embeddings)


def save_semantic_chunks(
    chunks: list[SemanticChunk],
    model: EmbeddingModel,
    output_path: Path,
    config: SemanticChunkConfig = DEFAULT_SEMANTIC_CHUNK_CONFIG,
) -> None:
    turn_ids = np.asarray([chunk.turn_id for chunk in chunks], dtype=np.int32)
    chunk_indices = np.asarray(
        [chunk.chunk_index for chunk in chunks],
        dtype=np.int16,
    )
    start_sentence_indices = np.asarray(
        [chunk.start_sentence_index for chunk in chunks],
        dtype=np.int16,
    )
    end_sentence_indices = np.asarray(
        [chunk.end_sentence_index for chunk in chunks],
        dtype=np.int16,
    )

    # Sentence embeddings only find boundaries. Retrieval vectors are direct
    # embeddings of merged chunk text.
    embeddings = encode_texts(model, [chunk.text for chunk in chunks], config)
    save_npz(
        output_path,
        turn_ids=turn_ids,
        chunk_indices=chunk_indices,
        start_sentence_indices=start_sentence_indices,
        end_sentence_indices=end_sentence_indices,
        embeddings=embeddings,
    )


def save_metadata(
    *,
    input_json_path: Path,
    output_dir: Path,
    turn_embeddings_path: Path,
    semantic_chunks_path: Path,
    metadata_path: Path,
    metadata_txt_path: Path | None = None,
    config: SemanticChunkConfig = DEFAULT_SEMANTIC_CHUNK_CONFIG,
    threshold: float | None = None,
    min_words: int | None = None,
) -> None:
    threshold = config.delta_threshold if threshold is None else threshold
    min_words = config.min_words if min_words is None else min_words
    metadata = {
        "model_name": config.model_name,
        "dtype": np.dtype(config.embedding_dtype).name,
        "normalize_embeddings": config.normalize_embeddings,
        "chunking_method": config.chunking_method,
        "delta_threshold": threshold,
        "min_words": min_words,
        "sentence_index_policy": "start inclusive, end exclusive",
        "input_json_path": str(input_json_path),
        "output_dir": str(output_dir),
        "turn_embeddings_npz": str(turn_embeddings_path),
        "semantic_chunks_npz": str(semantic_chunks_path),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as output_file:
        json.dump(metadata, output_file, ensure_ascii=False, indent=2)

    if metadata_txt_path is not None:
        metadata_txt_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "Semantic chunk embedding metadata",
            "",
            f"model_name: {metadata['model_name']}",
            f"dtype: {metadata['dtype']}",
            f"normalize_embeddings: {metadata['normalize_embeddings']}",
            f"chunking_method: {metadata['chunking_method']}",
            f"delta_threshold: {metadata['delta_threshold']}",
            f"min_words: {metadata['min_words']}",
            f"sentence_index_policy: {metadata['sentence_index_policy']}",
            f"input_json_path: {metadata['input_json_path']}",
            f"output_dir: {metadata['output_dir']}",
            f"turn_embeddings_npz: {metadata['turn_embeddings_npz']}",
            f"semantic_chunks_npz: {metadata['semantic_chunks_npz']}",
        ]
        metadata_txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_embedding_artifacts(
    input_json_path: Path,
    *,
    model: EmbeddingModel,
    output_dir: Path = EMBEDDING_DIR,
    config: SemanticChunkConfig = DEFAULT_SEMANTIC_CHUNK_CONFIG,
    threshold: float | None = None,
    min_words: int | None = None,
) -> SemanticEmbeddingArtifacts:
    document = load_turn_document(input_json_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    turn_embeddings_path = output_dir / config.turn_embeddings_filename
    semantic_chunks_path = output_dir / config.semantic_chunks_filename
    metadata_path = output_dir / config.metadata_filename
    metadata_txt_path = output_dir / config.metadata_txt_filename

    save_turn_embeddings(document, model, turn_embeddings_path, config)
    chunks = build_semantic_chunks(
        document,
        model,
        config=config,
        threshold=threshold,
        min_words=min_words,
    )
    save_semantic_chunks(chunks, model, semantic_chunks_path, config)
    save_metadata(
        input_json_path=input_json_path,
        output_dir=output_dir,
        turn_embeddings_path=turn_embeddings_path,
        semantic_chunks_path=semantic_chunks_path,
        metadata_path=metadata_path,
        metadata_txt_path=metadata_txt_path,
        config=config,
        threshold=threshold,
        min_words=min_words,
    )

    return SemanticEmbeddingArtifacts(
        turn_embeddings_path=turn_embeddings_path,
        semantic_chunks_path=semantic_chunks_path,
        metadata_path=metadata_path,
        metadata_txt_path=metadata_txt_path,
    )


def load_sentence_transformer(
    config: SemanticChunkConfig = DEFAULT_SEMANTIC_CHUNK_CONFIG,
) -> EmbeddingModel:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.model_name)  # type: ignore


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build full-turn and semantic chunk embeddings from a TurnDocument JSON."
    )
    parser.add_argument(
        "--input-json",
        required=True,
        help="TurnDocument JSON path, usually an _02_per_extraction.json artifact",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional output directory for embedding artifacts",
    )
    add_semantic_chunk_arguments(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = SemanticChunkConfig.from_args(args)
    input_json_path = Path(args.input_json).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else EMBEDDING_DIR
    )
    model = load_sentence_transformer(config)
    artifacts = build_embedding_artifacts(
        input_json_path,
        model=model,
        output_dir=output_dir,
        config=config,
    )

    print(f"Turn embeddings: {artifacts.turn_embeddings_path}")
    print(f"Semantic chunks: {artifacts.semantic_chunks_path}")
    print(f"Metadata: {artifacts.metadata_path}")
    print(f"Metadata text: {artifacts.metadata_txt_path}")


if __name__ == "__main__":
    main()
