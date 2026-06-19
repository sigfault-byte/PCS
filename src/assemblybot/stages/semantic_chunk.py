from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from assemblybot.config import DATA_DIR
from assemblybot.helper.artifact import save_npz
from assemblybot.helper.semantic_chunking import (
    build_chunks_from_breakpoints,
    count_words,
    find_semantic_breakpoints,
    split_sentences,
)
from assemblybot.models.turn_document import TurnDocument

MODEL_NAME = "h4c5/sts-camembert-base"
OUTPUT_DIR = DATA_DIR / "embedding"
DELTA_THRESHOLD = -0.1
MIN_WORDS = 8
CHUNKING_METHOD = "semantic_delta_v1"

TURN_EMBEDDINGS_FILENAME = "turn_embeddings.npz"
SEMANTIC_CHUNKS_FILENAME = "semantic_chunks.npz"
METADATA_FILENAME = "semantic_chunk_metadata.json"
METADATA_TXT_FILENAME = "semantic_chunk_metadata.txt"


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


def encode_texts(model: EmbeddingModel, texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    embeddings = model.encode(texts, normalize_embeddings=True)
    return np.asarray(embeddings, dtype=np.float32)


def semantic_breakpoints_for_turn(
    sentences: list[str],
    model: EmbeddingModel,
    *,
    threshold: float = DELTA_THRESHOLD,
    min_words: int = MIN_WORDS,
) -> list[int]:
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
    sentence_embeddings = encode_texts(model, usable_texts)
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
    threshold: float = DELTA_THRESHOLD,
    min_words: int = MIN_WORDS,
) -> list[SemanticChunk]:
    sentences = split_sentences(text)
    if not sentences:
        return []

    breakpoints = semantic_breakpoints_for_turn(
        sentences,
        model,
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
    threshold: float = DELTA_THRESHOLD,
    min_words: int = MIN_WORDS,
) -> list[SemanticChunk]:
    chunks: list[SemanticChunk] = []
    for turn in document.turns:
        chunks.extend(
            build_semantic_chunks_for_turn(
                turn_id=turn.turn_id,
                text=turn.text,
                model=model,
                threshold=threshold,
                min_words=min_words,
            )
        )
    return chunks


def save_turn_embeddings(
    document: TurnDocument,
    model: EmbeddingModel,
    output_path: Path,
) -> None:
    turn_ids = np.asarray([turn.turn_id for turn in document.turns], dtype=np.int32)
    turn_texts = [turn.text for turn in document.turns]
    embeddings = encode_texts(model, turn_texts)
    save_npz(output_path, turn_ids=turn_ids, embeddings=embeddings)


def save_semantic_chunks(
    chunks: list[SemanticChunk],
    model: EmbeddingModel,
    output_path: Path,
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
    embeddings = encode_texts(model, [chunk.text for chunk in chunks])
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
    model_name: str = MODEL_NAME,
    threshold: float = DELTA_THRESHOLD,
    min_words: int = MIN_WORDS,
) -> None:
    metadata = {
        "model_name": model_name,
        "dtype": "float32",
        "normalize_embeddings": True,
        "chunking_method": CHUNKING_METHOD,
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
    output_dir: Path = OUTPUT_DIR,
    threshold: float = DELTA_THRESHOLD,
    min_words: int = MIN_WORDS,
) -> SemanticEmbeddingArtifacts:
    document = load_turn_document(input_json_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    turn_embeddings_path = output_dir / TURN_EMBEDDINGS_FILENAME
    semantic_chunks_path = output_dir / SEMANTIC_CHUNKS_FILENAME
    metadata_path = output_dir / METADATA_FILENAME
    metadata_txt_path = output_dir / METADATA_TXT_FILENAME

    save_turn_embeddings(document, model, turn_embeddings_path)
    chunks = build_semantic_chunks(
        document,
        model,
        threshold=threshold,
        min_words=min_words,
    )
    save_semantic_chunks(chunks, model, semantic_chunks_path)
    save_metadata(
        input_json_path=input_json_path,
        output_dir=output_dir,
        turn_embeddings_path=turn_embeddings_path,
        semantic_chunks_path=semantic_chunks_path,
        metadata_path=metadata_path,
        metadata_txt_path=metadata_txt_path,
        threshold=threshold,
        min_words=min_words,
    )

    return SemanticEmbeddingArtifacts(
        turn_embeddings_path=turn_embeddings_path,
        semantic_chunks_path=semantic_chunks_path,
        metadata_path=metadata_path,
        metadata_txt_path=metadata_txt_path,
    )


def load_sentence_transformer(model_name: str = MODEL_NAME) -> EmbeddingModel:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build full-turn and semantic chunk embeddings from a TurnDocument JSON."
    )
    parser.add_argument(
        "--input_json",
        help="TurnDocument JSON path, usually an _02_per_extraction.json artifact",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_json_path = Path(args.input_json).resolve()
    model = load_sentence_transformer()
    artifacts = build_embedding_artifacts(input_json_path, model=model)

    print(f"Turn embeddings: {artifacts.turn_embeddings_path}")
    print(f"Semantic chunks: {artifacts.semantic_chunks_path}")
    print(f"Metadata: {artifacts.metadata_path}")
    print(f"Metadata text: {artifacts.metadata_txt_path}")


if __name__ == "__main__":
    main()
