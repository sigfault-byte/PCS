from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from assemblybot.helper.semantic_chunking import (
    build_chunks_from_breakpoints,
    compute_adjacent_cosines,
    compute_cosine_deltas,
    count_words,
    find_semantic_breakpoints,
    split_sentences,
)
from assemblybot.models.turn_document import TurnDocument
from assemblybot.stages.semantic_chunk import (
    build_embedding_artifacts,
    build_semantic_chunks_for_turn,
)
from tests.per_test_helpers import make_turn


class FakeEmbeddingModel:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        if not normalize_embeddings:
            raise AssertionError("semantic chunk stage must normalize embeddings")

        self.calls.append(list(sentences))
        return np.asarray([self._vector_for(text) for text in sentences], dtype=np.float32)

    def _vector_for(self, text: str) -> list[float]:
        normalized = text.strip().lower()
        if normalized in {"dog", "dog dog", "dog dog dog"}:
            return [1.0, 0.0]
        if normalized in {"cat", "cat cat"}:
            return [0.0, 1.0]
        if normalized in {"bird", "bird bird"}:
            return [-1.0, 0.0]
        if not normalized:
            return [0.0, 0.0]
        return [0.6, 0.8]


class SemanticChunkHelperTest(unittest.TestCase):
    def test_split_sentences_is_deterministic_for_french_text(self) -> None:
        text = "Bonjour à tous. Ça va ? Très bien! L’État avance… oui"

        self.assertEqual(
            split_sentences(text),
            ["Bonjour à tous", "Ça va", "Très bien", "L’État avance… oui"],
        )

    def test_count_words_and_empty_sentence_filtering(self) -> None:
        self.assertEqual(count_words("  les Français  avancent vite "), 4)
        self.assertEqual(split_sentences("...!?"), [])

    def test_cosines_deltas_and_breakpoints(self) -> None:
        embeddings = np.asarray(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )

        cosines = compute_adjacent_cosines(embeddings)
        deltas = compute_cosine_deltas(cosines)

        np.testing.assert_allclose(cosines, [1.0, 1.0, 0.0, 1.0])
        np.testing.assert_allclose(deltas, [0.0, -1.0, 1.0])
        self.assertEqual(find_semantic_breakpoints(embeddings, -0.1), [3])

    def test_build_chunks_from_breakpoints_uses_exclusive_end_indices(self) -> None:
        chunks = build_chunks_from_breakpoints(
            ["dog", "dog", "dog", "cat", "cat"],
            [3],
        )

        self.assertEqual(
            chunks,
            [
                {
                    "chunk_index": 0,
                    "start_sentence_index": 0,
                    "end_sentence_index": 3,
                    "text": "dog dog dog",
                },
                {
                    "chunk_index": 1,
                    "start_sentence_index": 3,
                    "end_sentence_index": 5,
                    "text": "cat cat",
                },
            ],
        )


class SemanticChunkStageTest(unittest.TestCase):
    def test_final_chunk_embeddings_are_direct_merged_text_embeddings(self) -> None:
        model = FakeEmbeddingModel()
        text = "dog. dog. dog. cat. cat. dog. dog. bird. bird."

        chunks = build_semantic_chunks_for_turn(
            turn_id=42,
            text=text,
            model=model,
            min_words=0,
        )

        self.assertEqual(
            [(chunk.start_sentence_index, chunk.end_sentence_index, chunk.text) for chunk in chunks],
            [
                (0, 3, "dog dog dog"),
                (3, 5, "cat cat"),
                (5, 7, "dog dog"),
                (7, 9, "bird bird"),
            ],
        )

    def test_stage_writes_npz_files_and_metadata_with_expected_dtypes(self) -> None:
        model = FakeEmbeddingModel()
        document = TurnDocument(
            turns=[
                make_turn(1, "SPEAKER_1", "dog. dog. dog. cat. cat."),
                make_turn(2, "SPEAKER_2", ""),
            ],
            turns_analysis=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            input_path = tmpdir_path / "turns.json"
            output_dir = tmpdir_path / "embedding"
            input_path.write_text(
                json.dumps(document.to_dict(), ensure_ascii=False),
                encoding="utf-8",
            )

            artifacts = build_embedding_artifacts(
                input_path,
                model=model,
                output_dir=output_dir,
                min_words=0,
            )

            turn_npz = np.load(artifacts.turn_embeddings_path)
            chunk_npz = np.load(artifacts.semantic_chunks_path)
            metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
            metadata_txt = artifacts.metadata_txt_path.read_text(encoding="utf-8")

            self.assertEqual(turn_npz["turn_ids"].dtype, np.int32)
            self.assertEqual(turn_npz["embeddings"].dtype, np.float32)
            self.assertEqual(turn_npz["embeddings"].shape, (2, 2))
            self.assertEqual(chunk_npz["turn_ids"].dtype, np.int32)
            self.assertEqual(chunk_npz["chunk_indices"].dtype, np.int16)
            self.assertEqual(chunk_npz["start_sentence_indices"].dtype, np.int16)
            self.assertEqual(chunk_npz["end_sentence_indices"].dtype, np.int16)
            self.assertEqual(chunk_npz["embeddings"].dtype, np.float32)
            np.testing.assert_array_equal(chunk_npz["turn_ids"], np.asarray([1, 1]))
            np.testing.assert_array_equal(
                chunk_npz["start_sentence_indices"],
                np.asarray([0, 3], dtype=np.int16),
            )
            np.testing.assert_array_equal(
                chunk_npz["end_sentence_indices"],
                np.asarray([3, 5], dtype=np.int16),
            )

            self.assertEqual(metadata["model_name"], "h4c5/sts-camembert-base")
            self.assertEqual(metadata["dtype"], "float32")
            self.assertTrue(metadata["normalize_embeddings"])
            self.assertEqual(metadata["chunking_method"], "semantic_delta_v1")
            self.assertEqual(metadata["min_words"], 0)
            self.assertIn("model_name: h4c5/sts-camembert-base", metadata_txt)
            self.assertIn("chunking_method: semantic_delta_v1", metadata_txt)
            self.assertIn("min_words: 0", metadata_txt)

            self.assertIn(["dog", "dog", "dog", "cat", "cat"], model.calls)
            self.assertIn(["dog dog dog", "cat cat"], model.calls)


if __name__ == "__main__":
    unittest.main()
