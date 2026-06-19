from __future__ import annotations

import unittest

from assemblybot.stages.diarization_embeddings import (
    CROP_BOUNDARY_EPSILON_SECONDS,
    clamp_segment,
    expand_segment_to_min_duration,
    max_embedding_crop_end,
)


class DiarizationEmbeddingBoundsTest(unittest.TestCase):
    def test_clamp_keeps_end_just_inside_audio_duration(self) -> None:
        start, end = clamp_segment(1799.5, 1800.064, 1800.064)

        self.assertEqual(start, 1799.064)
        self.assertLess(end, 1800.064)
        self.assertAlmostEqual(
            end,
            1800.064 - CROP_BOUNDARY_EPSILON_SECONDS,
        )
        self.assertLess(round(end, 3), 1800.064)

    def test_max_embedding_crop_end_keeps_one_second_guard_for_long_audio(self) -> None:
        self.assertAlmostEqual(max_embedding_crop_end(1800.064), 1799.064)

    def test_expand_near_audio_end_does_not_return_exact_max_end(self) -> None:
        start, end = expand_segment_to_min_duration(
            start=1799.9,
            end=1800.064,
            min_duration=0.80,
            max_end=1800.064,
        )

        self.assertAlmostEqual(end, 1799.064)
        self.assertGreaterEqual(start, 0.0)
        self.assertLess(start, end)
        self.assertAlmostEqual(end - start, 0.80)


if __name__ == "__main__":
    unittest.main()
