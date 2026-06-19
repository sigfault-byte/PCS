from __future__ import annotations

import unittest

from assemblybot.stages.diarization_embeddings import (
    CROP_BOUNDARY_EPSILON_SECONDS,
    clamp_segment,
    expand_segment_to_min_duration,
)


class DiarizationEmbeddingBoundsTest(unittest.TestCase):
    def test_clamp_keeps_end_just_inside_audio_duration(self) -> None:
        start, end = clamp_segment(1799.5, 1800.064, 1800.064)

        self.assertEqual(start, 1799.5)
        self.assertLess(end, 1800.064)
        self.assertAlmostEqual(
            end,
            1800.064 - CROP_BOUNDARY_EPSILON_SECONDS,
        )

    def test_expand_near_audio_end_does_not_return_exact_max_end(self) -> None:
        start, end = expand_segment_to_min_duration(
            start=1799.9,
            end=1800.064,
            min_duration=0.80,
            max_end=1800.064,
        )

        self.assertLess(end, 1800.064)
        self.assertGreaterEqual(start, 0.0)
        self.assertLess(start, end)


if __name__ == "__main__":
    unittest.main()
