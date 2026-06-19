from __future__ import annotations

import unittest
from pathlib import Path

from assemblybot.alignment_config import DEFAULT_ALIGNMENT_CONFIG
from assemblybot.models.diarization import DiarizationRawSegment
from assemblybot.models.factories import create_empty_document
from assemblybot.models.time import TimeRange
from assemblybot.models.transcript import TranscriptRawSegment
from assemblybot.stages.alignment import build_transcript_diarization_matches


class AlignmentStageTest(unittest.TestCase):
    def test_missing_diarization_raises_clear_error(self) -> None:
        document = create_empty_document(Path("audio.wav"))
        document.transcript.raw_segments = [
            TranscriptRawSegment(
                segment_id=1,
                start_token_id=None,
                end_token_id=None,
                time=TimeRange.from_seconds(0.0, 1.0),
                raw_text="Bonjour",
            )
        ]

        with self.assertRaisesRegex(ValueError, "diarization.raw_segments"):
            build_transcript_diarization_matches(
                document,
                DEFAULT_ALIGNMENT_CONFIG,
            )

    def test_builds_match_when_transcript_and_diarization_exist(self) -> None:
        document = create_empty_document(Path("audio.wav"))
        document.transcript.raw_segments = [
            TranscriptRawSegment(
                segment_id=1,
                start_token_id=None,
                end_token_id=None,
                time=TimeRange.from_seconds(0.0, 1.0),
                raw_text="Bonjour",
            )
        ]
        document.diarization.raw_segments = [
            DiarizationRawSegment(
                segment_id=1,
                time=TimeRange.from_seconds(0.0, 1.0),
                speaker_id="SPEAKER_00",
            )
        ]

        matches = build_transcript_diarization_matches(
            document,
            DEFAULT_ALIGNMENT_CONFIG,
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].diarization_segment_ids, [1])
        self.assertEqual(matches[0].probable_speaker_id, "SPEAKER_00")


if __name__ == "__main__":
    unittest.main()
