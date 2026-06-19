from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from assemblybot.alignment_config import DEFAULT_ALIGNMENT_CONFIG
from assemblybot.helper.document import load_document, save_document
from assemblybot.models.diarization import DiarizationRawSegment
from assemblybot.models.factories import create_empty_document
from assemblybot.models.time import TimeRange
from assemblybot.models.transcript import TranscriptRawSegment
from assemblybot.models.vad import VadSegment
from assemblybot.stages.alignment import build_transcript_diarization_matches
from assemblybot.stages.whisper_segment_audit import audit_whisper_segments


def write_minimal_audio_audit(path: Path) -> None:
    frame = {
        "frame_center_seconds": 0.5,
        "rms": 0.1,
        "db": -20.0,
        "zcr": 0.01,
        "spectral_centroid": 1000.0,
        "spectral_bandwidth": 200.0,
        "spectral_flatness": 0.1,
        "db_rolling_median": -20.0,
        "db_delta": 0.0,
    }
    path.write_text(
        json.dumps(
            {
                "schema_version": "audio_audit.v1",
                "frames": [frame],
            }
        ),
        encoding="utf-8",
    )


class WhisperSegmentAuditStageTest(unittest.TestCase):
    def test_audit_output_preserves_aux_diarization_for_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            transcript_path = tmpdir_path / "transcript.json"
            aux_path = tmpdir_path / "aux.json"
            audit_path = tmpdir_path / "audio_audit.json"
            output_path = tmpdir_path / "flagged.json"
            sidecar_path = tmpdir_path / "segment_audit.json"

            transcript_doc = create_empty_document(tmpdir_path / "audio.wav")
            transcript_doc.transcript.raw_segments = [
                TranscriptRawSegment(
                    segment_id=1,
                    start_token_id=None,
                    end_token_id=None,
                    time=TimeRange.from_seconds(0.0, 1.0),
                    raw_text="Bonjour",
                )
            ]

            aux_doc = create_empty_document(tmpdir_path / "audio.wav")
            aux_doc.vad.segments = [
                VadSegment(
                    segment_id=1,
                    time=TimeRange.from_seconds(0.0, 1.0),
                )
            ]
            aux_doc.diarization.raw_segments = [
                DiarizationRawSegment(
                    segment_id=1,
                    time=TimeRange.from_seconds(0.0, 1.0),
                    speaker_id="SPEAKER_00",
                )
            ]
            aux_doc.diarization.speakers_count = 1

            save_document(transcript_doc, transcript_path)
            save_document(aux_doc, aux_path)
            write_minimal_audio_audit(audit_path)

            audit_whisper_segments(
                transcript_path=transcript_path,
                aux_path=aux_path,
                audio_audit_path=audit_path,
                output_path=output_path,
                write_sidecar=True,
                sidecar_output_path=sidecar_path,
            )

            output_doc = load_document(output_path)
            self.assertEqual(len(output_doc.transcript.raw_segments), 1)
            self.assertEqual(len(output_doc.vad.segments), 1)
            self.assertEqual(len(output_doc.diarization.raw_segments), 1)

            matches = build_transcript_diarization_matches(
                output_doc,
                DEFAULT_ALIGNMENT_CONFIG,
            )

            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0].diarization_segment_ids, [1])
            self.assertEqual(matches[0].probable_speaker_id, "SPEAKER_00")


if __name__ == "__main__":
    unittest.main()
