from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from assemblybot.orchestration import paths as orchestration_paths
from assemblybot.orchestration import queue as orchestration_queue
from assemblybot.orchestration.provenance import Provenance, build_provenance
from assemblybot.orchestration.runner import PipelineRunner


class QueueDirectoryPatch:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.audio = root / "audio"
        self.unprocessed = self.audio / "unprocessed"
        self.processing = self.audio / "processing"
        self.processed = self.audio / "processed"
        self.failed = self.audio / "failed"
        self.runs = root / "runs"

    def __enter__(self):
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(
            patch.object(orchestration_paths, "AUDIO_DIR", self.audio)
        )
        self.stack.enter_context(
            patch.object(orchestration_paths, "UNPROCESSED_DIR", self.unprocessed)
        )
        self.stack.enter_context(
            patch.object(orchestration_paths, "PROCESSING_DIR", self.processing)
        )
        self.stack.enter_context(
            patch.object(orchestration_paths, "PROCESSED_DIR", self.processed)
        )
        self.stack.enter_context(
            patch.object(orchestration_paths, "FAILED_DIR", self.failed)
        )
        self.stack.enter_context(
            patch.object(orchestration_paths, "RUNS_DIR", self.runs)
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stack.close()


class FakePipelineRunner(PipelineRunner):
    def __init__(self, *args, fail_stage: str | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fail_stage = fail_stage
        self.called_stages: list[str] = []

    def _write(self, stage_name: str, path: Path, payload: str = "{}\n") -> None:
        self.called_stages.append(stage_name)
        if self.fail_stage == stage_name:
            raise RuntimeError(f"boom in {stage_name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")

    def stage_vad(self, document, paths):
        self._write("vad.py", paths.vad_json)
        return {"canonical_json": paths.vad_json}

    def stage_diarization(self, document, paths):
        self._write("diarization.py", paths.diarization_json)
        paths.diarization_segment_embeddings_npz.write_bytes(b"segments")
        paths.diarization_speaker_centroids_npz.write_bytes(b"centroids")
        return {
            "canonical_json": paths.diarization_json,
            "segment_embeddings_npz": paths.diarization_segment_embeddings_npz,
            "speaker_centroids_npz": paths.diarization_speaker_centroids_npz,
        }

    def stage_transcription(self, document, paths):
        self._write("transcription.py", paths.transcription_json)
        paths.transcript_txt.write_text("transcript\n", encoding="utf-8")
        return {
            "canonical_json": paths.transcription_json,
            "transcript_txt": paths.transcript_txt,
        }

    def stage_audio_audit(self, paths):
        self._write("audio_audit.py", paths.audio_audit_json)
        return {"audio_audit_json": paths.audio_audit_json}

    def stage_whisper_segment_audit(self, paths):
        self._write("whisper_segment_audit.py", paths.flagged_transcription_json)
        paths.segment_audio_audit_json.write_text("{}\n", encoding="utf-8")
        return {
            "canonical_json": paths.flagged_transcription_json,
            "segment_audio_audit_json": paths.segment_audio_audit_json,
        }

    def stage_alignment(self, paths):
        self._write("alignment.py", paths.alignment_json)
        return {"canonical_json": paths.alignment_json}

    def stage_turns(self, paths):
        self._write("turns.py", paths.turns_json)
        return {"turns_json": paths.turns_json}

    def stage_per_extraction(self, paths):
        self._write("per_extraction.py", paths.per_extraction_json)
        return {"per_json": paths.per_extraction_json}

    def stage_semantic_chunk(self, paths):
        self.called_stages.append("semantic_chunk.py")
        if self.fail_stage == "semantic_chunk.py":
            raise RuntimeError("boom in semantic_chunk.py")
        paths.turn_embeddings_npz.write_bytes(b"turns")
        paths.semantic_chunks_npz.write_bytes(b"chunks")
        paths.semantic_chunk_metadata_json.write_text("{}\n", encoding="utf-8")
        paths.semantic_chunk_metadata_txt.write_text("metadata\n", encoding="utf-8")
        return {
            "turn_embeddings_npz": paths.turn_embeddings_npz,
            "semantic_chunks_npz": paths.semantic_chunks_npz,
            "metadata_json": paths.semantic_chunk_metadata_json,
            "metadata_txt": paths.semantic_chunk_metadata_txt,
        }

    def stage_build_sqlite(self, paths):
        self.called_stages.append("build_sqlite.py")
        if self.fail_stage == "build_sqlite.py":
            raise RuntimeError("boom in build_sqlite.py")
        paths.sqlite_db.parent.mkdir(parents=True, exist_ok=True)
        paths.sqlite_db.write_bytes(b"sqlite")
        return {"sqlite_db": paths.sqlite_db}


class PipelineOrchestrationTest(unittest.TestCase):
    def make_runner(self, *, fail_stage: str | None = None) -> FakePipelineRunner:
        return FakePipelineRunner(
            language="fr",
            deputies_ground_truth_csv=Path(
                "docs/liste_deputes_libre_office_2026-06.csv"
            ),
            ministers_ground_truth_csv=Path("docs/liste_ministre_2026.csv"),
            replace_sqlite=True,
            fail_stage=fail_stage,
        )

    def test_discover_candidates_creates_directories_and_sorts_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with QueueDirectoryPatch(Path(tmpdir)) as dirs:
                dirs.unprocessed.mkdir(parents=True)
                (dirs.unprocessed / "b.wav").write_bytes(b"b")
                (dirs.unprocessed / "a.wav").write_bytes(b"a")

                candidates = orchestration_queue.discover_candidates()

                self.assertEqual([path.name for path in candidates], ["a.wav", "b.wav"])
                self.assertTrue(dirs.processing.is_dir())
                self.assertTrue(dirs.processed.is_dir())
                self.assertTrue(dirs.failed.is_dir())
                self.assertTrue(dirs.runs.is_dir())

    def test_wav_validation_is_case_insensitive(self) -> None:
        self.assertTrue(orchestration_queue.is_wav_file(Path("session.WAV")))
        self.assertTrue(orchestration_queue.is_wav_file(Path("session.wav")))
        self.assertFalse(orchestration_queue.is_wav_file(Path("session.mp3")))

    def test_provenance_date_extraction_and_slugging(self) -> None:
        cases = [
            ("session-2026-06-19-title.wav", "2026-06-19", "session-title"),
            ("20260619_qag_simplification.wav", "2026-06-19", "qag-simplification"),
            (
                "1ere-seance-14-avril-2026-qag.wav",
                "2026-04-14",
                "1ere-seance-qag",
            ),
            ("plain session.wav", None, "plain-session"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            for filename, expected_date, expected_slug in cases:
                path = Path(tmpdir) / filename
                path.write_bytes(b"abc")
                provenance = build_provenance(
                    path,
                    ingestion_timestamp="2026-06-19T12:00:00+00:00",
                )

                self.assertEqual(provenance.detected_date, expected_date)
                self.assertEqual(provenance.detected_title_session_slug, expected_slug)
                self.assertEqual(provenance.file_size_bytes, 3)
                self.assertEqual(provenance.sha256, hashlib.sha256(b"abc").hexdigest())

    def test_run_directory_name_matches_required_pattern(self) -> None:
        provenance = Provenance(
            original_filename="session.wav",
            file_stem="session",
            detected_date=None,
            detected_title_session_slug="assembly-title",
            file_size_bytes=3,
            sha256="abcdef123456",
            ingestion_timestamp="2026-06-19T12:00:00+00:00",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = orchestration_paths.build_run_dir(provenance, Path(tmpdir))

            self.assertEqual(
                run_dir.name,
                "2026-06-19_no-date_assembly-title_abcdef12",
            )

    def test_invalid_non_wav_creates_manifest_and_moves_to_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with QueueDirectoryPatch(Path(tmpdir)) as dirs:
                dirs.unprocessed.mkdir(parents=True)
                input_path = dirs.unprocessed / "bad.mp3"
                input_path.write_bytes(b"audio")

                manifest = self.make_runner().run_candidate(input_path)

                self.assertEqual(manifest["status"], "skipped_invalid_input")
                self.assertFalse(input_path.exists())
                self.assertTrue((dirs.failed / "bad.mp3").is_file())
                manifest_paths = list(dirs.runs.glob("*/manifest.json"))
                self.assertEqual(len(manifest_paths), 1)
                saved_manifest = json.loads(manifest_paths[0].read_text(encoding="utf-8"))
                self.assertEqual(saved_manifest["status"], "skipped_invalid_input")

    def test_smoke_success_runs_stages_writes_manifest_and_moves_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with QueueDirectoryPatch(Path(tmpdir)) as dirs:
                dirs.unprocessed.mkdir(parents=True)
                input_path = dirs.unprocessed / "20260414_qag.wav"
                input_path.write_bytes(b"audio")
                runner = self.make_runner()

                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    manifest = runner.run_candidate(input_path)

                self.assertEqual(manifest["status"], "success")
                self.assertEqual(
                    runner.called_stages,
                    orchestration_paths.EXECUTED_STAGE_NAMES,
                )
                self.assertTrue((dirs.processed / "20260414_qag.wav").is_file())
                self.assertFalse(input_path.exists())
                text = output.getvalue()
                self.assertIn("[1/10] vad.py START", text)
                self.assertIn("[10/10] build_sqlite.py OK", text)

                manifest_path = next(dirs.runs.glob("*/manifest.json"))
                saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(saved_manifest["status"], "success")
                self.assertEqual(
                    saved_manifest["stages"][1]["sub_stages"][0]["stage"],
                    "diarization_embeddings.py",
                )
                self.assertEqual(
                    saved_manifest["artifacts"]["sqlite_db"],
                    "sqlite/assemblybot.sqlite",
                )

    def test_smoke_failure_moves_audio_to_failed_and_records_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with QueueDirectoryPatch(Path(tmpdir)) as dirs:
                dirs.unprocessed.mkdir(parents=True)
                input_path = dirs.unprocessed / "20260414_qag.wav"
                input_path.write_bytes(b"audio")

                manifest = self.make_runner(fail_stage="audio_audit.py").run_candidate(
                    input_path
                )

                self.assertEqual(manifest["status"], "failed")
                self.assertIn("boom in audio_audit.py", manifest["error_message"])
                self.assertTrue((dirs.failed / "20260414_qag.wav").is_file())
                saved_manifest = json.loads(
                    next(dirs.runs.glob("*/manifest.json")).read_text(encoding="utf-8")
                )
                self.assertEqual(saved_manifest["status"], "failed")
                self.assertIn("traceback", saved_manifest)


if __name__ == "__main__":
    unittest.main()
