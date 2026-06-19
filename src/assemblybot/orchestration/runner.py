from __future__ import annotations

import contextlib
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Iterator

from assemblybot.alignment_config import DEFAULT_ALIGNMENT_CONFIG
from assemblybot.audio_audit_config import DEFAULT_AUDIO_AUDIT_CONFIG
from assemblybot.faster_whisper_config import (
    DEFAULT_FASTER_WHISPER_TRANSCRIPTION_CONFIG,
    FasterWhisperTranscriptionConfig,
)
from assemblybot.helper.document import load_document, save_document, save_turn_document
from assemblybot.models.factories import create_empty_document
from assemblybot.models.turn_document import TurnDocument
from assemblybot.orchestration import paths as orchestration_paths
from assemblybot.orchestration.manifest import (
    default_config_manifest,
    make_initial_manifest,
    write_json,
    write_manifest,
)
from assemblybot.orchestration.paths import (
    PipelinePaths,
    build_pipeline_paths,
    build_run_dir,
    ensure_run_directories,
    relative_to_run,
)
from assemblybot.orchestration.provenance import build_provenance, now_utc_iso
from assemblybot.orchestration.queue import is_wav_file, move_to_directory
from assemblybot.orchestration.resources import release_accelerator_memory
from assemblybot.per_config import DEFAULT_PER_CONFIG
from assemblybot.pyannote_config import DEFAULT_PYANNOTE_DIARIZATION_CONFIG
from assemblybot.semantic_chunk_config import DEFAULT_SEMANTIC_CHUNK_CONFIG
from assemblybot.silero_config import DEFAULT_SILERO_VAD_CONFIG
from assemblybot.stages.alignment import (
    build_transcript_diarization_matches,
    propagate_adjacent_transcript_anomaly_flags,
)
from assemblybot.stages.audio_audit import write_audio_audit
from assemblybot.stages.build_sqlite import build_sqlite_database
from assemblybot.stages.diarization import diarize_audio
from assemblybot.stages.per_extraction import (
    build_ner_pipeline,
    enrich_turn_document,
    load_all_known_people,
    load_turn_document,
)
from assemblybot.stages.semantic_chunk import (
    build_embedding_artifacts,
    load_sentence_transformer,
)
from assemblybot.stages.transcription import transcribe_audio
from assemblybot.stages.turns import consolidate_turns
from assemblybot.stages.vad import run_vad
from assemblybot.stages.whisper_segment_audit import audit_whisper_segments
from assemblybot.turns_config import DEFAULT_TURNS_CONFIG


class Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@contextlib.contextmanager
def tee_output(log_path: Path) -> Iterator[None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


class PipelineRunner:
    def __init__(
        self,
        *,
        language: str,
        deputies_ground_truth_csv: Path,
        ministers_ground_truth_csv: Path,
        replace_sqlite: bool,
        quiet: bool = False,
    ) -> None:
        self.language = language
        self.deputies_ground_truth_csv = deputies_ground_truth_csv
        self.ministers_ground_truth_csv = ministers_ground_truth_csv
        self.replace_sqlite = replace_sqlite
        self.quiet = quiet

    def announce(self, message: str) -> None:
        if not self.quiet:
            print(message, flush=True)

    def run_candidate(self, source_path: Path) -> dict[str, Any]:
        source_path = source_path.resolve()
        provenance = build_provenance(source_path)
        run_dir = build_run_dir(provenance)
        paths = build_pipeline_paths(run_dir, provenance.original_filename)
        ensure_run_directories(paths)
        config = default_config_manifest(
            language=self.language,
            deputies_ground_truth_csv=self.deputies_ground_truth_csv,
            ministers_ground_truth_csv=self.ministers_ground_truth_csv,
            paths=paths,
            replace_sqlite=self.replace_sqlite,
        )
        write_json(paths.config_json, config)
        manifest = make_initial_manifest(
            status="failed",
            provenance=provenance,
            paths=paths,
            config=config,
        )

        if not is_wav_file(source_path):
            self.announce(f"Skipping invalid input: {source_path.name}")
            failed_path = move_to_directory(source_path, orchestration_paths.FAILED_DIR)
            manifest["status"] = "skipped_invalid_input"
            manifest["audio_destination"] = str(failed_path)
            manifest["error_message"] = "Input file is not a .wav file."
            write_manifest(paths, manifest)
            return manifest

        processing_path: Path | None = None
        try:
            self.announce(f"Preparing run: {paths.run_dir}")
            processing_path = move_to_directory(
                source_path,
                orchestration_paths.PROCESSING_DIR,
            )
            shutil.copy2(processing_path, paths.input_audio)
            self.run_stages(paths, manifest)
            processed_path = move_to_directory(
                processing_path,
                orchestration_paths.PROCESSED_DIR,
            )
            manifest["status"] = "success"
            manifest["audio_destination"] = str(processed_path)
            write_manifest(paths, manifest)
            self.announce(f"Pipeline success: {paths.run_dir}")
            return manifest
        except Exception as exc:
            error_traceback = traceback.format_exc()
            manifest["status"] = "failed"
            manifest["error_message"] = str(exc)
            manifest["traceback"] = error_traceback
            if processing_path is not None and processing_path.exists():
                failed_path = move_to_directory(
                    processing_path,
                    orchestration_paths.FAILED_DIR,
                )
                manifest["audio_destination"] = str(failed_path)
            write_manifest(paths, manifest)
            self.announce(f"Pipeline failed: {exc}")
            print(error_traceback, flush=True)
            return manifest

    def run_stages(self, paths: PipelinePaths, manifest: dict[str, Any]) -> None:
        with tee_output(paths.log_file):
            document = create_empty_document(
                paths.input_audio,
                language_expected=self.language,
            )

            stage_calls: list[tuple[str, Callable[[], dict[str, Path | None]]]] = [
                (
                    "vad.py",
                    lambda: self.stage_vad(document, paths),
                ),
                (
                    "diarization.py",
                    lambda: self.stage_diarization(document, paths),
                ),
                (
                    "transcription.py",
                    lambda: self.stage_transcription(document, paths),
                ),
                (
                    "audio_audit.py",
                    lambda: self.stage_audio_audit(paths),
                ),
                (
                    "whisper_segment_audit.py",
                    lambda: self.stage_whisper_segment_audit(paths),
                ),
                (
                    "alignment.py",
                    lambda: self.stage_alignment(paths),
                ),
                (
                    "turns.py",
                    lambda: self.stage_turns(paths),
                ),
                (
                    "per_extraction.py",
                    lambda: self.stage_per_extraction(paths),
                ),
                (
                    "semantic_chunk.py",
                    lambda: self.stage_semantic_chunk(paths),
                ),
                (
                    "build_sqlite.py",
                    lambda: self.stage_build_sqlite(paths),
                ),
            ]

            total = len(stage_calls)
            for index, (stage_name, stage_func) in enumerate(stage_calls, start=1):
                self.run_stage(index, total, stage_name, stage_func, paths, manifest)

    def run_stage(
        self,
        index: int,
        total: int,
        stage_name: str,
        stage_func: Callable[[], dict[str, Path | None]],
        paths: PipelinePaths,
        manifest: dict[str, Any],
    ) -> None:
        started_at = now_utc_iso()
        start_time = time.time()
        self.announce(f"[{index}/{total}] {stage_name} START")
        record: dict[str, Any] = {
            "stage": stage_name,
            "status": "running",
            "started_at": started_at,
            "completed_at": None,
            "duration_seconds": None,
            "artifacts": {},
        }
        manifest["stages"].append(record)
        write_manifest(paths, manifest)

        try:
            artifacts = stage_func()
        except Exception as exc:
            duration = time.time() - start_time
            record["status"] = "failed"
            record["completed_at"] = now_utc_iso()
            record["duration_seconds"] = duration
            record["error_message"] = str(exc)
            write_manifest(paths, manifest)
            self.announce(f"[{index}/{total}] {stage_name} FAILED in {duration:.1f}s")
            raise

        duration = time.time() - start_time
        record["status"] = "success"
        record["completed_at"] = now_utc_iso()
        record["duration_seconds"] = duration
        record["artifacts"] = {
            name: relative_to_run(path, paths.run_dir)
            for name, path in artifacts.items()
            if path is not None
        }
        if stage_name == "diarization.py":
            record["sub_stages"] = [
                {
                    "stage": "diarization_embeddings.py",
                    "status": "success",
                    "artifacts": {
                        "segment_embeddings_npz": relative_to_run(
                            paths.diarization_segment_embeddings_npz,
                            paths.run_dir,
                        ),
                        "speaker_centroids_npz": relative_to_run(
                            paths.diarization_speaker_centroids_npz,
                            paths.run_dir,
                        ),
                    },
                }
            ]
        manifest["artifacts"].update(record["artifacts"])
        write_manifest(paths, manifest)
        self.announce(f"[{index}/{total}] {stage_name} OK in {duration:.1f}s")
        for label, path in record["artifacts"].items():
            self.announce(f"  {label}: {path}")
        self.release_memory_after_stage(stage_name)

    def release_memory_after_stage(self, stage_name: str) -> None:
        if stage_name not in {"diarization.py", "transcription.py"}:
            return
        actions = release_accelerator_memory()
        self.announce(
            "  released accelerator memory: "
            + ", ".join(actions)
        )

    def stage_vad(self, document: Any, paths: PipelinePaths) -> dict[str, Path | None]:
        run_vad(
            document=document,
            input_audio_path=paths.input_audio,
            output_json_path=paths.vad_json,
            config=DEFAULT_SILERO_VAD_CONFIG,
        )
        return {"canonical_json": paths.vad_json}

    def stage_diarization(
        self,
        document: Any,
        paths: PipelinePaths,
    ) -> dict[str, Path | None]:
        diarize_audio(
            document=document,
            input_audio_path=paths.input_audio,
            output_json_path=paths.diarization_json,
            output_segment_embeddings_path=paths.diarization_segment_embeddings_npz,
            output_speaker_centroids_path=paths.diarization_speaker_centroids_npz,
            config=DEFAULT_PYANNOTE_DIARIZATION_CONFIG,
        )
        return {
            "canonical_json": paths.diarization_json,
            "segment_embeddings_npz": paths.diarization_segment_embeddings_npz,
            "speaker_centroids_npz": paths.diarization_speaker_centroids_npz,
        }

    def stage_transcription(
        self,
        document: Any,
        paths: PipelinePaths,
    ) -> dict[str, Path | None]:
        config = FasterWhisperTranscriptionConfig(
            language=self.language,
            transcription_model_name=DEFAULT_FASTER_WHISPER_TRANSCRIPTION_CONFIG.transcription_model_name,
            device=DEFAULT_FASTER_WHISPER_TRANSCRIPTION_CONFIG.device,
            compute_type=DEFAULT_FASTER_WHISPER_TRANSCRIPTION_CONFIG.compute_type,
            beam_size=DEFAULT_FASTER_WHISPER_TRANSCRIPTION_CONFIG.beam_size,
            vad_filter=DEFAULT_FASTER_WHISPER_TRANSCRIPTION_CONFIG.vad_filter,
            vad_min_silence_duration_ms=(
                DEFAULT_FASTER_WHISPER_TRANSCRIPTION_CONFIG.vad_min_silence_duration_ms
            ),
            vad_speech_pad_ms=DEFAULT_FASTER_WHISPER_TRANSCRIPTION_CONFIG.vad_speech_pad_ms,
            temperature=DEFAULT_FASTER_WHISPER_TRANSCRIPTION_CONFIG.temperature,
            condition_on_previous_text=(
                DEFAULT_FASTER_WHISPER_TRANSCRIPTION_CONFIG.condition_on_previous_text
            ),
            word_timestamps=DEFAULT_FASTER_WHISPER_TRANSCRIPTION_CONFIG.word_timestamps,
        )
        transcribe_audio(
            document=document,
            input_audio_path=paths.input_audio,
            output_json_path=paths.transcription_json,
            output_txt_path=paths.transcript_txt,
            config=config,
        )
        return {
            "canonical_json": paths.transcription_json,
            "transcript_txt": paths.transcript_txt,
        }

    def stage_audio_audit(self, paths: PipelinePaths) -> dict[str, Path | None]:
        write_audio_audit(
            input_audio_path=paths.input_audio,
            output_path=paths.audio_audit_json,
            config=DEFAULT_AUDIO_AUDIT_CONFIG,
        )
        return {"audio_audit_json": paths.audio_audit_json}

    def stage_whisper_segment_audit(self, paths: PipelinePaths) -> dict[str, Path | None]:
        audit_whisper_segments(
            transcript_path=paths.transcription_json,
            aux_path=paths.diarization_json,
            audio_audit_path=paths.audio_audit_json,
            output_path=paths.flagged_transcription_json,
            write_sidecar=True,
            sidecar_output_path=paths.segment_audio_audit_json,
        )
        return {
            "canonical_json": paths.flagged_transcription_json,
            "segment_audio_audit_json": paths.segment_audio_audit_json,
        }

    def stage_alignment(self, paths: PipelinePaths) -> dict[str, Path | None]:
        document = load_document(paths.flagged_transcription_json)
        config = DEFAULT_ALIGNMENT_CONFIG
        propagate_adjacent_transcript_anomaly_flags(
            document.transcript.raw_segments,
            config,
        )
        document.alignment.transcript_diarization_matches = (
            build_transcript_diarization_matches(document, config)
        )
        save_document(document, paths.alignment_json)
        return {"canonical_json": paths.alignment_json}

    def stage_turns(self, paths: PipelinePaths) -> dict[str, Path | None]:
        document = load_document(paths.alignment_json)
        turns = consolidate_turns(document, DEFAULT_TURNS_CONFIG)
        save_turn_document(
            TurnDocument(
                turns=turns,
                turns_analysis=[],
            ),
            paths.turns_json,
        )
        return {"turns_json": paths.turns_json}

    def stage_per_extraction(self, paths: PipelinePaths) -> dict[str, Path | None]:
        document = load_turn_document(paths.turns_json)
        known_people = load_all_known_people(
            self.deputies_ground_truth_csv,
            self.ministers_ground_truth_csv,
            DEFAULT_PER_CONFIG,
        )
        ner = build_ner_pipeline(DEFAULT_PER_CONFIG)
        enriched_document = enrich_turn_document(
            document,
            known_people,
            ner,
            DEFAULT_PER_CONFIG,
        )
        save_turn_document(enriched_document, paths.per_extraction_json)
        return {"per_json": paths.per_extraction_json}

    def stage_semantic_chunk(self, paths: PipelinePaths) -> dict[str, Path | None]:
        model = load_sentence_transformer(DEFAULT_SEMANTIC_CHUNK_CONFIG)
        artifacts = build_embedding_artifacts(
            paths.per_extraction_json,
            model=model,
            output_dir=paths.rag_embedding_dir,
            config=DEFAULT_SEMANTIC_CHUNK_CONFIG,
        )
        return {
            "turn_embeddings_npz": artifacts.turn_embeddings_path,
            "semantic_chunks_npz": artifacts.semantic_chunks_path,
            "metadata_json": artifacts.metadata_path,
            "metadata_txt": artifacts.metadata_txt_path,
        }

    def stage_build_sqlite(self, paths: PipelinePaths) -> dict[str, Path | None]:
        build_sqlite_database(
            output_db_path=paths.sqlite_db,
            alignment_json_path=paths.alignment_json,
            per_json_path=paths.per_extraction_json,
            audio_path=paths.input_audio,
            turn_embeddings_npz_path=paths.turn_embeddings_npz,
            semantic_chunks_npz_path=paths.semantic_chunks_npz,
            embedding_metadata_json_path=paths.semantic_chunk_metadata_json,
            replace=self.replace_sqlite,
        )
        return {"sqlite_db": paths.sqlite_db}
