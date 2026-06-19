from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from assemblybot.alignment_config import DEFAULT_ALIGNMENT_CONFIG
from assemblybot.audio_audit_config import DEFAULT_AUDIO_AUDIT_CONFIG
from assemblybot.build_sqlite_config import BuildSqliteConfig
from assemblybot.faster_whisper_config import (
    DEFAULT_FASTER_WHISPER_TRANSCRIPTION_CONFIG,
    FasterWhisperTranscriptionConfig,
)
from assemblybot.orchestration.paths import EXECUTED_STAGE_NAMES, PipelinePaths
from assemblybot.orchestration.provenance import Provenance, now_utc_iso
from assemblybot.per_config import DEFAULT_PER_CONFIG
from assemblybot.pyannote_config import DEFAULT_PYANNOTE_DIARIZATION_CONFIG
from assemblybot.semantic_chunk_config import DEFAULT_SEMANTIC_CHUNK_CONFIG
from assemblybot.silero_config import DEFAULT_SILERO_VAD_CONFIG
from assemblybot.turns_config import DEFAULT_TURNS_CONFIG


def json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, frozenset):
        return sorted(json_safe(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type):
        return value.__name__
    if hasattr(value, "item"):
        return value.item()
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def default_config_manifest(
    *,
    language: str,
    deputies_ground_truth_csv: Path,
    ministers_ground_truth_csv: Path,
    paths: PipelinePaths,
    replace_sqlite: bool,
) -> dict[str, Any]:
    transcription_config = FasterWhisperTranscriptionConfig(
        language=language,
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
    build_sqlite_config = BuildSqliteConfig(
        output_db_path=paths.sqlite_db,
        turn_embeddings_npz_path=paths.turn_embeddings_npz,
        semantic_chunks_npz_path=paths.semantic_chunks_npz,
        embedding_metadata_json_path=paths.semantic_chunk_metadata_json,
        replace_existing_db=replace_sqlite,
    )
    return {
        "language": language,
        "deputies_ground_truth_csv": deputies_ground_truth_csv,
        "ministers_ground_truth_csv": ministers_ground_truth_csv,
        "vad": DEFAULT_SILERO_VAD_CONFIG,
        "diarization": DEFAULT_PYANNOTE_DIARIZATION_CONFIG,
        "transcription": transcription_config,
        "audio_audit": DEFAULT_AUDIO_AUDIT_CONFIG,
        "alignment": DEFAULT_ALIGNMENT_CONFIG,
        "turns": DEFAULT_TURNS_CONFIG,
        "per_extraction": DEFAULT_PER_CONFIG,
        "semantic_chunk": DEFAULT_SEMANTIC_CHUNK_CONFIG,
        "build_sqlite": build_sqlite_config,
    }


def make_initial_manifest(
    *,
    status: str,
    provenance: Provenance,
    paths: PipelinePaths,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "assemblybot.pipeline_manifest.v1",
        "status": status,
        "created_at": now_utc_iso(),
        "updated_at": now_utc_iso(),
        "input_provenance": asdict(provenance),
        "stage_order": list(EXECUTED_STAGE_NAMES),
        "config": json_safe(config),
        "artifacts": {},
        "stages": [],
        "audio_destination": None,
        "error_message": None,
        "traceback": None,
        "run_dir": str(paths.run_dir),
    }


def write_manifest(paths: PipelinePaths, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = now_utc_iso()
    write_json(paths.manifest_json, manifest)
