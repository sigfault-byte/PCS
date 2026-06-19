from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from assemblybot.config import DATA_DIR, PROJECT_ROOT
from assemblybot.orchestration.provenance import Provenance


AUDIO_DIR = DATA_DIR / "audio"
UNPROCESSED_DIR = AUDIO_DIR / "unprocessed"
PROCESSING_DIR = AUDIO_DIR / "processing"
PROCESSED_DIR = AUDIO_DIR / "processed"
FAILED_DIR = AUDIO_DIR / "failed"
RUNS_DIR = DATA_DIR / "runs"

DEFAULT_DEPUTIES_CSV = PROJECT_ROOT / "docs" / "liste_deputes_libre_office_2026-06.csv"
DEFAULT_MINISTERS_CSV = PROJECT_ROOT / "docs" / "liste_ministre_2026.csv"

EXECUTED_STAGE_NAMES = [
    "vad.py",
    "diarization.py",
    "transcription.py",
    "audio_audit.py",
    "whisper_segment_audit.py",
    "alignment.py",
    "turns.py",
    "per_extraction.py",
    "semantic_chunk.py",
    "build_sqlite.py",
]


@dataclass(frozen=True)
class PipelinePaths:
    run_dir: Path
    input_audio: Path
    config_json: Path
    vad_json: Path
    diarization_json: Path
    diarization_segment_embeddings_npz: Path
    diarization_speaker_centroids_npz: Path
    transcription_json: Path
    transcript_txt: Path
    audio_audit_json: Path
    flagged_transcription_json: Path
    segment_audio_audit_json: Path
    alignment_json: Path
    turns_json: Path
    per_extraction_json: Path
    rag_embedding_dir: Path
    turn_embeddings_npz: Path
    semantic_chunks_npz: Path
    semantic_chunk_metadata_json: Path
    semantic_chunk_metadata_txt: Path
    sqlite_db: Path
    log_file: Path
    manifest_json: Path


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 10000):
        candidate = path.with_name(f"{path.name}-{index}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find a unique path for {path}")


def build_run_dir(provenance: Provenance, runs_dir: Path | None = None) -> Path:
    runs_dir = runs_dir or RUNS_DIR
    ingestion_date = provenance.ingestion_timestamp[:10]
    detected_date = provenance.detected_date or "no-date"
    name = (
        f"{ingestion_date}_"
        f"{detected_date}_"
        f"{provenance.detected_title_session_slug}_"
        f"{provenance.sha8}"
    )
    return unique_path(runs_dir / name)


def build_pipeline_paths(run_dir: Path, original_filename: str) -> PipelinePaths:
    stem = Path(original_filename).stem
    return PipelinePaths(
        run_dir=run_dir,
        input_audio=run_dir / "input" / original_filename,
        config_json=run_dir / "config" / "pipeline_config.json",
        vad_json=run_dir / "interim" / f"{stem}_0_vad.json",
        diarization_json=run_dir / "interim" / f"{stem}_01_diarization.json",
        diarization_segment_embeddings_npz=(
            run_dir / "embeddings" / "pyannote" / f"{stem}_01_segment_embeddings.npz"
        ),
        diarization_speaker_centroids_npz=(
            run_dir / "embeddings" / "pyannote" / f"{stem}_01_speaker_centroids.npz"
        ),
        transcription_json=run_dir / "interim" / f"{stem}_02_transcription.json",
        transcript_txt=run_dir / "interim" / f"{stem}_02_transcript.txt",
        audio_audit_json=run_dir / "audit" / "meta_json" / f"{stem}_audio_audit.json",
        flagged_transcription_json=(
            run_dir / "audit" / "turn_json" / f"{stem}_02_transcription_flagged.json"
        ),
        segment_audio_audit_json=(
            run_dir / "audit" / "turn_json" / "whisper_segment_audio_audit.json"
        ),
        alignment_json=run_dir / "interim" / f"{stem}_03_alignment.json",
        turns_json=run_dir / "audit" / "turn_json" / f"{stem}_01_turns.json",
        per_extraction_json=(
            run_dir / "audit" / "turn_json" / f"{stem}_02_per_extraction.json"
        ),
        rag_embedding_dir=run_dir / "embeddings" / "rag",
        turn_embeddings_npz=run_dir / "embeddings" / "rag" / "turn_embeddings.npz",
        semantic_chunks_npz=run_dir / "embeddings" / "rag" / "semantic_chunks.npz",
        semantic_chunk_metadata_json=(
            run_dir / "embeddings" / "rag" / "semantic_chunk_metadata.json"
        ),
        semantic_chunk_metadata_txt=(
            run_dir / "embeddings" / "rag" / "semantic_chunk_metadata.txt"
        ),
        sqlite_db=run_dir / "sqlite" / "assemblybot.sqlite",
        log_file=run_dir / "logs" / "pipeline.log",
        manifest_json=run_dir / "manifest.json",
    )


def ensure_run_directories(paths: PipelinePaths) -> None:
    for path in asdict(paths).values():
        if isinstance(path, Path):
            path.parent.mkdir(parents=True, exist_ok=True)


def relative_to_run(path: Path | None, run_dir: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(run_dir.resolve()))
    except ValueError:
        return str(path)
