from pathlib import Path

from .document import CanonicalDocument, PipelineInfo, SourceInfo
from .time import now_utc_iso

VALID_STAGE_NAMES = {"transcription", "diarization", "merge", "enrichment", "export"}


def create_empty_document(
    input_path: str | Path,
    language_expected: str = "fr",
) -> CanonicalDocument:
    input_path = Path(input_path).resolve()
    now = now_utc_iso()

    return CanonicalDocument(
        schema_version="0.1.0",
        source=SourceInfo(
            source_id=input_path.stem,
            input_path=str(input_path),
            input_filename=input_path.name,
            media_type="audio",
            language_expected=language_expected,
            duration_seconds=None,
            file_sha256=None,
        ),
        pipeline=PipelineInfo(
            created_at=now,
            updated_at=now,
        ),
    )


def mark_stage_completed(
    document: CanonicalDocument,
    stage_name: str,
    output_path: str | None = None,
) -> None:
    if stage_name not in VALID_STAGE_NAMES:
        raise ValueError(f"Unknown stage name: {stage_name}")

    artifact = getattr(document.pipeline.stage_outputs, stage_name)
    now = now_utc_iso()

    artifact.status = "completed"
    artifact.completed_at = now
    artifact.error_message = None

    if output_path is not None:
        artifact.output_path = output_path

    document.pipeline.updated_at = now


def mark_stage_running(
    document: CanonicalDocument,
    stage_name: str,
) -> None:
    if stage_name not in VALID_STAGE_NAMES:
        raise ValueError(f"Unknown stage name: {stage_name}")

    artifact = getattr(document.pipeline.stage_outputs, stage_name)
    now = now_utc_iso()

    artifact.status = "running"
    artifact.error_message = None

    document.pipeline.updated_at = now


def mark_stage_failed(
    document: CanonicalDocument,
    stage_name: str,
    error_message: str,
) -> None:
    if stage_name not in VALID_STAGE_NAMES:
        raise ValueError(f"Unknown stage name: {stage_name}")

    artifact = getattr(document.pipeline.stage_outputs, stage_name)
    now = now_utc_iso()

    artifact.status = "failed"
    artifact.error_message = error_message

    document.pipeline.updated_at = now
