from pathlib import Path

from .document import CanonicalDocument, PipelineInfo, SourceInfo
from .time import now_utc_iso


def create_empty_document(
    input_path: str | Path,
    language_expected: str = "fr",
) -> CanonicalDocument:
    input_path = Path(input_path).resolve()
    now = now_utc_iso()

    return CanonicalDocument(
        schema_version="0.1.0",
        source=SourceInfo(
            input_path=str(input_path),
            input_filename=input_path.name,
            media_type="audio",
            language_expected=language_expected,
            duration_seconds=None,
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
    if stage_name not in document.pipeline.stages_completed:
        document.pipeline.stages_completed.append(stage_name)

    document.pipeline.updated_at = now_utc_iso()

    if output_path is not None and hasattr(document.pipeline.stage_outputs, stage_name):
        setattr(document.pipeline.stage_outputs, stage_name, output_path)
