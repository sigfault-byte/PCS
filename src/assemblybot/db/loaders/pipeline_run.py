from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from assemblybot.db.schema.pipeline_run import PipelineRunRecord
from assemblybot.db.schema.session import SessionRecord
from assemblybot.helper.document import load_document
from assemblybot.models.document import CanonicalDocument


PIPELINE_STAGE_NAMES = ("vad", "transcription", "diarization")


class DuplicatePipelineRunError(ValueError):
    """Raised when a pipeline run already exists for a session and stage."""


def engine_config_to_dict(engine: object) -> dict[str, Any]:
    """Serialize a stage engine dataclass into JSON-compatible config data."""
    if is_dataclass(engine):
        return asdict(engine)

    if isinstance(engine, dict):
        return dict(engine)

    raise TypeError(f"Unsupported engine config type: {type(engine).__name__}")


def build_pipeline_run_records(
    document: CanonicalDocument,
    session_id: int,
) -> list[PipelineRunRecord]:
    """Build pipeline run ORM rows for stages present in the canonical document."""
    stage_engines = {
        "vad": document.vad.engine,
        "transcription": document.transcript.engine,
        "diarization": document.diarization.engine,
    }
    records: list[PipelineRunRecord] = []

    for stage in PIPELINE_STAGE_NAMES:
        engine = stage_engines[stage]
        config_json = engine_config_to_dict(engine)
        engine_name = config_json.get("name")
        if not engine_name:
            raise ValueError(f"Missing engine name for stage: {stage}")

        records.append(
            PipelineRunRecord(
                schema_ver=document.schema_version,
                session_id=session_id,
                stage=stage,
                engine_name=engine_name,
                model=config_json.get("model"),
                device=config_json.get("device"),
                config_json=config_json,
            )
        )

    return records


def load_pipeline_run_records(
    db_session: Session,
    json_path: str | Path,
    session_id: int,
) -> list[PipelineRunRecord]:
    """Load canonical stage engine metadata into the `pipeline_run` table."""
    parent = db_session.get(SessionRecord, session_id)
    if parent is None:
        raise ValueError(f"Session does not exist for session_id={session_id}")

    document = load_document(Path(json_path))
    records = build_pipeline_run_records(document, session_id)
    stages = [record.stage for record in records]

    existing_stages = db_session.scalars(
        select(PipelineRunRecord.stage).where(
            PipelineRunRecord.session_id == session_id,
            PipelineRunRecord.stage.in_(stages),
        )
    ).all()
    if existing_stages:
        stage_list = ", ".join(sorted(existing_stages))
        raise DuplicatePipelineRunError(
            f"Pipeline run already exists for session_id={session_id}: {stage_list}"
        )

    db_session.add_all(records)
    db_session.flush()
    return records
