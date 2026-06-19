from __future__ import annotations

from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from assemblybot.db.schema.diarization import DiarizationSegmentRecord
from assemblybot.db.schema.pipeline_run import PipelineRunRecord
from assemblybot.db.schema.session import SessionRecord
from assemblybot.db.schema.speaker import SpeakerClusterRecord
from assemblybot.db.schema.transcript import TranscriptSegmentRecord
from assemblybot.helper.document import load_document
from assemblybot.models.document import CanonicalDocument


class DuplicateSegmentLoadError(ValueError):
    """Raised when segment rows already exist for a pipeline run."""


class MissingPipelineRunError(ValueError):
    """Raised when a required pipeline run row is missing."""


class MissingSpeakerClusterError(ValueError):
    """Raised when a diarization segment references an unloaded speaker cluster."""


def get_pipeline_run_id(
    db_session: Session,
    session_id: int,
    stage: str,
) -> int:
    run_id = db_session.scalar(
        select(PipelineRunRecord.id).where(
            PipelineRunRecord.session_id == session_id,
            PipelineRunRecord.stage == stage,
        )
    )
    if run_id is None:
        raise MissingPipelineRunError(
            f"Missing pipeline_run for session_id={session_id}, stage={stage!r}"
        )

    return run_id


def build_transcript_segment_records(
    document: CanonicalDocument,
    pipeline_run_id: int,
) -> list[TranscriptSegmentRecord]:
    return [
        TranscriptSegmentRecord(
            id=segment.segment_id,
            pipeline_run_id=pipeline_run_id,
            text=segment.raw_text,
            start_seconds=segment.time.start_seconds,
            end_seconds=segment.time.end_seconds,
            flags=int(segment.flags),
            avg_log_prob=segment.avg_logprob,
            no_speech_prob=segment.no_speech_prob,
            compression_ratio=segment.compression_ratio,
        )
        for segment in document.transcript.raw_segments
    ]


def build_speaker_cluster_lookup(
    db_session: Session,
    session_id: int,
) -> dict[str, int]:
    rows = db_session.execute(
        select(SpeakerClusterRecord.label, SpeakerClusterRecord.id).where(
            SpeakerClusterRecord.session_id == session_id
        )
    ).all()

    return {label: speaker_cluster_id for label, speaker_cluster_id in rows}


def build_diarization_segment_records(
    document: CanonicalDocument,
    pipeline_run_id: int,
    speaker_clusters_by_label: dict[str, int],
) -> list[DiarizationSegmentRecord]:
    records: list[DiarizationSegmentRecord] = []

    for segment in document.diarization.raw_segments:
        speaker_cluster_id = speaker_clusters_by_label.get(segment.speaker_id)
        if speaker_cluster_id is None:
            raise MissingSpeakerClusterError(
                f"Missing speaker_cluster for speaker label {segment.speaker_id!r}"
            )

        records.append(
            DiarizationSegmentRecord(
                id=segment.segment_id,
                pipeline_run_id=pipeline_run_id,
                speaker_cluster_id=speaker_cluster_id,
                start_seconds=segment.time.start_seconds,
                end_seconds=segment.time.end_seconds,
                flags=int(segment.flags),
                overlap_speaker_ids=list(segment.overlap_speaker_ids),
            )
        )

    return records


def load_segment_records(
    db_session: Session,
    alignment_json_path: str | Path,
    session_id: int,
) -> tuple[list[TranscriptSegmentRecord], list[DiarizationSegmentRecord]]:
    parent = db_session.get(SessionRecord, session_id)
    if parent is None:
        raise ValueError(f"Session does not exist for session_id={session_id}")

    transcription_run_id = get_pipeline_run_id(
        db_session,
        session_id,
        "transcription",
    )
    diarization_run_id = get_pipeline_run_id(
        db_session,
        session_id,
        "diarization",
    )

    existing_transcript_segment_id = db_session.scalar(
        select(TranscriptSegmentRecord.id).where(
            TranscriptSegmentRecord.pipeline_run_id == transcription_run_id
        )
    )
    if existing_transcript_segment_id is not None:
        raise DuplicateSegmentLoadError(
            "Transcript segments already exist for "
            f"pipeline_run_id={transcription_run_id}"
        )

    existing_diarization_segment_id = db_session.scalar(
        select(DiarizationSegmentRecord.id).where(
            DiarizationSegmentRecord.pipeline_run_id == diarization_run_id
        )
    )
    if existing_diarization_segment_id is not None:
        raise DuplicateSegmentLoadError(
            "Diarization segments already exist for "
            f"pipeline_run_id={diarization_run_id}"
        )

    document = load_document(Path(alignment_json_path))
    speaker_clusters_by_label = build_speaker_cluster_lookup(db_session, session_id)
    transcript_records = build_transcript_segment_records(
        document,
        transcription_run_id,
    )
    diarization_records = build_diarization_segment_records(
        document,
        diarization_run_id,
        speaker_clusters_by_label,
    )

    db_session.add_all(transcript_records)
    db_session.add_all(diarization_records)
    db_session.flush()
    return transcript_records, diarization_records
