import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from assemblybot.helper.directory import build_default_output_path
from assemblybot.helper.document import load_document, save_document
from assemblybot.models.flags import SegmentFlag, flags_to_list
from assemblybot.models.transcript import TranscriptRawSegment


@dataclass(frozen=True)
class AuditThresholds:
    vad_partial_coverage: float = 0.80
    vad_long_segment_seconds: float = 10.0
    vad_long_segment_min_coverage: float = 0.60
    vad_internal_gap_seconds: float = 0.75
    low_avg_logprob: float = -1.0
    high_no_speech_prob: float = 0.60
    high_compression_ratio: float = 2.8
    short_segment_seconds: float = 0.40
    long_short_text_seconds: float = 8.0
    long_short_text_min_words: int = 4
    long_short_text_min_chars: int = 25
    silence_event_max_seconds: float = 3.0
    silence_event_median_db: float = -55.0
    silence_event_db_delta_p95: float = 15.0

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


AUDIT_THRESHOLDS = AuditThresholds()
AUDIO_AUDIT_FEATURES = (
    "frame_center_seconds",
    "rms",
    "db",
    "zcr",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_flatness",
    "db_rolling_median",
    "db_delta",
)


@dataclass
class SegmentAudioStats:
    vad_coverage: float
    diarization_overlap_seconds: float
    diarization_overlap_region_count: int
    frame_count: int
    db_mean: float | None
    db_p10: float | None
    db_p50: float | None
    db_p90: float | None
    db_delta_p95: float | None
    rms_mean: float | None
    zcr_mean: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    output = float(value)
    if not np.isfinite(output):
        return None
    return output


def segment_duration(segment: TranscriptRawSegment) -> float:
    return max(0.0, segment.time.end_seconds - segment.time.start_seconds)


def text_shape(raw_text: str) -> tuple[int, int]:
    stripped = raw_text.strip()
    return len("".join(stripped.split())), len(stripped.split())


def interval_overlap_seconds(
    start_seconds: float,
    end_seconds: float,
    intervals: list[tuple[float, float]],
) -> float:
    total = 0.0
    for interval_start, interval_end in intervals:
        overlap_start = max(start_seconds, interval_start)
        overlap_end = min(end_seconds, interval_end)
        if overlap_end > overlap_start:
            total += overlap_end - overlap_start
    return total


def clipped_overlaps(
    start_seconds: float,
    end_seconds: float,
    intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    overlaps: list[tuple[float, float]] = []
    for interval_start, interval_end in intervals:
        overlap_start = max(start_seconds, interval_start)
        overlap_end = min(end_seconds, interval_end)
        if overlap_end > overlap_start:
            overlaps.append((overlap_start, overlap_end))
    overlaps.sort()
    return overlaps


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start_seconds, end_seconds in intervals:
        if not merged or start_seconds > merged[-1][1]:
            merged.append((start_seconds, end_seconds))
        else:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end_seconds))
    return merged


def max_internal_gap_seconds(overlaps: list[tuple[float, float]]) -> float:
    if len(overlaps) < 2:
        return 0.0
    max_gap = 0.0
    for previous, current in zip(overlaps, overlaps[1:]):
        max_gap = max(max_gap, current[0] - previous[1])
    return max_gap


def vad_coverage_for_segment(
    segment: TranscriptRawSegment,
    vad_intervals: list[tuple[float, float]],
) -> float:
    duration = segment_duration(segment)
    if duration <= 0.0:
        return 0.0
    overlap = interval_overlap_seconds(
        segment.time.start_seconds,
        segment.time.end_seconds,
        vad_intervals,
    )
    return min(1.0, max(0.0, overlap / duration))


def check_diarization_overlap(
    segment: TranscriptRawSegment,
    overlap_intervals: list[tuple[float, float]],
) -> tuple[SegmentFlag, float, int]:
    overlaps = clipped_overlaps(
        segment.time.start_seconds,
        segment.time.end_seconds,
        overlap_intervals,
    )
    overlap_seconds = sum(
        end_seconds - start_seconds for start_seconds, end_seconds in overlaps
    )
    if overlap_seconds > 0.0:
        return SegmentFlag.MULTI_SPEAKER_CANDIDATE, overlap_seconds, len(overlaps)
    return SegmentFlag.NONE, 0.0, 0


def check_vad_alignment(
    segment: TranscriptRawSegment,
    vad_intervals: list[tuple[float, float]],
    thresholds: AuditThresholds,
) -> tuple[SegmentFlag, float]:
    flags = SegmentFlag.NONE
    duration = segment_duration(segment)
    coverage = vad_coverage_for_segment(segment, vad_intervals)

    if coverage == 0.0:
        flags |= SegmentFlag.OUTSIDE_VAD
    elif coverage < thresholds.vad_partial_coverage:
        flags |= SegmentFlag.PARTIAL_VAD_OVERLAP

    if (
        duration >= thresholds.vad_long_segment_seconds
        and coverage < thresholds.vad_long_segment_min_coverage
    ):
        flags |= SegmentFlag.LONG_WHISPER_SEGMENT_LOW_VAD_COVERAGE

    overlaps = merge_intervals(
        clipped_overlaps(
            segment.time.start_seconds,
            segment.time.end_seconds,
            vad_intervals,
        )
    )
    if (
        coverage > 0.0
        and max_internal_gap_seconds(overlaps) >= thresholds.vad_internal_gap_seconds
    ):
        flags |= SegmentFlag.INSIDE_VAD_GAP

    return flags, coverage


def check_whisper_quality(
    segment: TranscriptRawSegment,
    thresholds: AuditThresholds,
) -> SegmentFlag:
    flags = SegmentFlag.NONE
    if (
        segment.avg_logprob is not None
        and segment.avg_logprob <= thresholds.low_avg_logprob
    ):
        flags |= SegmentFlag.LOW_WHISPER_CONFIDENCE
    if (
        segment.no_speech_prob is not None
        and segment.no_speech_prob >= thresholds.high_no_speech_prob
    ):
        flags |= SegmentFlag.HIGH_NO_SPEECH_PROB
    if (
        segment.compression_ratio is not None
        and segment.compression_ratio >= thresholds.high_compression_ratio
    ):
        flags |= SegmentFlag.HIGH_COMPRESSION_RATIO
    return flags


def check_duration_text_shape(
    segment: TranscriptRawSegment,
    thresholds: AuditThresholds,
) -> SegmentFlag:
    flags = SegmentFlag.NONE
    duration = segment_duration(segment)
    char_count, word_count = text_shape(segment.raw_text)

    if duration < thresholds.short_segment_seconds and char_count > 0:
        flags |= SegmentFlag.SHORT_SEGMENT
    if duration >= thresholds.long_short_text_seconds and (
        word_count < thresholds.long_short_text_min_words
        or char_count < thresholds.long_short_text_min_chars
    ):
        flags |= SegmentFlag.LONG_DURATION_SHORT_TEXT
    return flags


def load_audio_audit_arrays(audio_audit_path: Path) -> dict[str, np.ndarray]:
    print(f"Loading audio audit frames: {audio_audit_path}", flush=True)
    with audio_audit_path.open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)

    frames = data.get("frames", [])
    if not frames:
        raise ValueError(f"Audio audit contains no frames: {audio_audit_path}")

    arrays: dict[str, np.ndarray] = {}
    for feature_name in AUDIO_AUDIT_FEATURES:
        arrays[feature_name] = np.asarray(
            [frame[feature_name] for frame in frames],
            dtype=np.float32,
        )
    print(f"Loaded {len(frames)} audio audit frames.", flush=True)
    return arrays


def audio_stats_for_segment(
    segment: TranscriptRawSegment,
    audit_arrays: dict[str, np.ndarray],
    vad_coverage: float,
    diarization_overlap_seconds: float,
    diarization_overlap_region_count: int,
) -> SegmentAudioStats:
    frame_centers = audit_arrays["frame_center_seconds"]
    start_index = int(
        np.searchsorted(frame_centers, segment.time.start_seconds, side="left")
    )
    end_index = int(
        np.searchsorted(frame_centers, segment.time.end_seconds, side="right")
    )
    frame_count = max(0, end_index - start_index)
    if frame_count == 0:
        return SegmentAudioStats(
            vad_coverage=vad_coverage,
            diarization_overlap_seconds=diarization_overlap_seconds,
            diarization_overlap_region_count=diarization_overlap_region_count,
            frame_count=0,
            db_mean=None,
            db_p10=None,
            db_p50=None,
            db_p90=None,
            db_delta_p95=None,
            rms_mean=None,
            zcr_mean=None,
        )

    segment_db = audit_arrays["db"][start_index:end_index]
    segment_db_delta = audit_arrays["db_delta"][start_index:end_index]
    return SegmentAudioStats(
        vad_coverage=vad_coverage,
        diarization_overlap_seconds=diarization_overlap_seconds,
        diarization_overlap_region_count=diarization_overlap_region_count,
        frame_count=frame_count,
        db_mean=finite_float_or_none(np.mean(segment_db)),
        db_p10=finite_float_or_none(np.percentile(segment_db, 10)),
        db_p50=finite_float_or_none(np.percentile(segment_db, 50)),
        db_p90=finite_float_or_none(np.percentile(segment_db, 90)),
        db_delta_p95=finite_float_or_none(np.percentile(segment_db_delta, 95)),
        rms_mean=finite_float_or_none(
            np.mean(audit_arrays["rms"][start_index:end_index])
        ),
        zcr_mean=finite_float_or_none(
            np.mean(audit_arrays["zcr"][start_index:end_index])
        ),
    )


def check_audio_audit(
    segment: TranscriptRawSegment,
    stats: SegmentAudioStats,
    thresholds: AuditThresholds,
) -> SegmentFlag:
    flags = SegmentFlag.NONE
    if stats.frame_count == 0 or stats.db_p50 is None or stats.db_delta_p95 is None:
        return flags

    if (
        segment_duration(segment) <= thresholds.silence_event_max_seconds
        and stats.db_p50 <= thresholds.silence_event_median_db
        and stats.db_delta_p95 >= thresholds.silence_event_db_delta_p95
    ):
        flags |= SegmentFlag.MOSTLY_SILENCE_WITH_SHORT_EVENT
    return flags


def sidecar_record(
    segment: TranscriptRawSegment,
    stats: SegmentAudioStats,
) -> dict[str, Any]:
    char_count, word_count = text_shape(segment.raw_text)
    flags = int(segment.flags)
    return {
        "segment_id": segment.segment_id,
        "start_seconds": segment.time.start_seconds,
        "end_seconds": segment.time.end_seconds,
        "duration_seconds": segment.time.duration_seconds,
        "text_char_count": char_count,
        "text_word_count": word_count,
        "vad_coverage": stats.vad_coverage,
        "diarization_overlap_seconds": stats.diarization_overlap_seconds,
        "diarization_overlap_region_count": stats.diarization_overlap_region_count,
        "frame_count": stats.frame_count,
        "db_mean": stats.db_mean,
        "db_p10": stats.db_p10,
        "db_p50": stats.db_p50,
        "db_p90": stats.db_p90,
        "db_delta_p95": stats.db_delta_p95,
        "rms_mean": stats.rms_mean,
        "zcr_mean": stats.zcr_mean,
        "avg_logprob": segment.avg_logprob,
        "no_speech_prob": segment.no_speech_prob,
        "compression_ratio": segment.compression_ratio,
        "flags": flags,
        "flag_names": flags_to_list(flags),
    }


def write_segment_audio_audit(
    output_path: Path,
    transcript_path: Path,
    aux_path: Path,
    audio_audit_path: Path,
    thresholds: AuditThresholds,
    records: list[dict[str, Any]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": "whisper_segment_audio_audit.v1",
        "source": {
            "transcript_path": str(transcript_path),
            "vad_diarization_path": str(aux_path),
            "audio_audit_path": str(audio_audit_path),
        },
        "parameters": thresholds.to_dict(),
        "segments": records,
    }
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(document, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def aux_intervals_from_document(
    aux_document_path: Path,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    aux_document = load_document(aux_document_path)
    vad_intervals = [
        (segment.time.start_seconds, segment.time.end_seconds)
        for segment in aux_document.vad.segments
    ]
    overlap_intervals = [
        (region.time.start_seconds, region.time.end_seconds)
        for region in aux_document.diarization.overlap_regions
    ]
    vad_intervals.sort()
    overlap_intervals.sort()
    print(
        f"Loaded {len(vad_intervals)} VAD intervals and "
        f"{len(overlap_intervals)} diarization overlap regions.",
        flush=True,
    )
    return vad_intervals, overlap_intervals


def audit_whisper_segments(
    transcript_path: Path,
    aux_path: Path,
    audio_audit_path: Path,
    output_path: Path,
    write_sidecar: bool,
    sidecar_output_path: Path,
    thresholds: AuditThresholds = AUDIT_THRESHOLDS,
) -> int:
    print(f"Loading transcript document: {transcript_path}", flush=True)
    transcript_document = load_document(transcript_path)
    print(f"Loading VAD/diarization document: {aux_path}", flush=True)
    vad_intervals, overlap_intervals = aux_intervals_from_document(aux_path)
    audit_arrays = load_audio_audit_arrays(audio_audit_path)

    segment_count = len(transcript_document.transcript.raw_segments)
    print(f"Auditing {segment_count} Whisper segments...", flush=True)
    sidecar_records: list[dict[str, Any]] = []

    for index, segment in enumerate(transcript_document.transcript.raw_segments, 1):
        flags = SegmentFlag(segment.flags)
        vad_flags, vad_coverage = check_vad_alignment(
            segment, vad_intervals, thresholds
        )
        overlap_flags, overlap_seconds, overlap_region_count = check_diarization_overlap(
            segment,
            overlap_intervals,
        )
        stats = audio_stats_for_segment(
            segment,
            audit_arrays,
            vad_coverage,
            overlap_seconds,
            overlap_region_count,
        )
        flags |= vad_flags
        flags |= overlap_flags
        flags |= check_whisper_quality(segment, thresholds)
        flags |= check_duration_text_shape(segment, thresholds)
        flags |= check_audio_audit(segment, stats, thresholds)
        segment.flags = flags

        if write_sidecar:
            sidecar_records.append(sidecar_record(segment, stats))

        if index == 1 or index == segment_count or index % 1000 == 0:
            print(f"  audited segment {index}/{segment_count}", flush=True)

    print(f"Writing enriched transcript document: {output_path}", flush=True)
    save_document(transcript_document, output_path)

    if write_sidecar:
        print(f"Writing segment audio audit sidecar: {sidecar_output_path}", flush=True)
        write_segment_audio_audit(
            sidecar_output_path,
            transcript_path,
            aux_path,
            audio_audit_path,
            thresholds,
            sidecar_records,
        )

    return segment_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Whisper segments against VAD and audio audit features."
    )
    parser.add_argument(
        "--input-json",
        required=True,
        help="Path to transcript/canonical document JSON.",
    )
    parser.add_argument(
        "--input-vad-diarization-json",
        required=True,
        help="Path to canonical document containing VAD and optionally diarization.",
    )
    parser.add_argument(
        "--input-audit",
        required=True,
        help="Path to audio_audit.json.",
    )
    parser.add_argument(
        "--output",
        help="Optional enriched transcript JSON path.",
    )
    parser.add_argument(
        "--write-segment-audio-audit",
        action="store_true",
        help="Write compact per-segment audio audit sidecar JSON.",
    )
    parser.add_argument(
        "--segment-audio-audit-output",
        help="Optional sidecar output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transcript_path = Path(args.input_json).expanduser().resolve()
    aux_path = Path(args.input_vad_diarization_json).expanduser().resolve()
    audio_audit_path = Path(args.input_audit).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else build_default_output_path(
            transcript_path,
            "_whisper_segment_audit",
            "json",
            transcript_path.parent,
        )
    )
    sidecar_output_path = (
        Path(args.segment_audio_audit_output).expanduser().resolve()
        if args.segment_audio_audit_output
        else audio_audit_path.parent / "whisper_segment_audio_audit.json"
    )

    segment_count = audit_whisper_segments(
        transcript_path=transcript_path,
        aux_path=aux_path,
        audio_audit_path=audio_audit_path,
        output_path=output_path,
        write_sidecar=args.write_segment_audio_audit,
        sidecar_output_path=sidecar_output_path,
    )
    print(f"Audited segment count: {segment_count}", flush=True)
    print(f"Saved enriched transcript JSON: {output_path}", flush=True)
    if args.write_segment_audio_audit:
        print(f"Saved segment audio audit JSON: {sidecar_output_path}", flush=True)


if __name__ == "__main__":
    main()
