from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------
# Time helpers
# ------------------------------------------------------------


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def overlap_duration(
    start_a: float | None,
    end_a: float | None,
    start_b: float | None,
    end_b: float | None,
) -> float:
    if None in (start_a, end_a, start_b, end_b):
        return 0.0
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def compute_gap(prev_end: float | None, next_start: float | None) -> float | None:
    if prev_end is None or next_start is None:
        return None
    return next_start - prev_end


def is_near_multiple(value: float | None, base: float, tolerance: float) -> bool:
    if value is None:
        return False
    nearest = round(value / base) * base
    return abs(value - nearest) <= tolerance


# ------------------------------------------------------------
# Records
# ------------------------------------------------------------


@dataclass(slots=True)
class DiarSegment:
    diar_segment_id: str
    speaker: str
    start_sec: float | None
    end_sec: float | None

    @property
    def duration(self) -> float | None:
        if self.start_sec is None or self.end_sec is None:
            return None
        return self.end_sec - self.start_sec


# ------------------------------------------------------------
# Extraction
# ------------------------------------------------------------


def extract_transcript_segments(doc: dict[str, Any]) -> list[dict[str, Any]]:
    transcript = doc.get("transcript", {})

    for key in ("raw_segments", "segments"):
        value = transcript.get(key)
        if isinstance(value, list):
            return value

    if isinstance(doc.get("raw_segments"), list):
        return doc["raw_segments"]

    raise KeyError("Could not find transcript segments in document")


def extract_tokens(doc: dict[str, Any]) -> list[dict[str, Any]]:
    transcript = doc.get("transcript", {})

    for key in ("tokens", "raw_tokens", "word_timestamps", "words"):
        value = transcript.get(key)
        if isinstance(value, list):
            return value

    if isinstance(doc.get("tokens"), list):
        return doc["tokens"]

    raise KeyError("Could not find transcript tokens in document")


def extract_diarization_segments(doc: dict[str, Any]) -> list[DiarSegment]:
    diarization = doc.get("diarization", {})

    raw_segments = None
    for key in ("collapsed_segments", "segments", "raw_segments"):
        value = diarization.get(key)
        if isinstance(value, list):
            raw_segments = value
            break

    if raw_segments is None:
        raise KeyError("Could not find diarization segments in document")

    diar_segments: list[DiarSegment] = []
    for i, seg in enumerate(raw_segments):
        diar_segments.append(
            DiarSegment(
                diar_segment_id=str(seg.get("segment_id", f"diar_{i:06d}")),
                speaker=str(seg.get("speaker", seg.get("speaker_id", "UNKNOWN"))),
                start_sec=safe_float(
                    seg.get("start_sec", seg.get("start_seconds", seg.get("start")))
                ),
                end_sec=safe_float(
                    seg.get("end_sec", seg.get("end_seconds", seg.get("end")))
                ),
            )
        )

    return diar_segments


# ------------------------------------------------------------
# Token dataframe
# ------------------------------------------------------------


def build_token_df(tokens: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for idx, tok in enumerate(tokens):
        token_id = tok.get("token_id", idx)
        start_sec = safe_float(
            tok.get("start_sec", tok.get("start_seconds", tok.get("start")))
        )
        end_sec = safe_float(tok.get("end_sec", tok.get("end_seconds", tok.get("end"))))
        text = tok.get("text", tok.get("token", tok.get("raw_token", "")))
        segment_id = tok.get("segment_id")

        rows.append(
            {
                "token_row_index": idx,
                "token_id": token_id,
                "segment_id": segment_id,
                "text": text,
                "text_stripped": str(text).strip(),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": (
                    end_sec - start_sec
                    if start_sec is not None and end_sec is not None
                    else None
                ),
                "is_empty": len(str(text).strip()) == 0,
                "is_punctuation_like": (
                    len(str(text).strip()) > 0
                    and all(ch in ",.;:!?()[]{}\"'’-—-" for ch in str(text).strip())
                ),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values(
        ["start_sec", "end_sec", "token_row_index"], na_position="last"
    ).reset_index(drop=True)

    df["prev_text"] = df["text_stripped"].shift(1)
    df["next_text"] = df["text_stripped"].shift(-1)
    df["prev_end_sec"] = df["end_sec"].shift(1)
    df["next_start_sec"] = df["start_sec"].shift(-1)
    df["gap_from_prev_sec"] = [
        compute_gap(a, b)
        for a, b in zip(df["prev_end_sec"], df["start_sec"], strict=False)
    ]
    df["gap_to_next_sec"] = [
        compute_gap(a, b)
        for a, b in zip(df["end_sec"], df["next_start_sec"], strict=False)
    ]
    df["is_repeat_with_prev"] = (df["text_stripped"] != "") & (
        df["text_stripped"] == df["prev_text"]
    )
    df["near_30s_boundary"] = df["start_sec"].apply(
        lambda x: is_near_multiple(x, 30.0, 0.15)
    )

    return df


# ------------------------------------------------------------
# Diar overlap for tokens
# ------------------------------------------------------------


def annotate_token_overlap(
    token_df: pd.DataFrame,
    diar_segments: list[DiarSegment],
) -> pd.DataFrame:
    if token_df.empty:
        return token_df

    overlap_rows: list[dict[str, Any]] = []

    for token in token_df.itertuples(index=False):
        best_overlap = 0.0
        best_speaker = None
        best_diar_segment_id = None
        total_overlap = 0.0
        speaker_overlaps: dict[str, float] = {}

        for diar in diar_segments:
            ov = overlap_duration(
                token.start_sec, token.end_sec, diar.start_sec, diar.end_sec
            )
            if ov <= 0:
                continue

            total_overlap += ov
            speaker_overlaps[diar.speaker] = (
                speaker_overlaps.get(diar.speaker, 0.0) + ov
            )

            if ov > best_overlap:
                best_overlap = ov
                best_speaker = diar.speaker
                best_diar_segment_id = diar.diar_segment_id

        duration = token.duration_sec if token.duration_sec is not None else 0.0
        overlap_ratio = (best_overlap / duration) if duration > 0 else 0.0
        total_overlap_ratio = (total_overlap / duration) if duration > 0 else 0.0

        overlap_rows.append(
            {
                "token_id": token.token_id,
                "has_diar_overlap": best_overlap > 0,
                "best_overlap_sec": best_overlap,
                "best_overlap_ratio": overlap_ratio,
                "total_overlap_sec": total_overlap,
                "total_overlap_ratio": total_overlap_ratio,
                "best_speaker": best_speaker,
                "best_diar_segment_id": best_diar_segment_id,
                "speaker_count_overlapping": len(speaker_overlaps),
            }
        )

    overlap_df = pd.DataFrame(overlap_rows)
    return token_df.merge(overlap_df, on="token_id", how="left")


# ------------------------------------------------------------
# Segment dataframe
# ------------------------------------------------------------


def build_segment_df(
    transcript_segments: list[dict[str, Any]],
    token_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    token_df_by_segment = token_df.copy()
    if "segment_id" in token_df_by_segment.columns:
        token_groups = dict(
            tuple(token_df_by_segment.groupby("segment_id", dropna=False))
        )
    else:
        token_groups = {}

    for i, seg in enumerate(transcript_segments):
        segment_id = seg.get("segment_id", f"seg_{i:06d}")
        seg_start = safe_float(
            seg.get("start_sec", seg.get("start_seconds", seg.get("start")))
        )
        seg_end = safe_float(seg.get("end_sec", seg.get("end_seconds", seg.get("end"))))
        start_token_id = seg.get("start_token_id")
        end_token_id = seg.get("end_token_id")
        text = seg.get("text", "")

        if isinstance(start_token_id, int) and isinstance(end_token_id, int):
            tok_group = token_df[
                (token_df["token_id"] >= start_token_id)
                & (token_df["token_id"] <= end_token_id)
            ].copy()
        else:
            tok_group = pd.DataFrame(columns=token_df.columns)

        actual_token_count = len(tok_group)
        expected_token_count = None
        if (
            isinstance(start_token_id, int)
            and isinstance(end_token_id, int)
            and end_token_id >= start_token_id
        ):
            expected_token_count = end_token_id - start_token_id + 1

        missing_token_count = None
        if expected_token_count is not None:
            missing_token_count = expected_token_count - actual_token_count

        sorted_tok = tok_group.sort_values(["start_sec", "end_sec"], na_position="last")
        internal_gaps = sorted_tok["gap_to_next_sec"].dropna()
        internal_gaps = internal_gaps[internal_gaps > 0]

        first_token_start = (
            sorted_tok["start_sec"].min() if not sorted_tok.empty else None
        )
        last_token_end = sorted_tok["end_sec"].max() if not sorted_tok.empty else None
        token_coverage_duration = None
        if pd.notna(first_token_start) and pd.notna(last_token_end):
            token_coverage_duration = float(last_token_end - first_token_start)

        segment_duration = None
        if seg_start is not None and seg_end is not None:
            segment_duration = seg_end - seg_start

        coverage_ratio = None
        if (
            token_coverage_duration is not None
            and segment_duration
            and segment_duration > 0
        ):
            coverage_ratio = token_coverage_duration / segment_duration

        orphan_token_count = (
            int((~sorted_tok["has_diar_overlap"].fillna(False)).sum())
            if not sorted_tok.empty
            else 0
        )
        repeat_count = (
            int(sorted_tok["is_repeat_with_prev"].fillna(False).sum())
            if not sorted_tok.empty
            else 0
        )
        overlapping_speakers = (
            sorted_tok["best_speaker"].dropna().unique().tolist()
            if not sorted_tok.empty
            else []
        )

        rows.append(
            {
                "segment_id": segment_id,
                "text": text,
                "segment_start_sec": seg_start,
                "segment_end_sec": seg_end,
                "segment_duration_sec": segment_duration,
                "start_token_id": start_token_id,
                "end_token_id": end_token_id,
                "expected_token_count": expected_token_count,
                "actual_token_count": actual_token_count,
                "missing_token_count": missing_token_count,
                "first_token_start_sec": first_token_start,
                "last_token_end_sec": last_token_end,
                "token_coverage_duration_sec": token_coverage_duration,
                "coverage_ratio": coverage_ratio,
                "max_internal_gap_sec": float(internal_gaps.max())
                if not internal_gaps.empty
                else 0.0,
                "sum_internal_gaps_sec": float(internal_gaps.sum())
                if not internal_gaps.empty
                else 0.0,
                "internal_gap_count": int(len(internal_gaps)),
                "repeat_count": repeat_count,
                "orphan_token_count": orphan_token_count,
                "speaker_count": len(overlapping_speakers),
                "speakers": overlapping_speakers,
                "near_30s_boundary": is_near_multiple(seg_start, 30.0, 0.25),
                "has_token_count_mismatch": (
                    expected_token_count is not None
                    and expected_token_count != actual_token_count
                ),
            }
        )

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Summaries
# ------------------------------------------------------------


def summarize_token_df(token_df: pd.DataFrame) -> dict[str, Any]:
    if token_df.empty:
        return {"token_count": 0}

    gap_series = token_df["gap_from_prev_sec"].dropna()
    positive_gaps = gap_series[gap_series > 0]

    return {
        "token_count": int(len(token_df)),
        "empty_token_count": int(token_df["is_empty"].sum()),
        "repeat_with_prev_count": int(token_df["is_repeat_with_prev"].sum()),
        "tokens_without_diar_overlap": int(
            (~token_df["has_diar_overlap"].fillna(False)).sum()
        ),
        "near_30s_boundary_count": int(token_df["near_30s_boundary"].sum()),
        "positive_gap_count": int(len(positive_gaps)),
        "positive_gap_p50_sec": float(positive_gaps.quantile(0.5))
        if len(positive_gaps)
        else None,
        "positive_gap_p90_sec": float(positive_gaps.quantile(0.9))
        if len(positive_gaps)
        else None,
        "positive_gap_max_sec": float(positive_gaps.max())
        if len(positive_gaps)
        else None,
    }


def summarize_segment_df(segment_df: pd.DataFrame) -> dict[str, Any]:
    if segment_df.empty:
        return {"segment_count": 0}

    mismatch_df = segment_df[segment_df["has_token_count_mismatch"] == True]
    large_gap_df = segment_df[segment_df["max_internal_gap_sec"] > 0.3]
    repeated_df = segment_df[segment_df["repeat_count"] > 0]

    return {
        "segment_count": int(len(segment_df)),
        "mismatch_count": int(len(mismatch_df)),
        "large_gap_segment_count_gt_300ms": int(len(large_gap_df)),
        "repeat_segment_count": int(len(repeated_df)),
        "segments_near_30s_boundary": int(segment_df["near_30s_boundary"].sum()),
        "coverage_ratio_p10": float(segment_df["coverage_ratio"].dropna().quantile(0.1))
        if segment_df["coverage_ratio"].notna().any()
        else None,
        "coverage_ratio_p50": float(segment_df["coverage_ratio"].dropna().quantile(0.5))
        if segment_df["coverage_ratio"].notna().any()
        else None,
        "coverage_ratio_p90": float(segment_df["coverage_ratio"].dropna().quantile(0.9))
        if segment_df["coverage_ratio"].notna().any()
        else None,
    }


# ------------------------------------------------------------
# CSV exports
# ------------------------------------------------------------


def export_rankings(
    output_dir: Path, token_df: pd.DataFrame, segment_df: pd.DataFrame
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    token_df.to_csv(output_dir / "tokens_exploration.csv", index=False)
    segment_df.to_csv(output_dir / "segments_exploration.csv", index=False)

    suspicious_tokens = token_df.copy()
    suspicious_tokens["gap_from_prev_sec_filled"] = suspicious_tokens[
        "gap_from_prev_sec"
    ].fillna(-1)
    suspicious_tokens = suspicious_tokens.sort_values(
        by=[
            "has_diar_overlap",
            "is_repeat_with_prev",
            "gap_from_prev_sec_filled",
            "near_30s_boundary",
        ],
        ascending=[True, False, False, False],
    )
    suspicious_tokens.head(300).to_csv(
        output_dir / "top_suspicious_tokens.csv", index=False
    )

    suspicious_segments = segment_df.sort_values(
        by=[
            "has_token_count_mismatch",
            "max_internal_gap_sec",
            "orphan_token_count",
            "repeat_count",
        ],
        ascending=[False, False, False, False],
    )
    suspicious_segments.head(300).to_csv(
        output_dir / "top_suspicious_segments.csv", index=False
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explore Whisper transcript tokens/segments against diarization before merge logic."
    )
    parser.add_argument("input_json", help="Path to pipeline JSON document")
    parser.add_argument(
        "--output-dir",
        help="Directory for CSV outputs",
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_json).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else input_path.parent / f"{input_path.stem}_exploration"
    )

    doc = load_json(input_path)
    print("Loaded JSON")

    transcript_segments = extract_transcript_segments(doc)
    print(f"Transcript segments: {len(transcript_segments)}")
    tokens = extract_tokens(doc)
    print(f"Tokens: {len(tokens)}")
    diar_segments = extract_diarization_segments(doc)
    print(f"Diar segments: {len(diar_segments)}")
    token_df = build_token_df(tokens)
    token_df = annotate_token_overlap(token_df, diar_segments)
    segment_df = build_segment_df(transcript_segments, token_df)

    export_rankings(output_dir, token_df, segment_df)

    summary = {
        "input_json": str(input_path),
        "output_dir": str(output_dir),
        "token_summary": summarize_token_df(token_df),
        "segment_summary": summarize_segment_df(segment_df),
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
