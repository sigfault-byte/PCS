from __future__ import annotations

import argparse
import json
from pathlib import Path

from assemblybot.config import INTERIM_DIR
from assemblybot.models.time import now_utc_iso

SUSPICIOUS_SHORT_TOKEN_SECONDS = 0.08
REPEATED_TOKEN_CLUSTER_WINDOW_SECONDS = 2.0

RESCUE_LOOKAHEAD_SECONDS = 1.5
RESCUE_MAX_BURST_DURATION_SECONDS = 1.5
RESCUE_MAX_GAP_TO_PREVIOUS_SEGMENT_SECONDS = 0.4
TERMINAL_PUNCTUATION = (".", "!", "?", "…")


def build_default_output_path(input_json_path: Path) -> Path:
    """
    Build the default merged JSON output path.

    Example:
        something_03_diarization_collapsed.json
        -> something_04_merged.json
    """
    stem = input_json_path.stem.replace("_03_diarization_collapsed", "")
    return INTERIM_DIR / f"{stem}_04_merged.json"


def load_document(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_document(doc: dict, json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)


def reconstruct_text_from_tokens(tokens: list[dict]) -> str:
    """
    Rebuild exact raw text from raw tokens.

    Important:
    raw_token already carries Whisper spacing,
    so we must use ''.join(...) and not ' '.join(...).
    """
    return "".join(tok["raw_token"] for tok in tokens)


def build_token_provenance(tokens: list[dict]) -> tuple[int | None, int | None]:
    """
    Return the first/last token id used by a merged segment.
    """
    if not tokens:
        return None, None
    return tokens[0]["token_id"], tokens[-1]["token_id"]


def build_token_anchor_points(transcript_tokens: list[dict]) -> list[dict]:
    """
    Build one anchor point per token.

    Rule:
    - token i anchor = (token_i.start_seconds + token_{i+1}.end_seconds) / 2
    - last token anchor = (token_i.start_seconds + token_i.end_seconds) / 2

    This intentionally biases assignment toward the local speech flow
    and helps recover boundary tokens that would otherwise fall just
    outside diarization turn starts.
    """
    if not transcript_tokens:
        return []

    tokens_sorted = sorted(
        transcript_tokens,
        key=lambda tok: tok["start_seconds"],
    )

    anchored_tokens: list[dict] = []

    for i, tok in enumerate(tokens_sorted):
        if i < len(tokens_sorted) - 1:
            next_tok = tokens_sorted[i + 1]
            anchor = (tok["start_seconds"] + next_tok["end_seconds"]) / 2.0
        else:
            anchor = (tok["start_seconds"] + tok["end_seconds"]) / 2.0

        anchored_tokens.append(
            {
                "token": tok,
                "anchor_seconds": anchor,
            }
        )

    return anchored_tokens


def build_token_to_transcript_segment_map(
    transcript_segments: list[dict],
) -> dict[int, str]:
    """
    Map each token_id to its parent Whisper raw segment_id.
    """
    token_to_segment: dict[int, str] = {}

    for seg in transcript_segments:
        start_token_id = seg.get("start_token_id")
        end_token_id = seg.get("end_token_id")

        if start_token_id is None or end_token_id is None:
            continue
        if start_token_id < 0 or end_token_id < 0:
            continue

        for token_id in range(start_token_id, end_token_id + 1):
            token_to_segment[token_id] = seg["segment_id"]

    return token_to_segment


def build_transcript_segment_provenance(
    tokens: list[dict],
    token_to_transcript_segment: dict[int, str],
) -> list[str]:
    """
    Recover which Whisper raw segments contributed to this final segment.
    """
    segment_ids: list[str] = []
    seen: set[str] = set()

    for tok in tokens:
        seg_id = token_to_transcript_segment.get(tok["token_id"])
        if seg_id is None or seg_id in seen:
            continue
        seen.add(seg_id)
        segment_ids.append(seg_id)

    return segment_ids


def assign_tokens_to_collapsed_segments(
    transcript_tokens: list[dict],
    collapsed_segments: list[dict],
) -> tuple[dict[str, list[dict]], list[dict]]:
    """
    Assign each transcript token to the collapsed diarization segment
    whose time span contains the token anchor.

    Anchor rule:
    - token i anchor = (token_i.start + token_{i+1}.end) / 2
    - last token anchor = midpoint of the token itself

    Returns:
        assigned_by_segment_id, unassigned_tokens
    """
    assigned: dict[str, list[dict]] = {
        seg["segment_id"]: [] for seg in collapsed_segments
    }
    unassigned_tokens: list[dict] = []

    collapsed_sorted = sorted(
        collapsed_segments,
        key=lambda seg: seg["time"]["start_seconds"],
    )
    anchored_tokens = build_token_anchor_points(transcript_tokens)

    seg_idx = 0
    seg_count = len(collapsed_sorted)

    for item in anchored_tokens:
        tok = item["token"]
        anchor = item["anchor_seconds"]

        # Advance until the current collapsed segment could still contain this anchor.
        while seg_idx < seg_count:
            current_seg = collapsed_sorted[seg_idx]
            seg_end = current_seg["time"]["end_seconds"]

            if anchor <= seg_end:
                break

            seg_idx += 1

        if seg_idx >= seg_count:
            unassigned_tokens.append(tok)
            continue

        current_seg = collapsed_sorted[seg_idx]
        seg_start = current_seg["time"]["start_seconds"]
        seg_end = current_seg["time"]["end_seconds"]

        if seg_start <= anchor <= seg_end:
            assigned[current_seg["segment_id"]].append(tok)
        else:
            unassigned_tokens.append(tok)

    return assigned, unassigned_tokens


def build_final_segments(
    doc: dict,
    assigned_tokens_by_segment: dict[str, list[dict]],
) -> list[dict]:
    """
    Build final merged segments from collapsed diarization turns
    plus assigned transcript tokens.
    """
    transcript_segments = doc["transcript"]["raw_segments"]
    collapsed_segments = doc["diarization"]["collapsed_segments"]

    token_to_transcript_segment = build_token_to_transcript_segment_map(
        transcript_segments
    )

    final_segments: list[dict] = []

    collapsed_sorted = sorted(
        collapsed_segments,
        key=lambda seg: seg["time"]["start_seconds"],
    )

    language_detected = doc["transcript"].get("language_detected")

    for idx, cseg in enumerate(collapsed_sorted, start=1):
        collapsed_segment_id = cseg["segment_id"]
        speaker_id = cseg["speaker_id"]
        tokens = assigned_tokens_by_segment.get(collapsed_segment_id, [])

        text_raw = reconstruct_text_from_tokens(tokens)
        token_start_id, token_end_id = build_token_provenance(tokens)
        transcript_segment_ids = build_transcript_segment_provenance(
            tokens,
            token_to_transcript_segment,
        )

        final_segments.append(
            {
                "segment_id": f"seg_{idx:06d}",
                "time": cseg["time"],
                "speaker": {
                    "speaker_id": speaker_id,
                    "speaker_label": None,
                    "speaker_label_source": None,
                    "confidence": None,
                },
                "text": {
                    "raw": text_raw,
                    "normalized": None,
                    "language": language_detected,
                },
                "flags": 0,
                "entities": [],
                "keywords": [],
                "provenance": {
                    "transcript_segment_ids": transcript_segment_ids,
                    "diarization_segment_ids": cseg["source_diarization_segment_ids"],
                    "transcript_token_start_id": token_start_id,
                    "transcript_token_end_id": token_end_id,
                    "stage_created_by": "merge",
                },
            }
        )

    return final_segments


def merge_segments(
    input_json_path: Path,
    output_json_path: Path | None = None,
) -> dict:
    """
    Merge transcript raw tokens onto collapsed diarization turns.
    """
    input_json_path = input_json_path.resolve()

    if not input_json_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_json_path}")

    output_json_path = output_json_path or build_default_output_path(input_json_path)

    doc = load_document(input_json_path)

    transcript_tokens = doc["transcript"]["raw_tokens"]
    collapsed_segments = doc["diarization"]["collapsed_segments"]

    assigned_tokens_by_segment, unassigned_tokens = assign_tokens_to_collapsed_segments(
        transcript_tokens=transcript_tokens,
        collapsed_segments=collapsed_segments,
    )

    final_segments = build_final_segments(
        doc=doc,
        assigned_tokens_by_segment=assigned_tokens_by_segment,
    )

    doc["segments"] = final_segments

    if "merge" not in doc["pipeline"]["stages_completed"]:
        doc["pipeline"]["stages_completed"].append("merge")

    doc["pipeline"].setdefault("stage_outputs", {})
    doc["pipeline"]["stage_outputs"]["merge"] = str(output_json_path)
    doc["pipeline"]["updated_at"] = now_utc_iso()

    save_document(doc, output_json_path)

    print(f"Collapsed diarization segments: {len(collapsed_segments)}")
    print(f"Final merged segments: {len(final_segments)}")
    print(f"Unassigned tokens: {len(unassigned_tokens)}")
    print(f"Output: {output_json_path}")

    return doc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge transcript tokens onto collapsed diarization turns."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help="Path to transcript + collapsed diarization JSON",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output JSON path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    merge_segments(
        input_json_path=args.input_json,
        output_json_path=args.output_json,
    )


if __name__ == "__main__":
    main()
