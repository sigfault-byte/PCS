from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from pyannote.audio import Pipeline

from assemblybot.config import INTERIM_DIR
from assemblybot.models.diarization import DiarizationRawSegment
from assemblybot.models.factories import mark_stage_completed
from assemblybot.models.time import TimeRange


def build_default_output_path(input_audio_path: Path) -> Path:
    return INTERIM_DIR / f"{input_audio_path.stem}_02_diarization.json"


def load_document(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_document(doc: dict, json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def diarize_audio(
    input_audio_path: Path,
    input_json_path: Path,
    output_json_path: Path | None = None,
    hf_token: str | None = None,
    model_name: str = "pyannote/speaker-diarization-3.1",
    device: str = "auto",
) -> dict:
    input_audio_path = input_audio_path.resolve()
    input_json_path = input_json_path.resolve()

    if not input_audio_path.exists():
        raise FileNotFoundError(f"Input audio not found: {input_audio_path}")

    if not input_json_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_json_path}")

    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    if not hf_token:
        raise RuntimeError(
            "Missing Hugging Face token. Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN."
        )

    output_json_path = output_json_path or build_default_output_path(input_audio_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(device)
    torch_device = torch.device(device)

    print(f"Loading diarization pipeline: {model_name}")
    print(f"Using device: {device}")

    start_time = time.time()

    pipeline = Pipeline.from_pretrained(
        model_name,
        token=hf_token,
    )
    pipeline.to(torch_device)

    print(f"Diarizing: {input_audio_path.name}")
    diarization = pipeline(str(input_audio_path))

    elapsed = time.time() - start_time

    doc = load_document(input_json_path)

    # Fill diarization metadata
    doc["diarization"]["engine"]["model"] = model_name
    doc["diarization"]["engine"]["device"] = device
    doc["diarization"]["raw_segments"] = []

    speaker_ids: set[str] = set()

    for idx, (turn, _, speaker_label) in enumerate(
        diarization.itertracks(yield_label=True), start=1
    ):
        speaker_ids.add(speaker_label)

        raw_segment = DiarizationRawSegment(
            segment_id=f"dia_{idx:06d}",
            time=TimeRange.from_seconds(turn.start, turn.end),
            speaker_id=speaker_label,
            confidence=None,
        )

        doc["diarization"]["raw_segments"].append(
            {
                "segment_id": raw_segment.segment_id,
                "time": {
                    "start_seconds": raw_segment.time.start_seconds,
                    "end_seconds": raw_segment.time.end_seconds,
                    "duration_seconds": raw_segment.time.duration_seconds,
                    "start_ts": raw_segment.time.start_ts,
                    "end_ts": raw_segment.time.end_ts,
                },
                "speaker_id": raw_segment.speaker_id,
                "confidence": raw_segment.confidence,
            }
        )

    doc["diarization"]["speakers_count"] = len(speaker_ids)

    mark_stage_completed(
        doc,
        stage_name="diarization",
        output_path=str(output_json_path),
    )

    save_document(doc, output_json_path)

    print(f"✓ Speakers detected: {doc['diarization']['speakers_count']}")
    print(f"✓ Raw diarization segments: {len(doc['diarization']['raw_segments'])}")
    print(f"✓ Time: {elapsed / 60:.1f} min")
    print(f"✓ JSON transcript+diarization: {output_json_path}")

    return doc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pyannote diarization and enrich canonical JSON."
    )
    parser.add_argument("--input-audio", required=True, help="Path to input .wav file")
    parser.add_argument(
        "--input-json",
        required=True,
        help="Path to existing transcript JSON",
    )
    parser.add_argument(
        "--output-json",
        help="Optional output JSON path",
    )
    parser.add_argument(
        "--model",
        default="pyannote/speaker-diarization-3.1",
        help="Pyannote model name",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: auto, cpu, cuda",
    )
    parser.add_argument(
        "--hf-token",
        help="Optional Hugging Face token (otherwise reads HF_TOKEN or HUGGINGFACE_HUB_TOKEN)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    diarize_audio(
        input_audio_path=Path(args.input_audio),
        input_json_path=Path(args.input_json),
        output_json_path=Path(args.output_json) if args.output_json else None,
        hf_token=args.hf_token,
        model_name=args.model,
        device=args.device,
    )


if __name__ == "__main__":
    main()
