from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from faster_whisper import WhisperModel

from assemblybot.config import INTERIM_DIR
from assemblybot.models.factories import create_empty_document, mark_stage_completed
from assemblybot.models.time import TimeRange, seconds_to_timestamp
from assemblybot.models.transcript import TranscriptRawSegment


# Better safe than sorry -> move to global config
# -------------------------------
def resolve_device_and_compute(device: str, compute_type: str):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    print(f"Running on {device}!")
    return device, compute_type


# ------------------------------


def build_default_output_path(input_path: Path) -> Path:
    return INTERIM_DIR / f"{input_path.stem}_01_transcript.json"


def build_default_text_output_path(input_path: Path) -> Path:
    return INTERIM_DIR / f"{input_path.stem}_01_transcript.txt"


def transcribe_audio(
    input_path: Path,
    output_json_path: Path | None = None,
    output_txt_path: Path | None = None,
    model_name: str = "large-v3",
    device: str = "auto",
    compute_type: str = "int8",
    language: str = "fr",
    beam_size: int = 5,
    vad_filter: bool = True,
    min_silence_duration_ms: int = 500,
):
    input_path = input_path.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input audio not found: {input_path}")

    output_json_path = output_json_path or build_default_output_path(input_path)
    output_txt_path = output_txt_path or build_default_text_output_path(input_path)

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)

    doc = create_empty_document(input_path=input_path, language_expected=language)

    device, compute_type = resolve_device_and_compute(device, compute_type)

    print("Loading Whisper model...")
    print(f"Using device: {device} ({compute_type})")

    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    print("Model loaded")

    print(f"Transcribing: {input_path.name}")
    start_time = time.time()

    segments_iter, info = model.transcribe(
        str(input_path),
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
        vad_parameters={"min_silence_duration_ms": min_silence_duration_ms},
    )

    doc.transcript.engine.model = model_name
    doc.transcript.engine.device = device
    doc.transcript.engine.compute_type = compute_type
    doc.transcript.language_detected = info.language
    doc.transcript.language_probability = info.language_probability

    text_lines: list[str] = []

    for idx, segment in enumerate(segments_iter, start=1):
        cleaned_text = segment.text.strip()
        time_range = TimeRange.from_seconds(segment.start, segment.end)

        raw_segment = TranscriptRawSegment(
            segment_id=f"whisper_{idx:06d}",
            time=time_range,
            text=cleaned_text,
        )
        doc.transcript.raw_segments.append(raw_segment)

        text_lines.append(
            f"[{seconds_to_timestamp(segment.start)} -> "
            f"{seconds_to_timestamp(segment.end)}] {cleaned_text}"
        )

    elapsed = time.time() - start_time

    doc.transcript.segments_count = len(doc.transcript.raw_segments)
    doc.source.duration_seconds = (
        doc.transcript.raw_segments[-1].time.end_seconds
        if doc.transcript.raw_segments
        else 0.0
    )

    mark_stage_completed(
        doc,
        stage_name="transcription",
        output_path=str(output_json_path),
    )

    with output_txt_path.open("w", encoding="utf-8") as f:
        for line in text_lines:
            f.write(line + "\n")

    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(doc.to_dict(), f, ensure_ascii=False, indent=2)

    print(
        f"Detected language: {doc.transcript.language_detected} "
        f"(confidence: {doc.transcript.language_probability:.2f})"
    )
    print(f"Segments: {doc.transcript.segments_count}")
    print(f"Time: {elapsed / 60:.1f} min")
    print(f"Text transcript: {output_txt_path}")
    print(f"JSON transcript: {output_json_path}")

    return doc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file with faster-whisper"
    )
    parser.add_argument("--input", required=True, help="Path to input .wav file")
    parser.add_argument("--output-json", help="Optional output JSON path")
    parser.add_argument("--output-txt", help="Optional output text path")
    parser.add_argument("--model", default="large-v3", help="Whisper model name")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda")
    parser.add_argument("--compute-type", default="int8", help="Compute type")
    parser.add_argument("--language", default="fr", help="Language code")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size")
    parser.add_argument(
        "--min-silence-ms",
        type=int,
        default=500,
        help="VAD min silence duration in milliseconds",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable VAD filtering",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    transcribe_audio(
        input_path=Path(args.input),
        output_json_path=Path(args.output_json) if args.output_json else None,
        output_txt_path=Path(args.output_txt) if args.output_txt else None,
        model_name=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        beam_size=args.beam_size,
        vad_filter=not args.no_vad,
        min_silence_duration_ms=args.min_silence_ms,
    )


if __name__ == "__main__":
    main()
