from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from faster_whisper import WhisperModel

from assemblybot.config import INTERIM_DIR
from assemblybot.models.factories import create_empty_document, mark_stage_completed
from assemblybot.models.time import TimeRange, seconds_to_timestamp
from assemblybot.models.transcript import TranscriptRawSegment, TranscriptRawToken


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
    compute_type: str = "auto",
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
        word_timestamps=True,
    )

    doc.transcript.engine.model = model_name
    doc.transcript.engine.device = device
    doc.transcript.engine.compute_type = compute_type
    doc.transcript.language_detected = info.language
    doc.transcript.language_probability = info.language_probability

    text_lines: list[str] = []

    # logs / progress
    total_duration = info.duration or 0.0  # seconds
    start_time = time.time()

    next_token_id = 0

    for idx, segment in enumerate(segments_iter, start=1):
        progress = segment.end / total_duration if total_duration else 0.0
        elapsed = time.time() - start_time
        speed = segment.end / elapsed if elapsed > 0 else 0.0

        time_range = TimeRange.from_seconds(segment.start, segment.end)

        bar_width = 30
        filled = int(progress * bar_width)
        bar = "=" * filled + " " * (bar_width - filled)

        sys.stdout.write(f"\r[{bar}] {progress * 100:5.1f}% | {speed:4.2f}x realtime")
        sys.stdout.flush()

        segment_token_start_id = next_token_id

        for word in segment.words or []:
            if word.start is None or word.end is None:
                continue

            raw_token_text = word.word
            if raw_token_text is None:
                continue

            raw_token = TranscriptRawToken(
                token_id=next_token_id,
                start_seconds=float(word.start),
                end_seconds=float(word.end),
                raw_token=raw_token_text,
            )
            doc.transcript.raw_tokens.append(raw_token)
            next_token_id += 1

        segment_token_end_id = next_token_id - 1

        if segment_token_end_id < segment_token_start_id:
            # Fallback: keep the segment anchor even if no word timings were emitted.
            segment_token_start_id = -1
            segment_token_end_id = -1

        raw_segment = TranscriptRawSegment(
            segment_id=f"whisper_{idx:06d}",
            start_token_id=segment_token_start_id,
            end_token_id=segment_token_end_id,
            time=time_range,
        )
        doc.transcript.raw_segments.append(raw_segment)

        text_lines.append(
            f"[{seconds_to_timestamp(segment.start)} -> "
            f"{seconds_to_timestamp(segment.end)}]"
        )

    elapsed = time.time() - start_time
    print()

    doc.transcript.tokens_count = len(doc.transcript.raw_tokens)
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
    print(f"Tokens: {doc.transcript.tokens_count}")
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
    parser.add_argument("--compute-type", default="auto", help="Compute type")
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
