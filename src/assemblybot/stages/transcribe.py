from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel

from assemblybot.config import INTERIM_DIR


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptResult:
    input_file: str
    language: str
    language_probability: float
    model_name: str
    device: str
    compute_type: str
    elapsed_seconds: float
    segments: list[TranscriptSegment]


def format_timestamp(seconds: float) -> str:
    return f"{seconds:.2f}s"


def build_default_output_path(input_path: Path) -> Path:
    return INTERIM_DIR / f"{input_path.stem}_transcript.json"


def build_default_text_output_path(input_path: Path) -> Path:
    return INTERIM_DIR / f"{input_path.stem}_transcript.txt"


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
) -> TranscriptResult:
    input_path = input_path.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input audio not found: {input_path}")

    output_json_path = output_json_path or build_default_output_path(input_path)
    output_txt_path = output_txt_path or build_default_text_output_path(input_path)

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading Whisper model...")
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

    segments: list[TranscriptSegment] = [
        TranscriptSegment(
            start=segment.start,
            end=segment.end,
            text=segment.text.strip(),
        )
        for segment in segments_iter
    ]

    elapsed = time.time() - start_time

    result = TranscriptResult(
        input_file=str(input_path),
        language=info.language,
        language_probability=info.language_probability,
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        elapsed_seconds=elapsed,
        segments=segments,
    )

    with output_txt_path.open("w", encoding="utf-8") as f:
        for segment in segments:
            line = (
                f"[{format_timestamp(segment.start)} -> "
                f"{format_timestamp(segment.end)}] {segment.text}"
            )
            f.write(line + "\n")

    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)

    print(
        f"Detected language: {result.language} "
        f"(confidence: {result.language_probability:.2f})"
    )
    print(f"Segments: {len(result.segments)}")
    print(f"Time: {elapsed / 60:.1f} min")
    print(f"Text transcript: {output_txt_path}")
    print(f"JSON transcript: {output_json_path}")

    return result


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
        help="Disable VAD filter",
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
