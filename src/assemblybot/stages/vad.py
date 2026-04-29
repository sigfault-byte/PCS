from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from silero_vad import get_speech_timestamps, load_silero_vad

from assemblybot.helper.directory import build_default_output_path
from assemblybot.helper.document import save_document
from assemblybot.models.document import CanonicalDocument
from assemblybot.models.factories import (
    create_empty_document,
    mark_stage_completed,
    mark_stage_failed,
    mark_stage_running,
)
from assemblybot.models.time import TimeRange
from assemblybot.models.vad import VadEngine, VadSection, VadSegment

DEFAULT_MODEL_ID = "silero-vad"
DEFAULT_ONNX = False
DEFAULT_OPSET_VERSION = 16
DEFAULT_THRESHOLD = 0.5
DEFAULT_MIN_SPEECH_DURATION_MS = 250
DEFAULT_MIN_SILENCE_DURATION_MS = 100
DEFAULT_SPEECH_PAD_MS = 30


def load_audio_as_mono_waveform(input_audio_path: Path) -> tuple[torch.Tensor, int, float]:
    """Load audio with soundfile and convert multichannel audio to mono."""
    import soundfile as sf

    audio, sample_rate = sf.read(str(input_audio_path), dtype="float32")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    waveform = torch.from_numpy(audio)
    duration_seconds = float(waveform.shape[-1] / sample_rate)

    return waveform, sample_rate, duration_seconds


def load_vad_model(
    onnx: bool,
    opset_version: int,
) -> torch.nn.Module:
    """Load Silero VAD from the installed package."""
    return load_silero_vad(onnx=onnx, opset_version=opset_version)


def build_vad_segments(
    speech_timestamps: list[dict[str, float]],
) -> list[VadSegment]:
    """Convert Silero timestamp dictionaries into canonical VAD segments."""
    return [
        VadSegment(
            segment_id=f"vad_{idx:06d}",
            time=TimeRange.from_seconds(
                float(timestamp["start"]),
                float(timestamp["end"]),
            ),
            confidence=None,
        )
        for idx, timestamp in enumerate(speech_timestamps, start=1)
    ]


def apply_vad_to_document(
    document: CanonicalDocument,
    segments: list[VadSegment],
    media_duration_seconds: float,
    model_id: str,
    threshold: float,
    min_speech_duration_ms: int,
    min_silence_duration_ms: int,
    speech_pad_ms: int,
) -> None:
    """Write VAD output into the typed canonical document."""
    speech_seconds_total = sum(segment.time.duration_seconds for segment in segments)
    non_speech_seconds_total = max(0.0, media_duration_seconds - speech_seconds_total)

    document.vad = VadSection(
        engine=VadEngine(
            name="silero-vad",
            model=model_id,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        ),
        segments=segments,
        speech_seconds_total=speech_seconds_total,
        non_speech_seconds_total=non_speech_seconds_total,
    )
    document.source.duration_seconds = media_duration_seconds


def run_vad(
    document: CanonicalDocument,
    input_audio_path: Path,
    output_json_path: Path,
    *,
    onnx: bool = DEFAULT_ONNX,
    opset_version: int = DEFAULT_OPSET_VERSION,
    threshold: float = DEFAULT_THRESHOLD,
    min_speech_duration_ms: int = DEFAULT_MIN_SPEECH_DURATION_MS,
    min_silence_duration_ms: int = DEFAULT_MIN_SILENCE_DURATION_MS,
    speech_pad_ms: int = DEFAULT_SPEECH_PAD_MS,
) -> CanonicalDocument:
    """Run Silero VAD and save the updated canonical document."""
    input_audio_path = input_audio_path.resolve()
    output_json_path = output_json_path.resolve()

    if not input_audio_path.exists():
        raise FileNotFoundError(f"Input audio not found: {input_audio_path}")

    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    mark_stage_running(document, "vad")
    stage_start_time = time.time()

    try:
        print(f"Loading Silero VAD: {DEFAULT_MODEL_ID}")
        print("Using device: cpu")
        print(f"Analyzing: {input_audio_path.name}")

        model = load_vad_model(
            onnx=onnx,
            opset_version=opset_version,
        )

        waveform, sample_rate, media_duration_seconds = load_audio_as_mono_waveform(
            input_audio_path
        )

        speech_timestamps = get_speech_timestamps(
            waveform,
            model,
            sampling_rate=sample_rate,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
            return_seconds=True,
        )

        segments = build_vad_segments(speech_timestamps)

        apply_vad_to_document(
            document=document,
            segments=segments,
            media_duration_seconds=media_duration_seconds,
            model_id=DEFAULT_MODEL_ID,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )

        mark_stage_completed(
            document,
            "vad",
            output_path=str(output_json_path),
        )
        save_document(document, output_json_path)

        elapsed = time.time() - stage_start_time

        print(f"VAD segments: {len(document.vad.segments)}")
        print(f"Speech seconds: {document.vad.speech_seconds_total:.2f}")
        print(f"Non-speech seconds: {document.vad.non_speech_seconds_total:.2f}")
        print(f"Canonical JSON: {output_json_path}")
        print(f"Time: {elapsed / 60:.1f} min")

        return document

    except Exception as exc:
        mark_stage_failed(document, "vad", str(exc))
        save_document(document, output_json_path)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Silero VAD and write canonical pipeline JSON.",
    )

    parser.add_argument(
        "input_audio",
        help="Path to input audio file",
    )

    parser.add_argument(
        "--output-json",
        help="Optional output canonical JSON path",
    )

    parser.add_argument(
        "--onnx",
        action="store_true",
        default=DEFAULT_ONNX,
        help="Load the ONNX Silero model",
    )

    parser.add_argument(
        "--opset-version",
        type=int,
        default=DEFAULT_OPSET_VERSION,
        help="ONNX opset version used by silero-vad",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Speech probability threshold",
    )

    parser.add_argument(
        "--min-speech-duration-ms",
        type=int,
        default=DEFAULT_MIN_SPEECH_DURATION_MS,
        help="Minimum speech segment duration",
    )

    parser.add_argument(
        "--min-silence-duration-ms",
        type=int,
        default=DEFAULT_MIN_SILENCE_DURATION_MS,
        help="Minimum silence duration used to split speech",
    )

    parser.add_argument(
        "--speech-pad-ms",
        type=int,
        default=DEFAULT_SPEECH_PAD_MS,
        help="Padding added around detected speech",
    )

    parser.add_argument(
        "--language",
        default="fr",
        help="Expected source language stored in the canonical document",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_audio_path = Path(args.input_audio).resolve()

    output_json_path = (
        Path(args.output_json).resolve()
        if args.output_json
        else build_default_output_path(
            input_audio_path,
            "_0_vad",
            "json",
        )
    )

    document = create_empty_document(
        input_audio_path,
        language_expected=args.language,
    )

    run_vad(
        document=document,
        input_audio_path=input_audio_path,
        output_json_path=output_json_path,
        onnx=args.onnx,
        opset_version=args.opset_version,
        threshold=args.threshold,
        min_speech_duration_ms=args.min_speech_duration_ms,
        min_silence_duration_ms=args.min_silence_duration_ms,
        speech_pad_ms=args.speech_pad_ms,
    )


if __name__ == "__main__":
    main()
