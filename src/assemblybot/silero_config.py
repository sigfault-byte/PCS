from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class SileroVadConfig:
    """Runtime settings for the Silero VAD stage."""

    model_id: str = "silero-vad"
    onnx: bool = False
    opset_version: int = 16
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 100
    speech_pad_ms: int = 30

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "SileroVadConfig":
        return cls(
            onnx=args.onnx,
            opset_version=args.opset_version,
            threshold=args.threshold,
            min_speech_duration_ms=args.min_speech_duration_ms,
            min_silence_duration_ms=args.min_silence_duration_ms,
            speech_pad_ms=args.speech_pad_ms,
        )


DEFAULT_SILERO_VAD_CONFIG = SileroVadConfig()


def add_silero_vad_arguments(parser: argparse.ArgumentParser) -> None:
    """Register Silero-specific CLI options on a stage parser."""
    parser.add_argument(
        "--onnx",
        action="store_true",
        default=DEFAULT_SILERO_VAD_CONFIG.onnx,
        help="Load the ONNX Silero model",
    )

    parser.add_argument(
        "--opset-version",
        type=int,
        default=DEFAULT_SILERO_VAD_CONFIG.opset_version,
        help="ONNX opset version used by silero-vad",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SILERO_VAD_CONFIG.threshold,
        help="Speech probability threshold",
    )

    parser.add_argument(
        "--min-speech-duration-ms",
        type=int,
        default=DEFAULT_SILERO_VAD_CONFIG.min_speech_duration_ms,
        help="Minimum speech segment duration",
    )

    parser.add_argument(
        "--min-silence-duration-ms",
        type=int,
        default=DEFAULT_SILERO_VAD_CONFIG.min_silence_duration_ms,
        help="Minimum silence duration used to split speech",
    )

    parser.add_argument(
        "--speech-pad-ms",
        type=int,
        default=DEFAULT_SILERO_VAD_CONFIG.speech_pad_ms,
        help="Padding added around detected speech",
    )
