from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class FasterWhisperTranscriptionConfig:
    """Runtime settings for the faster-whisper transcription stage."""

    transcription_model_name: str = "large-v3"
    device: str = "auto"
    compute_type: str = "auto"
    language: str = "fr"
    beam_size: int = 5
    vad_filter: bool = True
    vad_min_silence_duration_ms: int = 1000
    vad_speech_pad_ms: int = 400
    temperature: tuple[float, ...] = (0.0, 0.2, 0.4)
    condition_on_previous_text: bool = True
    word_timestamps: bool = True

    @classmethod
    def from_args(
        cls,
        args: argparse.Namespace,
    ) -> "FasterWhisperTranscriptionConfig":
        return cls(
            transcription_model_name=args.transcription_model,
            device=args.device,
            compute_type=args.compute_type,
            language=args.language,
            beam_size=args.beam_size,
            vad_filter=not args.no_vad_filter,
            vad_min_silence_duration_ms=args.vad_min_silence_duration_ms,
            vad_speech_pad_ms=args.vad_speech_pad_ms,
            temperature=tuple(args.temperature),
            condition_on_previous_text=not args.no_condition_on_previous_text,
            word_timestamps=not args.no_word_timestamps,
        )


DEFAULT_FASTER_WHISPER_TRANSCRIPTION_CONFIG = FasterWhisperTranscriptionConfig()


def add_faster_whisper_transcription_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    """Register faster-whisper-specific CLI options on a stage parser."""
    parser.add_argument(
        "--transcription-model",
        default=DEFAULT_FASTER_WHISPER_TRANSCRIPTION_CONFIG.transcription_model_name,
        help="faster-whisper model name",
    )

    parser.add_argument(
        "--device",
        default=DEFAULT_FASTER_WHISPER_TRANSCRIPTION_CONFIG.device,
        help="Device to use: auto, cpu, cuda",
    )

    parser.add_argument(
        "--compute-type",
        default=DEFAULT_FASTER_WHISPER_TRANSCRIPTION_CONFIG.compute_type,
        help="Compute type: auto, int8, float16, float32, etc.",
    )

    parser.add_argument(
        "--language",
        default=DEFAULT_FASTER_WHISPER_TRANSCRIPTION_CONFIG.language,
        help="Expected transcription language",
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=DEFAULT_FASTER_WHISPER_TRANSCRIPTION_CONFIG.beam_size,
        help="Beam size for decoding",
    )

    parser.add_argument(
        "--no-vad-filter",
        action="store_true",
        default=not DEFAULT_FASTER_WHISPER_TRANSCRIPTION_CONFIG.vad_filter,
        help="Disable faster-whisper VAD filtering",
    )

    parser.add_argument(
        "--vad-min-silence-duration-ms",
        type=int,
        default=(
            DEFAULT_FASTER_WHISPER_TRANSCRIPTION_CONFIG.vad_min_silence_duration_ms
        ),
        help="Minimum silence duration for VAD filtering",
    )

    parser.add_argument(
        "--vad-speech-pad-ms",
        type=int,
        default=DEFAULT_FASTER_WHISPER_TRANSCRIPTION_CONFIG.vad_speech_pad_ms,
        help="Speech padding for VAD filtering",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        nargs="+",
        default=list(DEFAULT_FASTER_WHISPER_TRANSCRIPTION_CONFIG.temperature),
        help="Sampling temperature fallback schedule for decoding",
    )

    parser.add_argument(
        "--no-condition-on-previous-text",
        action="store_true",
        default=(
            not DEFAULT_FASTER_WHISPER_TRANSCRIPTION_CONFIG.condition_on_previous_text
        ),
        help="Disable conditioning each decode window on previous text",
    )

    parser.add_argument(
        "--no-word-timestamps",
        action="store_true",
        default=not DEFAULT_FASTER_WHISPER_TRANSCRIPTION_CONFIG.word_timestamps,
        help="Disable word timestamps",
    )
