from __future__ import annotations

import argparse
import time
from pathlib import Path

from assemblybot.helper.directory import build_default_output_path
from assemblybot.helper.document import load_document, save_document
from assemblybot.models.document import CanonicalDocument
from assemblybot.models.factories import (
    create_empty_document,
    mark_stage_completed,
    mark_stage_failed,
    mark_stage_running,
)
from assemblybot.models.time import TimeRange
from assemblybot.models.transcript import TranscriptRawSegment, TranscriptRawToken


def resolve_device_and_compute(
    device: str,
    compute_type: str,
) -> tuple[str, str]:
    """
    Normalize runtime configuration for faster-whisper.

    CPU generally prefers int8.
    CUDA can use float16 unless caller explicitly asks otherwise.
    """
    if device == "auto":
        try:
            import torch  # type: ignore

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    return device, compute_type


def build_transcript_text_lines(
    raw_segments: list[TranscriptRawSegment],
) -> list[str]:
    """
    Build a simple human-readable transcript text artifact from raw segments.
    """
    lines: list[str] = []
    for segment in raw_segments:
        lines.append(
            f"[{segment.time.start_ts} -> {segment.time.end_ts}] {segment.raw_text}"
        )
    return lines


def apply_transcript_to_document(
    document: CanonicalDocument,
    model_name: str,
    device: str,
    compute_type: str,
    language_detected: str | None,
    language_probability: float | None,
    raw_tokens: list[TranscriptRawToken],
    raw_segments: list[TranscriptRawSegment],
    media_duration_seconds: float | None,
) -> None:
    """
    Write raw transcription results into the typed canonical document.
    """
    document.transcript.engine.model = model_name
    document.transcript.engine.device = device
    document.transcript.engine.compute_type = compute_type
    document.transcript.language_detected = language_detected
    document.transcript.language_probability = language_probability
    document.transcript.raw_tokens = raw_tokens
    document.transcript.raw_segments = raw_segments

    # If transcription stage learned the full media duration, keep it on source.
    # If diarization already filled it, this should just confirm the same value.
    if media_duration_seconds is not None:
        document.source.duration_seconds = media_duration_seconds


def transcribe_audio(
    document: CanonicalDocument,
    input_audio_path: Path,
    output_json_path: Path,
    *,
    output_txt_path: Path | None = None,
    model_name: str = "large-v3",
    device: str = "auto",
    compute_type: str = "auto",
    language: str = "fr",
    beam_size: int = 5,
    vad_filter: bool = True,
    vad_min_silence_duration_ms: int = 2000,
    word_timestamps: bool = True,
) -> CanonicalDocument:
    """
    Main transcription stage orchestrator.

    This stage is intentionally "raw":
    - raw_tokens = smallest timestamped text truth
    - raw_segments = Whisper decoder chunk anchors
    - no downstream repair / merge logic here
    """
    input_audio_path = input_audio_path.resolve()
    output_json_path = output_json_path.resolve()

    if not input_audio_path.exists():
        raise FileNotFoundError(f"Input audio not found: {input_audio_path}")

    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    if output_txt_path is None:
        output_txt_path = build_default_output_path(
            input_audio_path,
            "_02_transcript",
            "txt",
        )
    output_txt_path = output_txt_path.resolve()
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)

    device, compute_type = resolve_device_and_compute(device, compute_type)

    mark_stage_running(document, "transcription")

    stage_start_time = time.time()

    try:
        from faster_whisper import WhisperModel  # type: ignore

        print(f"Loading faster-whisper model: {model_name}")
        print(f"Using device: {device} ({compute_type})")
        print(f"Transcribing: {input_audio_path.name}")

        model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )

        segments_iter, info = model.transcribe(
            str(input_audio_path),
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            vad_parameters={
                "min_silence_duration_ms": vad_min_silence_duration_ms,
            },
            word_timestamps=word_timestamps,
        )

        raw_tokens: list[TranscriptRawToken] = []
        raw_segments: list[TranscriptRawSegment] = []

        token_id_counter = 0
        media_duration_seconds: float = 0.0

        # We consume the iterator once, converting immediately into our own models.
        for segment_index, segment in enumerate(segments_iter, start=1):
            segment_token_start_id: int | None = None
            segment_token_end_id: int | None = None

            # Keep the segment text as emitted by the decoder.
            segment_raw_text = (segment.text or "").strip()

            # If word timestamps exist, use them as our smallest text-time truth.
            words = getattr(segment, "words", None) or []

            for word in words:
                word_start = getattr(word, "start", None)
                word_end = getattr(word, "end", None)
                word_text = getattr(word, "word", None)

                if word_start is None or word_end is None:
                    continue
                if word_text is None:
                    continue

                if segment_token_start_id is None:
                    segment_token_start_id = token_id_counter
                segment_token_end_id = token_id_counter

                raw_tokens.append(
                    TranscriptRawToken(
                        token_id=token_id_counter,
                        start_seconds=float(word_start),
                        end_seconds=float(word_end),
                        raw_token=str(word_text),
                    )
                )
                token_id_counter += 1

            segment_time = TimeRange.from_seconds(
                float(segment.start),
                float(segment.end),
            )

            raw_segments.append(
                TranscriptRawSegment(
                    segment_id=segment_index,
                    start_token_id=segment_token_start_id,
                    end_token_id=segment_token_end_id,
                    time=segment_time,
                    raw_text=segment_raw_text,
                    avg_logprob=getattr(segment, "avg_logprob", None),
                    no_speech_prob=getattr(segment, "no_speech_prob", None),
                    compression_ratio=getattr(segment, "compression_ratio", None),
                )
            )

            media_duration_seconds = max(
                media_duration_seconds,
                float(segment.end),
            )

            if segment_index % 100 == 0:
                print(
                    f"[segment {segment_index:06d}] "
                    f"up to {segment.end:.2f}s | "
                    f"tokens: {token_id_counter}"
                )

        apply_transcript_to_document(
            document=document,
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            language_detected=getattr(info, "language", None),
            language_probability=getattr(info, "language_probability", None),
            raw_tokens=raw_tokens,
            raw_segments=raw_segments,
            media_duration_seconds=media_duration_seconds or None,
        )

        # Save a simple transcript text artifact for human inspection/debugging.
        lines = build_transcript_text_lines(raw_segments)
        output_txt_path.write_text("\n".join(lines), encoding="utf-8")

        mark_stage_completed(
            document,
            "transcription",
            output_path=str(output_json_path),
        )
        save_document(document, output_json_path)

        elapsed = time.time() - stage_start_time

        print(f"Language detected: {document.transcript.language_detected}")
        print(f"Language probability: {document.transcript.language_probability}")
        print(f"Raw transcript tokens: {len(document.transcript.raw_tokens)}")
        print(f"Raw transcript segments: {len(document.transcript.raw_segments)}")
        print(f"Transcript TXT: {output_txt_path}")
        print(f"Canonical JSON: {output_json_path}")
        print(f"Time: {elapsed / 60:.1f} min")

        return document

    except Exception as exc:
        mark_stage_failed(document, "transcription", str(exc))
        save_document(document, output_json_path)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run faster-whisper transcription and create/update the canonical document."
    )

    parser.add_argument(
        "--input-audio",
        required=True,
        help="Path to input audio file",
    )

    parser.add_argument(
        "--input-json",
        help="Optional existing canonical document JSON to resume from",
    )

    parser.add_argument(
        "--output-json",
        help="Optional output JSON path (default: generated in interim directory)",
    )

    parser.add_argument(
        "--output-txt",
        help="Optional output transcript text path",
    )

    parser.add_argument(
        "--model",
        default="large-v3",
        help="faster-whisper model name",
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cpu, cuda",
    )

    parser.add_argument(
        "--compute-type",
        default="auto",
        help="Compute type: auto, int8, float16, float32, etc.",
    )

    parser.add_argument(
        "--language",
        default="fr",
        help="Expected transcription language",
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for decoding",
    )

    parser.add_argument(
        "--no-vad-filter",
        action="store_true",
        help="Disable faster-whisper VAD filtering",
    )

    parser.add_argument(
        "--vad-min-silence-duration-ms",
        type=int,
        default=2000,
        help="Minimum silence duration for VAD filtering",
    )

    parser.add_argument(
        "--no-word-timestamps",
        action="store_true",
        help="Disable word timestamps",
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
            "_02_transcription",
            "json",
        )
    )

    # Standalone mode: create a fresh document if none exists yet.
    # Pipeline mode: reuse the document already created upstream.
    if args.input_json:
        document = load_document(Path(args.input_json).resolve())
    else:
        document = create_empty_document(
            input_audio_path,
            language_expected=args.language,
        )

    transcribe_audio(
        document=document,
        input_audio_path=input_audio_path,
        output_json_path=output_json_path,
        output_txt_path=(Path(args.output_txt).resolve() if args.output_txt else None),
        model_name=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        beam_size=args.beam_size,
        vad_filter=not args.no_vad_filter,
        vad_min_silence_duration_ms=args.vad_min_silence_duration_ms,
        word_timestamps=not args.no_word_timestamps,
    )


if __name__ == "__main__":
    main()
