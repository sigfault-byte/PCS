from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from pyannote.audio import Inference, Model, Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core import Segment

from assemblybot.helper.artifact import save_npz
from assemblybot.helper.directory import build_default_output_path
from assemblybot.helper.document import load_document, save_document
from assemblybot.models.diarization import (
    CollapsedDiarizationSegment,
    DiarizationArtifacts,
    DiarizationRawSegment,
)
from assemblybot.models.document import CanonicalDocument
from assemblybot.models.factories import (
    create_empty_document,
    mark_stage_completed,
    mark_stage_failed,
    mark_stage_running,
)
from assemblybot.models.time import TimeRange

MIN_EMBEDDING_DURATION_SECONDS = 0.80
SKIP_ULTRA_SHORT_EMBEDDINGS_BELOW_SECONDS = 0.08


def clamp_segment(start: float, end: float, max_end: float) -> tuple[float, float]:
    """Clamp a segment to the valid audio range."""
    start = max(0.0, start)
    end = min(max_end, end)
    if end < start:
        end = start
    return start, end


def expand_segment_to_min_duration(
    start: float,
    end: float,
    min_duration: float,
    max_end: float,
) -> tuple[float, float]:
    """
    Expand a segment symmetrically so embedding extraction has enough audio.

    Very short diarization segments can produce unstable embeddings.
    This keeps the segment centered as much as possible while respecting
    audio boundaries.
    """
    duration = end - start
    if duration >= min_duration:
        return clamp_segment(start, end, max_end)

    deficit = min_duration - duration
    left = deficit / 2.0
    right = deficit - left

    new_start = start - left
    new_end = end + right

    if new_start < 0.0:
        new_end += -new_start
        new_start = 0.0

    if new_end > max_end:
        shift = new_end - max_end
        new_start -= shift
        new_end = max_end

    if new_start < 0.0:
        new_start = 0.0

    return clamp_segment(new_start, new_end, max_end)


def resolve_device(device: str) -> str:
    """Resolve 'auto' to a real torch device string."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_audio_as_waveform(input_audio_path: Path) -> tuple[torch.Tensor, int, float]:
    """
    Load audio and normalize it into the waveform shape expected by pyannote.

    Returns:
        waveform: torch.Tensor shaped (channels, time)
        sample_rate: int
        duration_seconds: float
    """
    audio, sample_rate = sf.read(str(input_audio_path), dtype="float32")

    if audio.ndim == 1:
        waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, time)
        duration_seconds = float(audio.shape[0] / sample_rate)
    else:
        waveform = torch.from_numpy(audio.T)  # (channels, time)
        duration_seconds = float(audio.shape[0] / sample_rate)

    return waveform, sample_rate, duration_seconds


def run_pyannote_diarization(
    waveform: torch.Tensor,
    sample_rate: int,
    hf_token: str,
    model_name: str,
    torch_device: torch.device,
) -> list[DiarizationRawSegment]:
    """
    Run pyannote diarization and convert the result immediately into our own
    typed raw diarization segments.
    """
    print(f"Loading diarization pipeline: {model_name}")
    pipeline = Pipeline.from_pretrained(model_name, token=hf_token)
    pipeline.to(torch_device)  # type: ignore[arg-type]

    with ProgressHook() as hook:
        diarization = pipeline(  # type: ignore[misc]
            {
                "waveform": waveform,
                "sample_rate": sample_rate,
            },
            hook=hook,
        )

    annotation = diarization.speaker_diarization

    raw_segments: list[DiarizationRawSegment] = []
    for idx, (turn, _, speaker_label) in enumerate(
        annotation.itertracks(yield_label=True),
        start=1,
    ):
        raw_segments.append(
            DiarizationRawSegment(
                segment_id=f"dia_{idx:06d}",
                time=TimeRange.from_seconds(turn.start, turn.end),
                speaker_id=speaker_label,
            )
        )

    return raw_segments


def collapse_diarization_segments(
    raw_segments: list[DiarizationRawSegment],
) -> list[CollapsedDiarizationSegment]:
    """
    Merge adjacent diarization segments when they belong to the same speaker.

    This is a convenience view derived from raw diarization, not source truth.
    """
    if not raw_segments:
        return []

    collapsed: list[CollapsedDiarizationSegment] = []

    current = CollapsedDiarizationSegment(
        segment_id="cdia_000001",
        time=TimeRange.from_seconds(
            raw_segments[0].time.start_seconds,
            raw_segments[0].time.end_seconds,
        ),
        speaker_id=raw_segments[0].speaker_id,
        source_diarization_segment_ids=[raw_segments[0].segment_id],
    )

    counter = 1

    for segment in raw_segments[1:]:
        if segment.speaker_id == current.speaker_id:
            current.time = TimeRange.from_seconds(
                current.time.start_seconds,
                segment.time.end_seconds,
            )
            current.source_diarization_segment_ids.append(segment.segment_id)
            continue

        collapsed.append(current)
        counter += 1
        current = CollapsedDiarizationSegment(
            segment_id=f"cdia_{counter:06d}",
            time=TimeRange.from_seconds(
                segment.time.start_seconds,
                segment.time.end_seconds,
            ),
            speaker_id=segment.speaker_id,
            source_diarization_segment_ids=[segment.segment_id],
        )

    collapsed.append(current)
    return collapsed


def apply_diarization_to_document(
    document: CanonicalDocument,
    raw_segments: list[DiarizationRawSegment],
    collapsed_segments: list[CollapsedDiarizationSegment],
    model_name: str,
    device: str,
) -> None:
    """Write diarization results into the typed canonical document."""
    document.diarization.engine.model = model_name
    document.diarization.engine.device = device
    document.diarization.raw_segments = raw_segments
    document.diarization.collapsed_segments = collapsed_segments
    document.diarization.speakers_count = len({seg.speaker_id for seg in raw_segments})


def extract_segment_embeddings(
    raw_segments: list[DiarizationRawSegment],
    waveform: torch.Tensor,
    sample_rate: int,
    audio_duration_seconds: float,
    hf_token: str,
    embedding_model_name: str,
    torch_device: torch.device,
) -> tuple[list[str], list[str], list[np.ndarray], np.ndarray]:
    """
    Extract one embedding per diarization segment.

    Returns:
        segment_ids
        segment_speaker_ids
        segment_embeddings_list
        stacked_embeddings_matrix
    """
    print(f"Loading embedding model: {embedding_model_name}")
    embedding_model = Model.from_pretrained(
        embedding_model_name,
        use_auth_token=hf_token,
    )
    embedding_inference = Inference(embedding_model, window="whole")  # type: ignore[misc]
    embedding_inference.to(torch_device)

    print("Extracting one embedding per diarization segment...")

    audio_source = {
        "waveform": waveform,
        "sample_rate": sample_rate,
    }

    segment_ids: list[str] = []
    segment_speaker_ids: list[str] = []
    segment_embeddings_list: list[np.ndarray] = []

    for segment in raw_segments:
        start = segment.time.start_seconds
        end = segment.time.end_seconds
        duration = end - start

        if duration < SKIP_ULTRA_SHORT_EMBEDDINGS_BELOW_SECONDS:
            print(
                f"Skipping ultra-short segment {segment.segment_id} ({duration:.3f}s)"
            )
            continue

        safe_start, safe_end = expand_segment_to_min_duration(
            start=start,
            end=end,
            min_duration=MIN_EMBEDDING_DURATION_SECONDS,
            max_end=audio_duration_seconds,
        )

        excerpt = Segment(safe_start, safe_end)
        emb = embedding_inference.crop(audio_source, excerpt)
        emb = np.asarray(emb).squeeze().astype(np.float32)

        if emb.ndim != 1:
            raise RuntimeError(
                f"Unexpected embedding shape for {segment.segment_id}: {emb.shape}"
            )

        segment_ids.append(segment.segment_id)
        segment_speaker_ids.append(segment.speaker_id)
        segment_embeddings_list.append(emb)

    if segment_embeddings_list:
        segment_embeddings = np.vstack(segment_embeddings_list).astype(np.float32)
    else:
        segment_embeddings = np.empty((0, 0), dtype=np.float32)

    return (
        segment_ids,
        segment_speaker_ids,
        segment_embeddings_list,
        segment_embeddings,
    )


def compute_speaker_centroids(
    segment_speaker_ids: list[str],
    segment_embeddings_list: list[np.ndarray],
) -> tuple[list[str], np.ndarray]:
    """
    Compute one centroid embedding per speaker from the per-segment embeddings.
    """
    speaker_to_embeddings: dict[str, list[np.ndarray]] = {}
    for speaker_id, emb in zip(segment_speaker_ids, segment_embeddings_list):
        speaker_to_embeddings.setdefault(speaker_id, []).append(emb)

    if not speaker_to_embeddings:
        return [], np.empty((0, 0), dtype=np.float32)

    sorted_speaker_ids = sorted(speaker_to_embeddings.keys())
    speaker_centroids = np.vstack(
        [
            np.mean(np.vstack(speaker_to_embeddings[speaker_id]), axis=0)
            for speaker_id in sorted_speaker_ids
        ]
    ).astype(np.float32)

    return sorted_speaker_ids, speaker_centroids


def save_embedding_artifacts(
    output_segment_embeddings_path: Path,
    output_speaker_centroids_path: Path,
    segment_ids: list[str],
    segment_speaker_ids: list[str],
    segment_embeddings: np.ndarray,
    speaker_ids: list[str],
    speaker_centroids: np.ndarray,
) -> None:
    """Persist NPZ artifacts for later speaker analysis / mapping."""
    output_segment_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    output_speaker_centroids_path.parent.mkdir(parents=True, exist_ok=True)

    save_npz(
        output_segment_embeddings_path,
        segment_ids=np.array(segment_ids, dtype=object),
        segment_speaker_ids=np.array(segment_speaker_ids, dtype=object),
        segment_embeddings=segment_embeddings,
    )

    save_npz(
        output_speaker_centroids_path,
        speaker_ids=np.array(speaker_ids, dtype=object),
        speaker_centroids=speaker_centroids,
    )


def apply_artifacts_to_document(
    document: CanonicalDocument,
    output_segment_embeddings_path: Path,
    output_speaker_centroids_path: Path,
    embedding_model_name: str,
) -> None:
    """Store artifact paths in the typed diarization section."""
    document.diarization.artifacts = DiarizationArtifacts(
        embeddings_npy_path=str(output_segment_embeddings_path),
        centroids_npy_path=str(output_speaker_centroids_path),
        embedding_model=embedding_model_name,
    )


def diarize_audio(
    document: CanonicalDocument,
    input_audio_path: Path,
    output_json_path: Path,
    output_segment_embeddings_path: Path | None = None,
    output_speaker_centroids_path: Path | None = None,
    hf_token: str | None = None,
    model_name: str = "pyannote/speaker-diarization-3.1",
    embedding_model_name: str = "pyannote/embedding",
    device: str = "auto",
) -> CanonicalDocument:
    """
    Main stage orchestrator.

    This function:
    1. validates inputs
    2. marks diarization as running
    3. loads audio
    4. runs pyannote diarization
    5. writes diarization results to the canonical document
    6. extracts per-segment embeddings and speaker centroids
    7. saves JSON + NPZ artifacts
    """
    input_audio_path = input_audio_path.resolve()
    output_json_path = output_json_path.resolve()

    if not input_audio_path.exists():
        raise FileNotFoundError(f"Input audio not found: {input_audio_path}")

    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    if not hf_token:
        raise RuntimeError(
            "Missing Hugging Face token. Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN."
        )

    output_segment_embeddings_path = (
        output_segment_embeddings_path
        or build_default_output_path(
            input_audio_path,
            "_01_segment_embeddings",
            "npz",
        )
    ).resolve()

    output_speaker_centroids_path = (
        output_speaker_centroids_path
        or build_default_output_path(
            input_audio_path,
            "_01_speaker_centroids",
            "npz",
        )
    ).resolve()

    device = resolve_device(device)
    torch_device = torch.device(device)

    mark_stage_running(document, "diarization")

    start_time = time.time()

    try:
        print(f"Using device: {device}")
        print(f"Diarizing: {input_audio_path.name}")

        waveform, sample_rate, duration_seconds = load_audio_as_waveform(
            input_audio_path
        )

        # Stage 1 owns source duration because it reads the media directly.
        document.source.duration_seconds = duration_seconds

        raw_segments = run_pyannote_diarization(
            waveform=waveform,
            sample_rate=sample_rate,
            hf_token=hf_token,
            model_name=model_name,
            torch_device=torch_device,
        )

        # Collapsed diarization is a later derived view.
        # We do not build it here because it requires heuristics.
        collapsed_segments: list[CollapsedDiarizationSegment] = []

        apply_diarization_to_document(
            document=document,
            raw_segments=raw_segments,
            collapsed_segments=collapsed_segments,
            model_name=model_name,
            device=device,
        )

        (
            segment_ids,
            segment_speaker_ids,
            segment_embeddings_list,
            segment_embeddings,
        ) = extract_segment_embeddings(
            raw_segments=raw_segments,
            waveform=waveform,
            sample_rate=sample_rate,
            audio_duration_seconds=duration_seconds,
            hf_token=hf_token,
            embedding_model_name=embedding_model_name,
            torch_device=torch_device,
        )

        speaker_ids, speaker_centroids = compute_speaker_centroids(
            segment_speaker_ids=segment_speaker_ids,
            segment_embeddings_list=segment_embeddings_list,
        )

        save_embedding_artifacts(
            output_segment_embeddings_path=output_segment_embeddings_path,
            output_speaker_centroids_path=output_speaker_centroids_path,
            segment_ids=segment_ids,
            segment_speaker_ids=segment_speaker_ids,
            segment_embeddings=segment_embeddings,
            speaker_ids=speaker_ids,
            speaker_centroids=speaker_centroids,
        )

        apply_artifacts_to_document(
            document=document,
            output_segment_embeddings_path=output_segment_embeddings_path,
            output_speaker_centroids_path=output_speaker_centroids_path,
            embedding_model_name=embedding_model_name,
        )

        mark_stage_completed(
            document,
            "diarization",
            output_path=str(output_json_path),
        )
        save_document(document, output_json_path)

        elapsed = time.time() - start_time

        print(f"Speakers detected: {document.diarization.speakers_count}")
        print(f"Raw diarization segments: {len(document.diarization.raw_segments)}")
        print(
            f"Collapsed diarization segments: {len(document.diarization.collapsed_segments)}"
        )
        print(f"Segment embeddings: {output_segment_embeddings_path}")
        print(f"Speaker centroids: {output_speaker_centroids_path}")
        print(f"Canonical JSON: {output_json_path}")
        print(f"Time: {elapsed / 60:.1f} min")

        return document

    except Exception as exc:
        mark_stage_failed(document, "diarization", str(exc))
        save_document(document, output_json_path)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pyannote diarization and create/update the canonical document."
    )

    parser.add_argument(
        "--input-audio",
        required=True,
        help="Path to input audio file (.wav recommended)",
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
        "--model",
        default="pyannote/speaker-diarization-3.1",
        help="Pyannote diarization model name",
    )

    parser.add_argument(
        "--embedding-model",
        default="pyannote/embedding",
        help="Pyannote embedding model name",
    )

    parser.add_argument(
        "--output-segment-embeddings-npz",
        help="Optional output path for per-segment embeddings",
    )

    parser.add_argument(
        "--output-speaker-centroids-npz",
        help="Optional output path for speaker centroid embeddings",
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cpu, cuda",
    )

    parser.add_argument(
        "--hf-token",
        help="Optional Hugging Face token (otherwise reads HF_TOKEN or HUGGINGFACE_HUB_TOKEN)",
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
            "_01_diarization",
            "json",
        )
    )

    # Standalone mode: create a fresh document if none exists yet.
    # Pipeline mode: reuse the document that was already created upstream.
    if args.input_json:
        document = load_document(Path(args.input_json).resolve())
    else:
        document = create_empty_document(input_audio_path)

    diarize_audio(
        document=document,
        input_audio_path=input_audio_path,
        output_json_path=output_json_path,
        output_segment_embeddings_path=(
            Path(args.output_segment_embeddings_npz).resolve()
            if args.output_segment_embeddings_npz
            else None
        ),
        output_speaker_centroids_path=(
            Path(args.output_speaker_centroids_npz).resolve()
            if args.output_speaker_centroids_npz
            else None
        ),
        hf_token=args.hf_token,
        model_name=args.model,
        embedding_model_name=args.embedding_model,
        device=args.device,
    )


if __name__ == "__main__":
    main()
