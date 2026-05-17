from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

from assemblybot.helper.directory import build_default_output_path
from assemblybot.helper.document import load_document, save_document
from assemblybot.models.diarization import (
    CollapsedDiarizationSegment,
    DiarizationArtifacts,
    DiarizationOverlapRegion,
    DiarizationRawSegment,
)
from assemblybot.models.document import CanonicalDocument
from assemblybot.models.factories import (
    create_empty_document,
    mark_stage_completed,
    mark_stage_failed,
    mark_stage_running,
)
from assemblybot.models.flags import SegmentFlag
from assemblybot.models.time import TimeRange

if TYPE_CHECKING:
    import torch


def resolve_device(device: str) -> str:
    """Resolve 'auto' to a real torch device string."""
    import torch

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
    import soundfile as sf
    import torch

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
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.utils.hook import ProgressHook

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
                segment_id=idx,
                time=TimeRange.from_seconds(turn.start, turn.end),
                speaker_id=speaker_label,
            )
        )

    return raw_segments


def compute_overlap_regions(
    raw_segments: list[DiarizationRawSegment],
) -> list[DiarizationOverlapRegion]:
    """Find speaker overlap intervals without changing raw segment timings."""
    events_by_time: dict[float, list[tuple[str, str]]] = {}
    for segment in raw_segments:
        events_by_time.setdefault(segment.time.start_seconds, []).append(
            ("start", segment.speaker_id)
        )
        events_by_time.setdefault(segment.time.end_seconds, []).append(
            ("end", segment.speaker_id)
        )

    active_counts: dict[str, int] = {}
    overlap_regions: list[DiarizationOverlapRegion] = []
    previous_time: float | None = None

    for current_time in sorted(events_by_time):
        active_speaker_ids = sorted(
            speaker_id
            for speaker_id, count in active_counts.items()
            if count > 0
        )

        if (
            previous_time is not None
            and current_time > previous_time
            and len(active_speaker_ids) >= 2
        ):
            overlap_regions.append(
                DiarizationOverlapRegion(
                    region_id=len(overlap_regions) + 1,
                    time=TimeRange.from_seconds(previous_time, current_time),
                    speaker_ids=active_speaker_ids,
                )
            )

        for event_type, speaker_id in events_by_time[current_time]:
            if event_type == "end":
                active_counts[speaker_id] = active_counts.get(speaker_id, 0) - 1
                if active_counts[speaker_id] <= 0:
                    del active_counts[speaker_id]

        for event_type, speaker_id in events_by_time[current_time]:
            if event_type == "start":
                active_counts[speaker_id] = active_counts.get(speaker_id, 0) + 1

        previous_time = current_time

    return overlap_regions


def annotate_raw_segment_overlaps(
    raw_segments: list[DiarizationRawSegment],
    overlap_regions: list[DiarizationOverlapRegion],
) -> None:
    """Annotate raw segments that intersect overlap regions."""
    for segment in raw_segments:
        overlap_speaker_ids: set[str] = set(segment.overlap_speaker_ids)

        for region in overlap_regions:
            intersects = (
                segment.time.start_seconds < region.time.end_seconds
                and region.time.start_seconds < segment.time.end_seconds
            )
            if not intersects:
                continue

            overlap_speaker_ids.update(
                speaker_id
                for speaker_id in region.speaker_ids
                if speaker_id != segment.speaker_id
            )

            segment.flags |= SegmentFlag.DIARIZATION_OVERLAP
            if len(region.speaker_ids) >= 2:
                segment.flags |= SegmentFlag.MULTI_SPEAKER_CANDIDATE

        segment.overlap_speaker_ids = sorted(overlap_speaker_ids)


def apply_diarization_to_document(
    document: CanonicalDocument,
    raw_segments: list[DiarizationRawSegment],
    overlap_regions: list[DiarizationOverlapRegion],
    collapsed_segments: list[CollapsedDiarizationSegment],
    model_name: str,
    device: str,
) -> None:
    """Write diarization results into the typed canonical document."""
    document.diarization.engine.model = model_name
    document.diarization.engine.device = device
    document.diarization.raw_segments = raw_segments
    document.diarization.overlap_regions = overlap_regions
    document.diarization.collapsed_segments = collapsed_segments
    document.diarization.speakers_count = len({seg.speaker_id for seg in raw_segments})


def diarize_audio(
    document: CanonicalDocument,
    input_audio_path: Path,
    output_json_path: Path,
    output_segment_embeddings_path: Path | None = None,
    output_speaker_centroids_path: Path | None = None,
    hf_token: str | None = None,
    model_name: str = "pyannote/speaker-diarization-3.1",
    embedding_model_name: str = "pyannote/embedding",
    extract_embeddings: bool = True,
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

    if extract_embeddings:
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
    import torch

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
        overlap_regions = compute_overlap_regions(raw_segments)
        annotate_raw_segment_overlaps(raw_segments, overlap_regions)

        # Collapsed diarization is a later derived view.
        # We do not build it here because it requires heuristics.
        collapsed_segments: list[CollapsedDiarizationSegment] = []

        apply_diarization_to_document(
            document=document,
            raw_segments=raw_segments,
            overlap_regions=overlap_regions,
            collapsed_segments=collapsed_segments,
            model_name=model_name,
            device=device,
        )

        if extract_embeddings:
            if output_segment_embeddings_path is None:
                raise RuntimeError("Missing segment embeddings output path.")
            if output_speaker_centroids_path is None:
                raise RuntimeError("Missing speaker centroids output path.")

            from assemblybot.stages.diarization_embeddings import (
                save_diarization_embedding_artifacts,
            )

            save_diarization_embedding_artifacts(
                document=document,
                raw_segments=raw_segments,
                waveform=waveform,
                sample_rate=sample_rate,
                audio_duration_seconds=duration_seconds,
                hf_token=hf_token,
                embedding_model_name=embedding_model_name,
                torch_device=torch_device,
                output_segment_embeddings_path=output_segment_embeddings_path,
                output_speaker_centroids_path=output_speaker_centroids_path,
            )
        else:
            document.diarization.artifacts = DiarizationArtifacts()

        mark_stage_completed(
            document,
            "diarization",
            output_path=str(output_json_path),
        )
        save_document(document, output_json_path)

        elapsed = time.time() - start_time

        print(f"Speakers detected: {document.diarization.speakers_count}")
        print(f"Raw diarization segments: {len(document.diarization.raw_segments)}")
        print(f"Overlap regions: {len(document.diarization.overlap_regions)}")
        print(
            f"Collapsed diarization segments: {len(document.diarization.collapsed_segments)}"
        )
        if extract_embeddings:
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
        "--no-embeddings",
        action="store_true",
        help="Skip embedding extraction, centroid computation, and NPZ artifacts",
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
        extract_embeddings=not args.no_embeddings,
        device=args.device,
    )


if __name__ == "__main__":
    main()
