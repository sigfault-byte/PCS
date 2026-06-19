from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from assemblybot.helper.artifact import save_npz
from assemblybot.models.diarization import DiarizationArtifacts, DiarizationRawSegment
from assemblybot.models.document import CanonicalDocument
from assemblybot.pyannote_config import PyannoteDiarizationConfig


CROP_BOUNDARY_EPSILON_SECONDS = 1e-6


def clamp_segment(start: float, end: float, max_end: float) -> tuple[float, float]:
    """Clamp a segment to the valid audio range."""
    safe_max_end = max_end
    if max_end > CROP_BOUNDARY_EPSILON_SECONDS:
        safe_max_end = max_end - CROP_BOUNDARY_EPSILON_SECONDS

    start = min(max(0.0, start), safe_max_end)
    end = min(safe_max_end, end)
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


def extract_segment_embeddings(
    raw_segments: list[DiarizationRawSegment],
    waveform: torch.Tensor,
    sample_rate: int,
    audio_duration_seconds: float,
    hf_token: str,
    torch_device: torch.device,
    config: PyannoteDiarizationConfig,
) -> tuple[list[int], list[str], list[np.ndarray], np.ndarray]:
    """
    Extract one embedding per diarization segment.

    Returns:
        segment_ids
        segment_speaker_ids
        segment_embeddings_list
        stacked_embeddings_matrix
    """
    print(f"Loading embedding model: {config.embedding_model_name}")
    from pyannote.audio import Inference, Model
    from pyannote.core import Segment

    embedding_model = Model.from_pretrained(
        config.embedding_model_name,
        use_auth_token=hf_token,
    )
    embedding_inference = Inference(embedding_model, window="whole")  # type: ignore[misc]
    embedding_inference.to(torch_device)

    print("Extracting one embedding per diarization segment...")

    audio_source = {
        "waveform": waveform,
        "sample_rate": sample_rate,
    }

    segment_ids: list[int] = []
    segment_speaker_ids: list[str] = []
    segment_embeddings_list: list[np.ndarray] = []

    for segment in raw_segments:
        start = segment.time.start_seconds
        end = segment.time.end_seconds
        duration = end - start

        if duration < config.skip_embeddings_below_seconds:
            print(
                f"Skipping ultra-short segment {segment.segment_id} ({duration:.3f}s)"
            )
            continue

        safe_start, safe_end = expand_segment_to_min_duration(
            start=start,
            end=end,
            min_duration=config.min_embedding_duration_seconds,
            max_end=audio_duration_seconds,
        )

        excerpt = Segment(safe_start, safe_end)
        emb = embedding_inference.crop(audio_source, excerpt)
        emb = np.asarray(emb).squeeze().astype(config.embedding_dtype)

        if emb.ndim != 1:
            raise RuntimeError(
                f"Unexpected embedding shape for {segment.segment_id}: {emb.shape}"
            )

        segment_ids.append(segment.segment_id)
        segment_speaker_ids.append(segment.speaker_id)
        segment_embeddings_list.append(emb)

    if segment_embeddings_list:
        segment_embeddings = np.vstack(segment_embeddings_list).astype(
            config.embedding_dtype
        )
    else:
        segment_embeddings = np.empty((0, 0), dtype=config.embedding_dtype)

    return (
        segment_ids,
        segment_speaker_ids,
        segment_embeddings_list,
        segment_embeddings,
    )


def compute_speaker_centroids(
    segment_speaker_ids: list[str],
    segment_embeddings_list: list[np.ndarray],
    config: PyannoteDiarizationConfig,
) -> tuple[list[str], np.ndarray]:
    """Compute one centroid embedding per speaker."""
    speaker_to_embeddings: dict[str, list[np.ndarray]] = {}
    for speaker_id, emb in zip(segment_speaker_ids, segment_embeddings_list):
        speaker_to_embeddings.setdefault(speaker_id, []).append(emb)

    if not speaker_to_embeddings:
        return [], np.empty((0, 0), dtype=config.embedding_dtype)

    sorted_speaker_ids = sorted(speaker_to_embeddings.keys())
    speaker_centroids = np.vstack(
        [
            np.mean(np.vstack(speaker_to_embeddings[speaker_id]), axis=0)
            for speaker_id in sorted_speaker_ids
        ]
    ).astype(config.embedding_dtype)

    return sorted_speaker_ids, speaker_centroids


def save_embedding_artifacts(
    output_segment_embeddings_path: Path,
    output_speaker_centroids_path: Path,
    segment_ids: list[int],
    segment_speaker_ids: list[str],
    segment_embeddings: np.ndarray,
    speaker_ids: list[str],
    speaker_centroids: np.ndarray,
) -> None:
    """Persist compressed NPZ artifacts for later speaker analysis."""
    output_segment_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    output_speaker_centroids_path.parent.mkdir(parents=True, exist_ok=True)

    save_npz(
        output_segment_embeddings_path,
        segment_ids=np.array(segment_ids, dtype=np.int64),
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
    """Store embedding artifact paths in the typed diarization section."""
    document.diarization.artifacts = DiarizationArtifacts(
        embeddings_npz_path=str(output_segment_embeddings_path),
        centroids_npz_path=str(output_speaker_centroids_path),
        embedding_model=embedding_model_name,
    )


def save_diarization_embedding_artifacts(
    document: CanonicalDocument,
    raw_segments: list[DiarizationRawSegment],
    waveform: torch.Tensor,
    sample_rate: int,
    audio_duration_seconds: float,
    hf_token: str,
    torch_device: torch.device,
    output_segment_embeddings_path: Path,
    output_speaker_centroids_path: Path,
    config: PyannoteDiarizationConfig,
) -> None:
    """Extract embeddings, compute centroids, save NPZ files, and update metadata."""
    (
        segment_ids,
        segment_speaker_ids,
        segment_embeddings_list,
        segment_embeddings,
    ) = extract_segment_embeddings(
        raw_segments=raw_segments,
        waveform=waveform,
        sample_rate=sample_rate,
        audio_duration_seconds=audio_duration_seconds,
        hf_token=hf_token,
        torch_device=torch_device,
        config=config,
    )

    speaker_ids, speaker_centroids = compute_speaker_centroids(
        segment_speaker_ids=segment_speaker_ids,
        segment_embeddings_list=segment_embeddings_list,
        config=config,
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
        embedding_model_name=config.embedding_model_name,
    )
