from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from pyannote.audio import Inference, Model, Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core import Segment

from assemblybot.config import INTERIM_DIR
from assemblybot.models.diarization import DiarizationRawSegment
from assemblybot.models.time import TimeRange, now_utc_iso

MIN_EMBEDDING_DURATION_SECONDS = 0.80
SKIP_ULTRA_SHORT_EMBEDDINGS_BELOW_SECONDS = 0.08


def clamp_segment(start: float, end: float, max_end: float) -> tuple[float, float]:
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


def build_default_output_path(input_audio_path: Path) -> Path:
    return INTERIM_DIR / f"{input_audio_path.stem}_02_diarization.json"  # type: ignore


def build_default_segment_embeddings_path(input_audio_path: Path) -> Path:
    return INTERIM_DIR / f"{input_audio_path.stem}_02_segment_embeddings.npz"  # type: ignore


def build_default_speaker_centroids_path(input_audio_path: Path) -> Path:
    return INTERIM_DIR / f"{input_audio_path.stem}_02_speaker_centroids.npz"  # type: ignore


def load_document(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_document(doc: dict, json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)


def save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def diarize_audio(
    input_audio_path: Path,
    input_json_path: Path,
    output_json_path: Path | None = None,
    output_segment_embeddings_path: Path | None = None,
    output_speaker_centroids_path: Path | None = None,
    hf_token: str | None = None,
    model_name: str = "pyannote/speaker-diarization-3.1",
    embedding_model_name: str = "pyannote/embedding",
    device: str = "auto",
) -> dict:
    input_audio_path = input_audio_path.resolve()
    input_json_path = input_json_path.resolve()

    if not input_audio_path.exists():
        raise FileNotFoundError(f"Input audio not found: {input_audio_path}")

    if not input_json_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_json_path}")

    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    if not hf_token:
        raise RuntimeError(
            "Missing Hugging Face token. Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN."
        )

    output_json_path = output_json_path or build_default_output_path(input_audio_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    output_segment_embeddings_path = (
        output_segment_embeddings_path
        or build_default_segment_embeddings_path(input_audio_path)
    )
    output_speaker_centroids_path = (
        output_speaker_centroids_path
        or build_default_speaker_centroids_path(input_audio_path)
    )

    device = resolve_device(device)
    torch_device = torch.device(device)

    print(f"Loading diarization pipeline: {model_name}")
    print(f"Using device: {device}")

    start_time = time.time()

    pipeline = Pipeline.from_pretrained(
        model_name,
        token=hf_token,
    )
    pipeline.to(torch_device)  # type: ignore

    print(f"Diarizing: {input_audio_path.name}")
    audio, sample_rate = sf.read(str(input_audio_path), dtype="float32")

    if audio.ndim == 1:
        waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, time)
    else:
        waveform = torch.from_numpy(audio.T)  # (channels, time)

    with ProgressHook() as hook:
        diarization = pipeline(  # type: ignore
            {
                "waveform": waveform,
                "sample_rate": sample_rate,
            },
            hook=hook,
        )

    annotation = diarization.speaker_diarization

    elapsed = time.time() - start_time

    doc = load_document(input_json_path)

    doc["diarization"]["engine"]["model"] = model_name
    doc["diarization"]["engine"]["device"] = device
    doc["diarization"]["raw_segments"] = []

    speaker_ids: set[str] = set()
    diarization_rows: list[dict] = []

    for idx, (turn, _, speaker_label) in enumerate(
        annotation.itertracks(yield_label=True), start=1
    ):
        speaker_ids.add(speaker_label)

        raw_segment = DiarizationRawSegment(
            segment_id=f"dia_{idx:06d}",
            time=TimeRange.from_seconds(turn.start, turn.end),
            speaker_id=speaker_label,
            confidence=None,
        )

        row = {
            "segment_id": raw_segment.segment_id,
            "time": {
                "start_seconds": raw_segment.time.start_seconds,
                "end_seconds": raw_segment.time.end_seconds,
                "duration_seconds": raw_segment.time.duration_seconds,
                "start_ts": raw_segment.time.start_ts,
                "end_ts": raw_segment.time.end_ts,
            },
            "speaker_id": raw_segment.speaker_id,
            "confidence": raw_segment.confidence,
        }

        diarization_rows.append(row)
        doc["diarization"]["raw_segments"].append(row)

    doc["diarization"]["speakers_count"] = len(speaker_ids)
    doc["diarization"]["segments_count"] = len(doc["diarization"]["raw_segments"])

    print(f"Loading embedding model: {embedding_model_name}")
    embedding_model = Model.from_pretrained(
        embedding_model_name,
        use_auth_token=hf_token,
    )
    embedding_inference = Inference(embedding_model, window="whole")
    embedding_inference.to(torch_device)

    segment_ids: list[str] = []
    segment_speaker_ids: list[str] = []
    segment_embeddings_list: list[np.ndarray] = []

    print("Extracting one embedding per diarization segment...")

    audio_source = {
        "waveform": waveform,
        "sample_rate": sample_rate,
    }

    for row in diarization_rows:
        start = row["time"]["start_seconds"]
        end = row["time"]["end_seconds"]
        duration = end - start

        if duration < SKIP_ULTRA_SHORT_EMBEDDINGS_BELOW_SECONDS:
            print(f"Skipping ultra-short segment {row['segment_id']} ({duration:.3f}s)")
            continue

        safe_start, safe_end = expand_segment_to_min_duration(
            start=start,
            end=end,
            min_duration=MIN_EMBEDDING_DURATION_SECONDS,
            max_end=float(doc["source"]["duration_seconds"]),
        )

        excerpt = Segment(safe_start, safe_end)
        emb = embedding_inference.crop(audio_source, excerpt)
        emb = np.asarray(emb).squeeze().astype(np.float32)

        if emb.ndim != 1:
            raise RuntimeError(
                f"Unexpected embedding shape for {row['segment_id']}: {emb.shape}"
            )

        segment_ids.append(row["segment_id"])
        segment_speaker_ids.append(row["speaker_id"])
        segment_embeddings_list.append(emb)

    if segment_embeddings_list:
        segment_embeddings = np.vstack(segment_embeddings_list).astype(np.float32)
    else:
        segment_embeddings = np.empty((0, 0), dtype=np.float32)

    save_npz(
        output_segment_embeddings_path,
        segment_ids=np.array(segment_ids, dtype=object),
        segment_speaker_ids=np.array(segment_speaker_ids, dtype=object),
        segment_embeddings=segment_embeddings,
    )

    speaker_to_embeddings: dict[str, list[np.ndarray]] = {}
    for speaker_id, emb in zip(segment_speaker_ids, segment_embeddings_list):
        speaker_to_embeddings.setdefault(speaker_id, []).append(emb)

    sorted_speaker_ids = sorted(speaker_to_embeddings.keys())
    speaker_centroids = np.vstack(
        [
            np.mean(np.vstack(speaker_to_embeddings[speaker_id]), axis=0)
            for speaker_id in sorted_speaker_ids
        ]
    ).astype(np.float32)

    save_npz(
        output_speaker_centroids_path,
        speaker_ids=np.array(sorted_speaker_ids, dtype=object),
        speaker_centroids=speaker_centroids,
    )

    doc["diarization"]["artifacts"] = {
        "segment_embeddings_npz": str(output_segment_embeddings_path),
        "speaker_centroids_npz": str(output_speaker_centroids_path),
        "embedding_model": embedding_model_name,
    }

    if "diarization" not in doc["pipeline"]["stages_completed"]:
        doc["pipeline"]["stages_completed"].append("diarization")

    doc["pipeline"]["updated_at"] = now_utc_iso()
    doc["pipeline"]["stage_outputs"]["diarization"] = str(output_json_path)

    save_document(doc, output_json_path)

    print(f"Speakers detected: {doc['diarization']['speakers_count']}")
    print(f"Raw diarization segments: {len(doc['diarization']['raw_segments'])}")
    print(f"Segment embeddings: {output_segment_embeddings_path}")
    print(f"Speaker centroids: {output_speaker_centroids_path}")
    print(f"Time: {elapsed / 60:.1f} min")
    print(f"JSON transcript+diarization: {output_json_path}")

    return doc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pyannote diarization and enrich canonical JSON."
    )
    parser.add_argument("--input-audio", required=True, help="Path to input .wav file")
    parser.add_argument(
        "--input-json",
        required=True,
        help="Path to existing transcript JSON",
    )
    parser.add_argument(
        "--output-json",
        help="Optional output JSON path",
    )
    parser.add_argument(
        "--model",
        default="pyannote/speaker-diarization-3.1",
        help="Pyannote model name",
    )
    parser.add_argument(
        "--embedding-model",
        default="pyannote/embedding",
        help="Pyannote embedding model name",
    )
    parser.add_argument(
        "--output-segment-embeddings-npz",
        help="Optional output path for diarization segment embeddings",
    )
    parser.add_argument(
        "--output-speaker-centroids-npz",
        help="Optional output path for speaker centroid embeddings",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: auto, cpu, cuda",
    )
    parser.add_argument(
        "--hf-token",
        help="Optional Hugging Face token (otherwise reads HF_TOKEN or HUGGINGFACE_HUB_TOKEN)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    diarize_audio(
        input_audio_path=Path(args.input_audio),
        input_json_path=Path(args.input_json),
        output_json_path=Path(args.output_json) if args.output_json else None,
        output_segment_embeddings_path=(
            Path(args.output_segment_embeddings_npz)
            if args.output_segment_embeddings_npz
            else None
        ),
        output_speaker_centroids_path=(
            Path(args.output_speaker_centroids_npz)
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
