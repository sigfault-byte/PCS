import argparse
import json
from pathlib import Path

import librosa
import numpy as np

from assemblybot.config import AUDIO_AUDIT_DIR
from assemblybot.helper.directory import build_default_output_path
from assemblybot.models.audit import (
    AudioAuditBuildResult,
    AudioAuditParameters,
    AudioAuditSource,
    AudioAuditSummary,
    FeatureSummary,
)

TARGET_SAMPLE_RATE = 16_000
FRAME_LENGTH = 4096
HOP_LENGTH = 1600
ROLLING_MEDIAN_WINDOW_FRAMES = 21
FEATURE_CHUNK_FRAMES = 500  # lower ram consumtion
SPECTRAL_AMIN = 1e-10
FEATURE_DTYPE = np.float32


def finite_float(value: float, label: str) -> float:
    output = float(value)
    if not np.isfinite(output):
        raise ValueError(f"Non-finite value for {label}: {value!r}")
    return output


def validate_feature_arrays(features: dict[str, np.ndarray]) -> int:
    lengths = {name: len(values) for name, values in features.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Feature arrays have different lengths: {lengths}")

    for name, values in features.items():
        if not np.all(np.isfinite(values)):
            bad_indexes = np.where(~np.isfinite(values))[0][:10].tolist()
            raise ValueError(
                f"Feature '{name}' contains non-finite values at frame indexes {bad_indexes}"
            )

    return next(iter(lengths.values()))


def centered_rolling_median(values: np.ndarray, window_frames: int) -> np.ndarray:
    if window_frames < 1:
        raise ValueError("Rolling median window must be at least 1 frame")

    half_window = window_frames // 2
    medians = np.empty_like(values, dtype=FEATURE_DTYPE)

    for index in range(len(values)):
        start = max(0, index - half_window)
        end = min(len(values), index + half_window + 1)
        medians[index] = FEATURE_DTYPE(np.median(values[start:end]))

    return medians


def frame_count(sample_count: int) -> int:
    if sample_count < FRAME_LENGTH:
        return 0
    return 1 + (sample_count - FRAME_LENGTH) // HOP_LENGTH


def compute_chunk_features(
    y: np.ndarray,
    sample_rate: int | float,
    rows: int,
) -> dict[str, np.ndarray]:
    features = {
        "rms": np.empty(rows, dtype=FEATURE_DTYPE),
        "zcr": np.empty(rows, dtype=FEATURE_DTYPE),
        "spectral_centroid": np.empty(rows, dtype=FEATURE_DTYPE),
        "spectral_bandwidth": np.empty(rows, dtype=FEATURE_DTYPE),
        "spectral_flatness": np.empty(rows, dtype=FEATURE_DTYPE),
    }
    frequencies = np.fft.rfftfreq(FRAME_LENGTH, d=1.0 / sample_rate).astype(
        FEATURE_DTYPE
    )[:, np.newaxis]
    window = np.hanning(FRAME_LENGTH).astype(FEATURE_DTYPE, copy=False)[:, np.newaxis]
    total_chunks = (rows + FEATURE_CHUNK_FRAMES - 1) // FEATURE_CHUNK_FRAMES

    print(
        "Computing per-frame acoustic features "
        f"({rows} frames in {total_chunks} chunks)...",
        flush=True,
    )
    for chunk_index, frame_start in enumerate(range(0, rows, FEATURE_CHUNK_FRAMES), 1):
        frame_end = min(rows, frame_start + FEATURE_CHUNK_FRAMES)
        first_sample = frame_start * HOP_LENGTH
        last_sample = (frame_end - 1) * HOP_LENGTH + FRAME_LENGTH
        y_chunk = y[first_sample:last_sample].astype(FEATURE_DTYPE, copy=False)
        frames = librosa.util.frame(
            y_chunk,
            frame_length=FRAME_LENGTH,
            hop_length=HOP_LENGTH,
        )

        chunk_slice = slice(frame_start, frame_end)
        power_time = np.square(frames, dtype=FEATURE_DTYPE)
        features["rms"][chunk_slice] = np.sqrt(np.mean(power_time, axis=0))
        features["zcr"][chunk_slice] = (
            np.sum(np.diff(np.signbit(frames), axis=0), axis=0) / FRAME_LENGTH
        )

        # Compute spectral metrics from this bounded frame chunk only. This
        # intentionally avoids a full-file spectrogram, which is unnecessary
        # for the atomic per-frame audit and can consume very large RAM.
        magnitude = np.abs(np.fft.rfft(frames * window, axis=0)).astype(
            FEATURE_DTYPE,
            copy=False,
        )
        magnitude_sum = np.maximum(np.sum(magnitude, axis=0), SPECTRAL_AMIN)
        centroid = np.sum(frequencies * magnitude, axis=0) / magnitude_sum
        features["spectral_centroid"][chunk_slice] = centroid
        features["spectral_bandwidth"][chunk_slice] = np.sqrt(
            np.sum(np.square(frequencies - centroid) * magnitude, axis=0)
            / magnitude_sum
        )

        power_spectrum = np.maximum(
            np.square(magnitude, dtype=FEATURE_DTYPE), SPECTRAL_AMIN
        )
        features["spectral_flatness"][chunk_slice] = np.exp(
            np.mean(np.log(power_spectrum), axis=0)
        ) / np.mean(power_spectrum, axis=0)

        if chunk_index == 1 or chunk_index == total_chunks or chunk_index % 25 == 0:
            print(
                f"  processed chunk {chunk_index}/{total_chunks} "
                f"through frame {frame_end}",
                flush=True,
            )

    return features


def summarize_feature(values: np.ndarray) -> FeatureSummary:
    if len(values) == 0:
        raise ValueError("Cannot summarize an empty feature array")

    percentiles = np.percentile(values, [5, 10, 25, 50, 75, 90, 95, 99])
    return FeatureSummary(
        mean=finite_float(np.mean(values), "summary.mean"),  # type: ignore
        std=finite_float(np.std(values), "summary.std"),  # type: ignore
        min=finite_float(np.min(values), "summary.min"),
        p05=finite_float(percentiles[0], "summary.p05"),
        p10=finite_float(percentiles[1], "summary.p10"),
        p25=finite_float(percentiles[2], "summary.p25"),
        p50=finite_float(percentiles[3], "summary.p50"),
        p75=finite_float(percentiles[4], "summary.p75"),
        p90=finite_float(percentiles[5], "summary.p90"),
        p95=finite_float(percentiles[6], "summary.p95"),
        p99=finite_float(percentiles[7], "summary.p99"),
        max=finite_float(np.max(values), "summary.max"),
    )


def frame_record_at_index(
    features: dict[str, np.ndarray],
    frame_index: int,
    sample_rate: int | float,
) -> dict[str, float | int]:
    frame_duration_seconds = FRAME_LENGTH / sample_rate
    frame_start_seconds = frame_index * HOP_LENGTH / sample_rate

    # The timestamp is the frame center because each row describes the whole
    # analysis window, not just the instant where that window starts.
    # Apparently this is a convention, but the json keeps all 4 values.
    frame_center_seconds = frame_start_seconds + frame_duration_seconds / 2
    frame_end_seconds = frame_start_seconds + frame_duration_seconds

    return {
        "frame_index": frame_index,
        "frame_start_seconds": finite_float(
            frame_start_seconds, f"frames[{frame_index}].frame_start_seconds"
        ),
        "frame_center_seconds": finite_float(
            frame_center_seconds, f"frames[{frame_index}].frame_center_seconds"
        ),
        "frame_end_seconds": finite_float(
            frame_end_seconds, f"frames[{frame_index}].frame_end_seconds"
        ),
        "time_seconds": finite_float(
            frame_center_seconds, f"frames[{frame_index}].time_seconds"
        ),
        "rms": finite_float(features["rms"][frame_index], f"frames[{frame_index}].rms"),
        "db": finite_float(features["db"][frame_index], f"frames[{frame_index}].db"),
        "zcr": finite_float(features["zcr"][frame_index], f"frames[{frame_index}].zcr"),
        "spectral_centroid": finite_float(
            features["spectral_centroid"][frame_index],
            f"frames[{frame_index}].spectral_centroid",
        ),
        "spectral_bandwidth": finite_float(
            features["spectral_bandwidth"][frame_index],
            f"frames[{frame_index}].spectral_bandwidth",
        ),
        "spectral_flatness": finite_float(
            features["spectral_flatness"][frame_index],
            f"frames[{frame_index}].spectral_flatness",
        ),
        "db_rolling_median": finite_float(
            features["db_rolling_median"][frame_index],
            f"frames[{frame_index}].db_rolling_median",
        ),
        "db_delta": finite_float(
            features["db_delta"][frame_index], f"frames[{frame_index}].db_delta"
        ),
    }


def build_audio_audit(input_audio_path: Path) -> AudioAuditBuildResult:
    print(f"Loading audio: {input_audio_path}", flush=True)

    # The audit always targets 16 kHz. That keeps frame timing predictable for
    # speech-oriented downstream stages and avoids mixing source sample rates.
    y, sample_rate = librosa.load(input_audio_path, sr=TARGET_SAMPLE_RATE, mono=True)
    duration_seconds = finite_float(len(y) / sample_rate, "source.duration_seconds")
    print(
        f"Loaded {duration_seconds:.2f} seconds at {sample_rate} Hz "
        f"({len(y)} samples).",
        flush=True,
    )

    # frame_length is the number of samples each measurement window can see.
    # At 16 kHz, 4096 samples is 0.256 seconds of audio.
    frame_duration_seconds = FRAME_LENGTH / sample_rate

    # hop_length is the number of samples between neighboring measurements.
    # At 16 kHz, 1600 samples is 0.100 seconds.
    hop_duration_seconds = HOP_LENGTH / sample_rate

    # These are overlapping windows because hop_length is smaller than
    # frame_length. This gives dense local measurements while preserving the
    # raw frame-level evidence for later pipeline stages.
    rows = frame_count(len(y))
    if rows == 0:
        raise ValueError(
            f"Audio is too short for one frame: {len(y)} samples, "
            f"frame_length={FRAME_LENGTH}"
        )

    features = compute_chunk_features(y, sample_rate, rows)
    print("Computing dB, rolling median, and local dB delta...", flush=True)
    rms = features["rms"]

    # dB is relative to full-scale amplitude, not the loudest frame in this
    # file. This is closer to dBFS and more stable across files, so values will
    # not necessarily have 0 dB as the loudest frame.
    db = librosa.amplitude_to_db(rms, ref=1.0).astype(FEATURE_DTYPE, copy=False)
    db_rolling_median = centered_rolling_median(db, ROLLING_MEDIAN_WINDOW_FRAMES)
    db_delta = db - db_rolling_median.astype(FEATURE_DTYPE, copy=False)

    features = {
        "rms": features["rms"],
        "db": db,
        "zcr": features["zcr"],
        "spectral_centroid": features["spectral_centroid"],
        "spectral_bandwidth": features["spectral_bandwidth"],
        "spectral_flatness": features["spectral_flatness"],
        "db_rolling_median": db_rolling_median,
        "db_delta": db_delta,
    }
    rows = validate_feature_arrays(features)

    source = AudioAuditSource(
        audio_path=str(input_audio_path),
        audio_filename=input_audio_path.name,
        duration_seconds=duration_seconds,
    )
    parameters = AudioAuditParameters(
        target_sample_rate=TARGET_SAMPLE_RATE,
        sample_rate=sample_rate,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
        frame_duration_seconds=finite_float(
            frame_duration_seconds, "parameters.frame_duration_seconds"
        ),
        hop_duration_seconds=finite_float(
            hop_duration_seconds, "parameters.hop_duration_seconds"
        ),
        time_reference="frame_center",
    )

    print("Computing summary from per-frame feature arrays...", flush=True)
    summary = AudioAuditSummary(
        # This JSON is the atomic full-audio audit artifact. Later stages should
        # summarize and flag Whisper segments from it while the raw frame metrics
        # remain external, self-contained, and traceable.
        rows=rows,
        features={name: summarize_feature(values) for name, values in features.items()},
    )

    return AudioAuditBuildResult(
        source=source,
        parameters=parameters,
        summary=summary,
        features=features,
        sample_rate=sample_rate,
        rows=rows,
    )


def json_with_nested_indent(value: object, base_indent: str) -> str:
    return json.dumps(value, indent=2).replace("\n", "\n" + base_indent)


def write_audio_audit_json_streaming(
    audit: AudioAuditBuildResult,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Validating feature arrays before streaming frames...", flush=True)
    rows = validate_feature_arrays(audit.features)
    if rows != audit.rows:
        raise ValueError(f"Expected {audit.rows} rows but validated {rows} rows")

    print(f"Writing JSON artifact: {output_path}", flush=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        output_file.write("{\n")
        output_file.write('  "schema_version": ')
        json.dump("audio_audit.v1", output_file)
        output_file.write(",\n")

        output_file.write('  "source": ')
        output_file.write(json_with_nested_indent(audit.source.to_dict(), "  "))
        output_file.write(",\n")

        output_file.write('  "parameters": ')
        output_file.write(json_with_nested_indent(audit.parameters.to_dict(), "  "))
        output_file.write(",\n")

        output_file.write('  "summary": ')
        output_file.write(json_with_nested_indent(audit.summary.to_dict(), "  "))
        output_file.write(",\n")

        output_file.write('  "frames": [\n')
        print("Streaming JSON frame records...", flush=True)
        for frame_index in range(audit.rows):
            if frame_index > 0:
                output_file.write(",\n")
            frame_record = frame_record_at_index(
                audit.features,
                frame_index,
                audit.sample_rate,
            )
            output_file.write("    ")
            output_file.write(json_with_nested_indent(frame_record, "    "))
            if frame_index == 0 or frame_index + 1 == audit.rows:
                print(
                    f"  wrote frame {frame_index + 1}/{audit.rows}",
                    flush=True,
                )
            elif (frame_index + 1) % 25000 == 0:
                print(
                    f"  wrote frame {frame_index + 1}/{audit.rows}",
                    flush=True,
                )

        output_file.write("\n  ]\n")
        output_file.write("}\n")


def write_audio_audit(input_audio_path: Path, output_path: Path) -> int:
    audit = build_audio_audit(input_audio_path)
    write_audio_audit_json_streaming(audit, output_path)

    return audit.rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one self-contained librosa audio audit JSON artifact."
    )

    parser.add_argument(
        "--input-audio",
        help="Path to the input audio file.",
    )
    parser.add_argument(
        "--output",
        help="Optional output JSON path. Defaults to the configured audio audit directory.",
    )
    args = parser.parse_args()
    if not args.input_audio:
        parser.error("an input audio path is required; use --input-audio PATH")
    return args


def main() -> None:
    args = parse_args()
    input_audio_path = Path(args.input_audio).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else build_default_output_path(
            input_audio_path,
            "_audio_audit",
            "json",
            AUDIO_AUDIT_DIR,
        )
    )

    frame_count = write_audio_audit(input_audio_path, output_path)
    print(f"Saved audio audit JSON: {output_path}")
    print(f"Frame count: {frame_count}")


if __name__ == "__main__":
    main()
