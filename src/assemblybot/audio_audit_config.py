from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AudioAuditConfig:
    """Runtime settings for the librosa audio audit artifact."""

    target_sample_rate: int = 16_000
    frame_length: int = 4096
    hop_length: int = 1600
    rolling_median_window_frames: int = 21
    feature_chunk_frames: int = 500
    spectral_amin: float = 1e-10
    feature_dtype: type[np.floating] = np.float32

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "AudioAuditConfig":
        return cls(
            target_sample_rate=args.target_sample_rate,
            frame_length=args.frame_length,
            hop_length=args.hop_length,
            rolling_median_window_frames=args.rolling_median_window_frames,
            feature_chunk_frames=args.feature_chunk_frames,
        )


DEFAULT_AUDIO_AUDIT_CONFIG = AudioAuditConfig()


def add_audio_audit_arguments(parser: argparse.ArgumentParser) -> None:
    """Register audio-audit-specific CLI options on a stage parser."""
    parser.add_argument(
        "--target-sample-rate",
        type=int,
        default=DEFAULT_AUDIO_AUDIT_CONFIG.target_sample_rate,
        help="Target sample rate used for the audio audit.",
    )

    parser.add_argument(
        "--frame-length",
        type=int,
        default=DEFAULT_AUDIO_AUDIT_CONFIG.frame_length,
        help="Analysis frame length in samples.",
    )

    parser.add_argument(
        "--hop-length",
        type=int,
        default=DEFAULT_AUDIO_AUDIT_CONFIG.hop_length,
        help="Hop length between analysis frames in samples.",
    )

    parser.add_argument(
        "--rolling-median-window-frames",
        type=int,
        default=DEFAULT_AUDIO_AUDIT_CONFIG.rolling_median_window_frames,
        help="Centered rolling median window size in frames.",
    )

    parser.add_argument(
        "--feature-chunk-frames",
        type=int,
        default=DEFAULT_AUDIO_AUDIT_CONFIG.feature_chunk_frames,
        help="Number of frames processed per feature chunk.",
    )
