from __future__ import annotations

import argparse
from dataclasses import dataclass

from assemblybot.models.flags import SegmentFlag


@dataclass(frozen=True)
class AlignmentConfig:
    """Policy settings for transcript-to-diarization alignment."""

    anomaly_flags_to_propagate: tuple[SegmentFlag, ...] = (
        SegmentFlag.IMPOSSIBLE_SPEECH_RATE,
        SegmentFlag.INFORMATION_RATE_TOO_HIGH,
        SegmentFlag.MOSTLY_SILENCE_WITH_SHORT_EVENT,
    )
    speaker_evidence_longest_overlap_weight: float = 0.5
    speaker_evidence_extra_segment_penalty: float = 0.1

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "AlignmentConfig":
        return cls()


DEFAULT_ALIGNMENT_CONFIG = AlignmentConfig()


def add_alignment_arguments(parser: argparse.ArgumentParser) -> None:
    """Register alignment-specific CLI options on a stage parser."""
