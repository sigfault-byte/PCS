from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class TurnsConfig:
    """Policy settings for consolidating aligned segments into turns."""

    max_turn_silence_seconds: float = 5.0

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TurnsConfig":
        return cls()


DEFAULT_TURNS_CONFIG = TurnsConfig()


def add_turns_arguments(parser: argparse.ArgumentParser) -> None:
    """Register turn-consolidation-specific CLI options on a stage parser."""
