from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parent / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

# Ground truth names
from assemblybot.orchestration.paths import (
    DEFAULT_DEPUTIES_CSV,
    DEFAULT_MINISTERS_CSV,
)
from assemblybot.orchestration.queue import discover_candidates
from assemblybot.orchestration.runner import PipelineRunner


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the AssemblyBot pipeline for files in data/audio/unprocessed.",
    )
    parser.add_argument(
        "--file",
        action="append",
        dest="files",
        help="Specific file name or path to process. Can be passed multiple times.",
    )
    parser.add_argument("--limit", type=int, help="Limit the number of candidates.")
    parser.add_argument("--language", default="fr")
    parser.add_argument(
        "--deputies-ground-truth-csv",
        default=str(DEFAULT_DEPUTIES_CSV),
    )
    parser.add_argument(
        "--ministers-ground-truth-csv",
        default=str(DEFAULT_MINISTERS_CSV),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce orchestration chatter while preserving stage output.",
    )
    parser.add_argument(
        "--replace-sqlite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replace the per-run SQLite file if it exists.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    candidates = discover_candidates(files=args.files, limit=args.limit)
    runner = PipelineRunner(
        language=args.language,
        deputies_ground_truth_csv=Path(args.deputies_ground_truth_csv).resolve(),
        ministers_ground_truth_csv=Path(args.ministers_ground_truth_csv).resolve(),
        replace_sqlite=args.replace_sqlite,
        quiet=args.quiet,
    )

    runner.announce(f"Discovered {len(candidates)} candidate file(s).")
    failed_count = 0
    for candidate in candidates:
        if not candidate.exists():
            runner.announce(f"Missing candidate, skipping: {candidate}")
            failed_count += 1
            continue
        runner.announce(f"Starting candidate: {candidate.name}")
        manifest = runner.run_candidate(candidate)
        if manifest["status"] == "failed":
            failed_count += 1
        runner.announce(f"Finished candidate: {candidate.name} -> {manifest['status']}")

    runner.announce(
        f"Pipeline batch complete: {len(candidates) - failed_count} ok/skipped, "
        f"{failed_count} failed."
    )
    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
