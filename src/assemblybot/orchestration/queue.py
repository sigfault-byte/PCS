from __future__ import annotations

import shutil
from pathlib import Path

from assemblybot.orchestration import paths as orchestration_paths


def ensure_queue_directories() -> None:
    for directory in (
        orchestration_paths.UNPROCESSED_DIR,
        orchestration_paths.PROCESSING_DIR,
        orchestration_paths.PROCESSED_DIR,
        orchestration_paths.FAILED_DIR,
        orchestration_paths.RUNS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def is_wav_file(path: Path) -> bool:
    return path.suffix.lower() == ".wav"


def discover_candidates(
    *,
    unprocessed_dir: Path | None = None,
    files: list[str] | None = None,
    limit: int | None = None,
) -> list[Path]:
    unprocessed_dir = unprocessed_dir or orchestration_paths.UNPROCESSED_DIR
    ensure_queue_directories()
    if files:
        candidates = []
        for file_value in files:
            path = Path(file_value).expanduser()
            if not path.is_absolute():
                path = unprocessed_dir / path
            candidates.append(path.resolve())
    else:
        candidates = sorted(
            (path.resolve() for path in unprocessed_dir.iterdir() if path.is_file()),
            key=lambda path: path.name,
        )
    if limit is not None:
        candidates = candidates[:limit]
    return candidates


def unique_destination(directory: Path, filename: str) -> Path:
    destination = directory / filename
    if not destination.exists():
        return destination
    stem = destination.stem
    suffix = destination.suffix
    for index in range(2, 10000):
        candidate = directory / f"{stem}-{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find a unique destination for {destination}")


def move_to_directory(path: Path, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = unique_destination(directory, path.name)
    return Path(shutil.move(str(path), str(destination))).resolve()
