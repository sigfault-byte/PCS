from pathlib import Path

from ..config import INTERIM_DIR


def build_default_output_path(
    input_path: Path,
    suffix: str,
    extension: str,
) -> Path:
    return INTERIM_DIR / f"{input_path.stem}{suffix}.{extension}"
