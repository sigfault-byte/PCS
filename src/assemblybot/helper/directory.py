from pathlib import Path

from ..config import INTERIM_DIR


def build_default_output_path(
    input_path: Path,
    suffix: str,
    extension: str,
    output_dir: Path = INTERIM_DIR,
) -> Path:
    return output_dir / f"{input_path.stem}{suffix}.{extension}"
