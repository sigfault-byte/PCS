from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
INTERIM_DIR = DATA_DIR / "interim"
OUTPUT_DIR = DATA_DIR / "output"

for directory in (INPUT_DIR, INTERIM_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)
