from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "audio"
INTERIM_DIR = DATA_DIR / "interim"
OUTPUT_DIR = DATA_DIR / "output"
AUDIO_AUDIT_DIR = DATA_DIR / "audio-audit"

for directory in (INPUT_DIR, INTERIM_DIR, OUTPUT_DIR, AUDIO_AUDIT_DIR):
    directory.mkdir(parents=True, exist_ok=True)
