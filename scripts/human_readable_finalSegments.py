import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.assemblybot.helper.document import load_document

FILE = PROJECT_ROOT / "data" / "interim" / "assemblee_nov26_2024_03_merge.json"

document = load_document(FILE)

for segment in document.segments:
    print(
        f"[{segment.speaker.speaker_id}]"
        f"[{segment.time.start_ts} -> {segment.time.end_ts}]: "
        f'"{segment.text.raw}"'
    )
    print("\n")
