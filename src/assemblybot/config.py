from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "audio"
INTERIM_DIR = DATA_DIR / "interim"
OUTPUT_DIR = DATA_DIR / "output"
AUDIO_AUDIT_DIR = DATA_DIR / "audio-audit"
EMBEDDING_DIR = DATA_DIR / "embedding"
DB_DIR = DATA_DIR / "db"
DIARIZATION_EMBEDDING_DIR = INTERIM_DIR / "embedding"

DEFAULT_SQLITE_DB_PATH = DB_DIR / "assemblybot.sqlite"

DIARIZATION_SEGMENT_EMBEDDINGS_SUFFIX = "_01_segment_embeddings"
DIARIZATION_SPEAKER_CENTROIDS_SUFFIX = "_01_speaker_centroids"

for directory in (
    INPUT_DIR,
    INTERIM_DIR,
    OUTPUT_DIR,
    AUDIO_AUDIT_DIR,
    EMBEDDING_DIR,
    DB_DIR,
    DIARIZATION_EMBEDDING_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)
