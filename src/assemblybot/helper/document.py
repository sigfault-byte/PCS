import json
from pathlib import Path

from ..models.document import CanonicalDocument


def load_document(json_path: Path) -> CanonicalDocument:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return CanonicalDocument.from_dict(data)


def save_document(document: CanonicalDocument, json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(document.to_dict(), f, ensure_ascii=False, indent=2)  # type: ignore
