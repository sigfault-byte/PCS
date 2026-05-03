import argparse
import csv
import json
from pathlib import Path

from assemblybot.config import INTERIM_DIR

CSV_FIELDS = [
    "segment_id",
    "entities",
]


def build_default_output_path(input_json_path: Path) -> Path:
    stem = input_json_path.stem.replace("_04_nlp", "")
    return INTERIM_DIR / f"{stem}_regex_next_speaker.csv"


def load_document(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def has_parole_est_a(text: str) -> bool:
    return "la parole est" in text.lower()


def scan(input_path: Path, output_path: Path | None = None) -> Path:
    if output_path is None:
        output_path = build_default_output_path(input_path)

    document = load_document(input_path)
    segments = document.get("segments", [])

    rows: list[dict[str, str | int]] = []

    for s in segments:
        raw_text = s.get("text", {}).get("raw", "")

        if has_parole_est_a(raw_text):
            spacy_data_entities = s.get("nlp", {}).get("spacy", {}).get("entities", [])

            per_ent = [ent for ent in spacy_data_entities if ent.get("label") == "PER"]

            rows.append(
                {
                    "segment_id": s.get("segment_id", ""),
                    "entities": per_ent,  # type: ignore
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="regex match official formula into a CSV."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help="Path to the input JSON file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = scan(input_path=args.input_json, output_path=args.output)
    print(f"Regex extraction CSV written to: {output_path}")


if __name__ == "__main__":
    main()
