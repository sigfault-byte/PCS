import argparse
import json
import time
from pathlib import Path

import spacy
from spacy.language import Language

from assemblybot.config import INTERIM_DIR


def build_default_output_path(input_json_path: Path) -> Path:
    stem = input_json_path.stem.replace("_03_merged", "")
    return INTERIM_DIR / f"{stem}_04_nlp.json"


def load_document(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_document(doc: dict, json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)


def load_model() -> Language:
    return spacy.load("fr_core_news_sm")


def extract_entities(doc) -> list[dict]:
    entities = []

    for ent in doc.ents:
        entities.append(
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start,
                "end": ent.end,
            }
        )

    return entities


def extract_tokens(doc) -> list[dict]:
    tokens = []

    for token in doc:
        tokens.append(
            {
                "i": token.i,
                "text": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "dep": token.dep_,
                "head_i": token.head.i,
                "head_text": token.head.text,
                "ent_type": token.ent_type_,
                "morph": str(token.morph),
            }
        )

    return tokens


def process_segment(nlp: Language, segment: dict) -> dict:
    text_block = segment.get("text", {})
    text = text_block.get("normalized") or text_block.get("raw", "")
    doc = nlp(text)

    return {
        "spacy": {
            "entities": extract_entities(doc),
            "tokens": extract_tokens(doc),
        }
    }


def nlp_extract_signal(input_path: Path, output_path: Path | None = None) -> Path:
    start = time.time()
    document = load_document(input_path)
    step1 = time.time()
    nlp = load_model()
    step2 = time.time()

    if output_path is None:
        output_path = build_default_output_path(input_path)

    segments = document.get("segments", [])
    enriched_segments = []

    seg_size = len(segments)

    step3 = time.time()
    for idx, segment in enumerate(segments, start=1):
        enriched = dict(segment)
        enriched["nlp"] = process_segment(nlp, segment)
        enriched_segments.append(enriched)

        if idx % 50 == 0:
            print(f"Processed {idx}/{len(segments)} segments")
    step4 = time.time()

    document["segments"] = enriched_segments
    save_document(document, output_path)
    print(f"Loading doc: {step1 - start}s")
    print(f"Loading spacy: {step2 - step1}s")
    print(f"Loading parsing {seg_size} segments: {step4 - step3}s")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run spaCy NLP enrichment on diarization JSON."
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

    output_path = nlp_extract_signal(
        input_path=args.input_json,
        output_path=args.output,
    )

    print(f"NLP enrichment written to: {output_path}")


if __name__ == "__main__":
    main()
