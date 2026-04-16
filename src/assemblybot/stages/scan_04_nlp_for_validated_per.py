import argparse
import csv
import json
from pathlib import Path

from assemblybot.config import INTERIM_DIR

IMPORTANT_HEAD = {
    "monsieur",
    "madame",
    "m.",
    "mme",
    "président",
    "présidente",
    "ministre",
}
IMPORTANT_LEMMAS = {
    "adresser",
    "appeler",
    "annoncer",
    "confirmer",
    "demander",
    "déclarer",
    "dire",
    "donner",
    "entendre",
    "être",
    "indiquer",
    "interroger",
    "poser",
    "prendre",
    "présenter",
    "répondre",
    "signaler",
}

CSV_FIELDS = [
    "segment_id",
    "candidate_token_i",
    "candidate_token_text",
    "candidate_name",
    "score",
    "ent_type",
    "pos",
    "dep",
    "morph",
    "head_i",
    "head_text",
    "head_pos",
    "head_dep",
    "important_verbs",
    "important_lemmas",
    "child_texts",
    "child_deps",
    "reasons",
]


def build_default_output_path(input_json_path: Path) -> Path:
    stem = input_json_path.stem.replace("_04_nlp", "")
    return INTERIM_DIR / f"{stem}_validated_per.csv"


def load_document(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_children(tokens: list[dict], parent_token: dict) -> list[dict]:
    parent_i = parent_token["i"]
    return [tok for tok in tokens if tok.get("head_i") == parent_i]


def get_token_by_i(tokens: list[dict], i: int | None) -> dict | None:
    if i is None:
        return None
    for tok in tokens:
        if tok.get("i") == i:
            return tok
    return None


def collect_flat_name(tokens: list[dict], root_token: dict) -> str:
    parts = [root_token["text"]]
    flat_parts = [
        tok for tok in get_children(tokens, root_token) if tok.get("dep") == "flat:name"
    ]
    flat_parts.sort(key=lambda x: x["i"])
    parts.extend(tok["text"] for tok in flat_parts)
    return " ".join(parts)


def score_per_token(token: dict, tokens: list[dict]) -> tuple[int, list[str], str]:
    score = 0
    reasons: list[str] = []

    dep = token.get("dep", "")
    pos = token.get("pos", "")
    text = token.get("text", "")
    head = get_token_by_i(tokens, token.get("head_i"))

    score += 1
    reasons.append("ent_type=PER")

    if pos == "PROPN":
        score += 3
        reasons.append("pos=PROPN")
    elif pos == "NOUN":
        reasons.append("pos=NOUN")

    if dep == "flat:name":
        score += 3
        reasons.append("dep=flat:name")
    elif dep == "appos":
        score += 2
        reasons.append("dep=appos")
    elif dep == "nsubj":
        score += 2
        reasons.append("dep=nsubj")
    elif dep == "obl:agent":
        score += 2
        reasons.append("dep=obl:agent")
    elif dep == "vocative":
        score -= 2
        reasons.append("dep=vocative (weaker)")

    if head:
        head_dep = head.get("dep", "")
        head_text = head.get("text", "")
        head_pos = head.get("pos", "")

        if head_dep in {"obl:agent", "nsubj"}:
            score += 2
            reasons.append(f"head.dep={head_dep}")

        if head_text.lower() in IMPORTANT_HEAD:
            score += 2
            reasons.append(f"head.title={head_text}")

        if head_pos == "PROPN":
            score += 1
            reasons.append("head.pos=PROPN")

    children = get_children(tokens, token)

    if any(child.get("dep") == "flat:name" for child in children):
        score += 2
        reasons.append("has flat:name child")

    if any(child.get("dep") == "appos" for child in children):
        score += 1
        reasons.append("has appos child")

    extracted_name = text

    if dep == "flat:name" and head and head.get("pos") == "PROPN":
        extracted_name = collect_flat_name(tokens, head)
        reasons.append("name rebuilt from head PROPN")
    elif pos == "PROPN":
        extracted_name = collect_flat_name(tokens, token)
        if extracted_name != text:
            reasons.append("name extended with flat:name")

    return score, reasons, extracted_name


def scan(input_path: Path, output_path: Path | None = None) -> Path:
    if output_path is None:
        output_path = build_default_output_path(input_path)

    document = load_document(input_path)
    segments = document.get("segments", [])

    rows: list[dict[str, str | int]] = []

    for s in segments:
        spacy_data = s.get("nlp", {}).get("spacy", {})
        entities = spacy_data.get("entities", [])
        tokens = spacy_data.get("tokens", [])

        if not tokens or not entities:
            continue

        if not any(ent.get("label") == "PER" for ent in entities):
            continue

        per_tokens = [tok for tok in tokens if tok.get("ent_type") == "PER"]
        if not per_tokens:
            continue

        important_verbs = [
            tok for tok in tokens if tok.get("lemma") in IMPORTANT_LEMMAS
        ]
        important_verbs_text = " | ".join(
            tok.get("text", "") for tok in important_verbs
        )
        important_verbs_lemmas = " | ".join(
            tok.get("lemma", "") for tok in important_verbs
        )

        seen_rows: set[tuple[str, int | None]] = set()

        for tok in per_tokens:
            score, reasons, extracted_name = score_per_token(tok, tokens)
            if not score > 6:
                continue

            dedupe_key = (extracted_name, tok.get("i"))
            if dedupe_key in seen_rows:
                continue
            seen_rows.add(dedupe_key)

            head = get_token_by_i(tokens, tok.get("head_i"))
            children = get_children(tokens, tok)

            rows.append(
                {
                    "segment_id": s.get("segment_id", ""),
                    "candidate_token_i": tok.get("i", ""),
                    "candidate_token_text": tok.get("text", ""),
                    "candidate_name": extracted_name,
                    "score": score,
                    "ent_type": tok.get("ent_type", ""),
                    "pos": tok.get("pos", ""),
                    "dep": tok.get("dep", ""),
                    "morph": tok.get("morph", ""),
                    "head_i": tok.get("head_i", ""),
                    "head_text": head.get("text", "") if head else "",
                    "head_pos": head.get("pos", "") if head else "",
                    "head_dep": head.get("dep", "") if head else "",
                    "important_verbs": important_verbs_text,
                    "important_lemmas": important_verbs_lemmas,
                    "child_texts": " | ".join(
                        child.get("text", "") for child in children
                    ),
                    "child_deps": " | ".join(
                        child.get("dep", "") for child in children
                    ),
                    "reasons": " | ".join(reasons),
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
        description="Flatten spaCy PER speaker-candidate signals into a CSV for review."
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
        help="Optional output CSV path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = scan(input_path=args.input_json, output_path=args.output)
    print(f"NLP signal extraction CSV written to: {output_path}")


if __name__ == "__main__":
    main()
