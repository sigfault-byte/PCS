import argparse
import json
from pathlib import Path

from ..src.assemblybot.config import INTERIM_DIR


def build_default_output_path(input_json_path: Path) -> Path:
    stem = input_json_path.stem.replace("_04_nlp", "")
    return INTERIM_DIR / f"{stem}validated_per.csv"


def load_document(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_document(doc: dict, json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)


KEEP_DEPS = {
    "nsubj",
    "csubj",
    "obj",
    "iobj",
    "obl:agent",
    "appos",
    "flat:name",
    "nmod",
}

KEEP_POS = {"VERB", "AUX", "PROPN", "NOUN"}
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


def get_children(tokens, parent_token):
    parent_i = parent_token["i"]
    return [tok for tok in tokens if tok.get("head_i") == parent_i]


def get_token_by_i(tokens: list[dict], i: int) -> dict | None:
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
    """
    Returns:
        score: int
        reasons: list[str]
        extracted_name: str
    """
    score = 0
    reasons = []

    dep = token.get("dep", "")
    pos = token.get("pos", "")
    text = token.get("text", "")
    head = get_token_by_i(tokens, token.get("head_i"))  # type: ignore I KNOW it is an int

    # 1) NER already selected it, so start from a small base.
    score += 1
    reasons.append("ent_type=PER")

    # 2) Proper nouns are much stronger than NOUN for real names.
    if pos == "PROPN":
        score += 3
        reasons.append("pos=PROPN")
    elif pos == "NOUN":
        score += 0
        reasons.append("pos=NOUN")

    # 3) Dependency role hints.
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

    # 4) Head-based clues.
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

    # 5) Children clues.
    children = get_children(tokens, token)

    if any(child.get("dep") == "flat:name" for child in children):
        score += 2
        reasons.append("has flat:name child")

    if any(child.get("dep") == "appos" for child in children):
        score += 1
        reasons.append("has appos child")

    # Extract name:
    # if current token is appos/PROPN and has flat:name children -> collect from here
    # if current token is flat:name and head is PROPN -> collect from head
    extracted_name = text

    if dep == "flat:name" and head and head.get("pos") == "PROPN":
        extracted_name = collect_flat_name(tokens, head)
        reasons.append("name rebuilt from head PROPN")
    elif pos == "PROPN":
        extracted_name = collect_flat_name(tokens, token)
        if extracted_name != text:
            reasons.append("name extended with flat:name")

    return score, reasons, extracted_name


def scan(input_path: Path, output_path: Path | None = None) -> None:
    if output_path is None:
        pass
        # output_path = build_default_output_path(input_path)

    document = load_document(input_path)
    segments = document.get("segments", [])

    for s in segments:
        spacy_data = s.get("nlp", {}).get("spacy", {})
        entities = spacy_data.get("entities", [])
        tokens = spacy_data.get("tokens", [])

        # not token shouldnt be happening, but bypassing empty entities is a win
        if not tokens or not entities:
            continue

        # First filter: segment must contain at least one PER entity.
        if not any(ent.get("label") == "PER" for ent in entities):
            continue

        per_tokens = [tok for tok in tokens if tok.get("ent_type") == "PER"]
        if not per_tokens:
            continue

        print(f"\n=== SEGMENT {s.get('segment_id')} ===")

        # Optional: show important verbs found in the segment
        important_verbs = [
            tok for tok in tokens if tok.get("lemma") in IMPORTANT_LEMMAS
        ]
        if important_verbs:
            print("IMPORTANT VERBS:", [tok["text"] for tok in important_verbs])

        seen_names = set()

        for tok in per_tokens:
            score, reasons, extracted_name = score_per_token(tok, tokens)

            # Avoid printing the same rebuilt name 3 times
            dedupe_key = (extracted_name, tok.get("i"))
            if dedupe_key in seen_names:
                continue
            seen_names.add(dedupe_key)

            head = get_token_by_i(tokens, tok.get("head_i"))
            head_text = head["text"] if head else "?"
            children = get_children(tokens, tok)
            child_texts = [c["text"] for c in children]

            print(
                f"TOKEN={tok['text']!r} "
                f"NAME={extracted_name!r} "
                f"POS={tok.get('pos')} "
                f"DEP={tok.get('dep')} "
                f"HEAD={head_text!r} "
                f"SCORE={score}"
            )
            print("  REASONS:", ", ".join(reasons))
            print("  CHILDREN:", child_texts)

        print("-------------------------------")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run heuristics to extract speakers on the JSON enriched by spaCy."
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


def get_agent_anchor(tokens, verb):
    for t in tokens:
        if t["head_i"] == verb["i"] and t["dep"] == "obl:agent":
            return t
    return None


def main() -> None:
    args = parse_args()

    output_path = scan(
        input_path=args.input_json,
        output_path=args.output,
    )

    print(f"NLP signal extratction enrichment written to: {output_path}")


if __name__ == "__main__":
    main()
