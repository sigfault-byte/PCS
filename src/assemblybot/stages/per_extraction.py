import argparse
import json
from pathlib import Path

from assemblybot.helper.directory import build_default_output_path
from assemblybot.helper.document import save_turn_document
from assemblybot.models.turn_document import TurnDocument
from assemblybot.per_config import PerConfig, add_per_arguments
from assemblybot.stages.per_analysis import (
    NERCallable,
    build_mentioned_persons_by_turn,
    build_speaker_person_summary,
    build_speaker_identity_evidence_by_turn,
    build_turn_analysis,
    collect_person_mentions,
    predict_person_turns,
)
from assemblybot.stages.per_identity import KnownPerson, load_all_known_people

# TODO: drop those token if they are on the left. So the fuzzy match is only on the name
# TITLE_TOKENS = {
#     "madame", "monsieur", "m", "mme",
#     "president", "presidente",
#     "ministre", "depute", "deputee",
#     "la", "le"
# }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extracts and propagates PER identities onto turn analysis."
    )
    parser.add_argument(
        "--input-json",
        required=True,
        help="Existing turn document JSON to enrich",
    )
    parser.add_argument(
        "--output-json",
        help="Optional output JSON path (default: generated in interim directory)",
    )
    add_per_arguments(parser)

    return parser.parse_args(argv)


def load_turn_document(json_path: Path) -> TurnDocument:
    with json_path.open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)
    return TurnDocument.from_dict(data)


def enrich_turn_document(
    document: TurnDocument,
    known_people: list[KnownPerson],
    ner: NERCallable,
    config: PerConfig,
) -> TurnDocument:
    mentions = collect_person_mentions(document.turns, ner, config=config)
    predictions = predict_person_turns(
        document.turns,
        mentions,
        known_people,
        config,
    )
    speaker_summary = build_speaker_person_summary(
        document.turns,
        predictions,
        config,
    )
    speaker_identity_evidence_by_turn = build_speaker_identity_evidence_by_turn(
        document.turns,
        predictions,
        config,
    )
    mentioned_persons_by_turn = build_mentioned_persons_by_turn(
        mentions,
        known_people,
        config,
    )

    return TurnDocument(
        turns=document.turns,
        turns_analysis=build_turn_analysis(
            document.turns,
            speaker_summary,
            mentioned_persons_by_turn,
            speaker_identity_evidence_by_turn,
            config,
        ),
    )


def build_ner_pipeline(config: PerConfig) -> NERCallable:
    from transformers import pipeline

    return pipeline(
        "token-classification",
        model=config.ner_model_name,
        aggregation_strategy=config.ner_aggregation_strategy,
    )


def main() -> None:
    args = parse_args()
    config = PerConfig.from_args(args)

    input_json_path = Path(args.input_json).resolve()
    deputies_csv_path = Path(args.deputies_ground_truth_csv).resolve()
    ministers_csv_path = Path(args.ministers_ground_truth_csv).resolve()
    document = load_turn_document(input_json_path)
    known_people = load_all_known_people(
        deputies_csv_path,
        ministers_csv_path,
        config,
    )
    ner = build_ner_pipeline(config)

    base = input_json_path.stem.rsplit("_", 2)[0]
    output_json_path = (
        Path(args.output_json).resolve()
        if args.output_json
        else build_default_output_path(
            Path(base),
            "_02_per_extraction",
            "json",
        )
    )

    enriched_document = enrich_turn_document(document, known_people, ner, config)
    save_turn_document(enriched_document, output_json_path)


if __name__ == "__main__":
    main()
