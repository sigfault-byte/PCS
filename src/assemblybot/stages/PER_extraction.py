import argparse
import csv
import json
import unicodedata
from collections import Counter, defaultdict

from rapidfuzz import fuzz, process
from transformers import pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("PER extraction on the current turns.")
    )
    parser.add_argument(
        "--input-json",
        required=True,
        help="Existing canonical document JSON to load the aligned segments from",
    )
    parser.add_argument(
        "--output-json",
        help="Optional output JSON path (default: generated in interim directory)",
    )

    parser.add_argument(
        "--csv-ground-truth-PER",
        required=True,
        help="path to a CSV representing the exact PER of the speakers",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()


if __name__ == "__main__":
    main()
