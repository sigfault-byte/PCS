from __future__ import annotations

import argparse
from dataclasses import dataclass

from assemblybot.models.turn_document import PersonIdentity


@dataclass(frozen=True)
class PerConfig:
    """Runtime and policy settings for PER extraction and speaker inference."""

    ner_model_name: str = "Jean-Baptiste/camembert-ner"
    ner_aggregation_strategy: str = "simple"
    per_confidence_threshold: float = 0.8
    fuzzy_match_threshold: int = 80
    token_match_threshold: int = 65
    context_radius: int = 80
    sentence_boundary_chars: str = ".?!"
    speaker_search_window: int = 5
    next_speaker_weight: int = 2
    previous_speaker_weight: int = 1
    next_speaker_patterns: tuple[str, ...] = (
        "la parole est à",
        "je donne la parole à",
        "vous avez la parole",
    )
    previous_speaker_patterns: tuple[str, ...] = ()
    generic_person_mentions: frozenset[str] = frozenset(
        {
            "madame la deputee",
            "madame la ministre",
            "madame la presidente",
            "mesdames les deputees",
            "mesdames les ministres",
            "monsieur le depute",
            "monsieur le ministre",
            "monsieur le president",
            "messieurs les deputes",
            "messieurs les ministres",
        }
    )
    assembly_chair_turn_patterns: tuple[str, ...] = (
        "la parole est à",
        "je donne la parole à",
        "vous avez la parole",
        "la séance est ouverte",
        "la séance est levée",
        "l'ordre du jour appelle",
        "le scrutin est ouvert",
        "le scrutin est clos",
        "je vais mettre aux voix",
        "je mets aux voix",
    )
    assembly_chair_identity: PersonIdentity = PersonIdentity(
        id="assembly_chair:yael-braun-pivet",
        name="Yaël Braun-Pivet",
        role="Présidente de l'Assemblée nationale",
        kind="assembly_chair",
    )
    ground_truth_headers: tuple[str, ...] = (
        "identifiant",
        "Prénom",
        "Nom",
        "Région",
        "Département",
        "Numéro de circonscription",
        "Profession",
        "Groupe politique (complet)",
        "Groupe politique (abrégé)",
    )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "PerConfig":
        return cls()


DEFAULT_PER_CONFIG = PerConfig()


def add_per_arguments(parser: argparse.ArgumentParser) -> None:
    """Register PER-specific CLI options on a stage parser."""
    parser.add_argument(
        "--deputies-ground-truth-csv",
        required=True,
        help="Path to a CSV representing the exact PER of the deputy speakers",
    )
    parser.add_argument(
        "--ministers-ground-truth-csv",
        required=True,
        help="Path to a CSV representing ministers and Assembly president speakers",
    )
