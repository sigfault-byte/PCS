import csv
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz, process

from assemblybot.models.turn_document import PersonIdentity

FUZZY_MATCH_THRESHOLD = 80
TOKEN_MATCH_THRESHOLD = 65

GROUND_TRUTH_HEADERS = [
    "identifiant",
    "Prénom",
    "Nom",
    "Région",
    "Département",
    "Numéro de circonscription",
    "Profession",
    "Groupe politique (complet)",
    "Groupe politique (abrégé)",
]


@dataclass(frozen=True)
class KnownPerson:
    identity: PersonIdentity
    normalized_name: str

    @property
    def identity_key(self) -> tuple[str, str | None, str]:
        return (
            self.identity.kind,
            self.identity.id,
            self.identity.name,
        )


@dataclass(frozen=True)
class PersonResolution:
    identity: PersonIdentity
    match_score: float
    is_known_person: bool

    @property
    def identity_key(self) -> tuple[str, str | None, str]:
        return (
            self.identity.kind,
            self.identity.id,
            self.identity.name,
        )


def normalize_name(name: str) -> str:
    name = name.lower().strip()
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    return " ".join(name.split())


def name_tokens(normalized_name: str) -> list[str]:
    return [token for token in normalized_name.split() if token]


def raw_person_resolution(normalized_name: str, match_score: float = 0.0) -> PersonResolution:
    return PersonResolution(
        identity=PersonIdentity(
            id=None,
            name=normalized_name,
            role=None,
            kind="raw_per",
        ),
        match_score=match_score,
        is_known_person=False,
    )


def tokens_have_strong_overlap(
    query_tokens: list[str],
    candidate_tokens: list[str],
) -> bool:
    if not query_tokens or not candidate_tokens:
        return False

    has_exact_token_match = any(
        query_token == candidate_token
        for query_token in query_tokens
        for candidate_token in candidate_tokens
    )

    if not has_exact_token_match:
        return False

    return all(
        any(
            query_token == candidate_token
            or fuzz.ratio(query_token, candidate_token) >= TOKEN_MATCH_THRESHOLD
            for candidate_token in candidate_tokens
        )
        for query_token in query_tokens
    )


def person_kind_from_identifier(identifier: str) -> str:
    if identifier.startswith("minister:"):
        return "minister"

    if (
        identifier.startswith("assembly_chair:")
        or identifier.startswith("assembly_chari:")
        or identifier.startswith("assembly_chaii:")
    ):
        return "assembly_chair"

    return "deputy"


def validate_ground_truth_csv(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.reader(input_file)
        rows = list(reader)

    if not rows:
        raise ValueError(f"{csv_path} is empty")

    header = rows[0]

    if header != GROUND_TRUTH_HEADERS:
        raise ValueError(
            f"{csv_path} has invalid headers: {header}. "
            f"Expected: {GROUND_TRUTH_HEADERS}"
        )

    bad_rows = [
        line_number
        for line_number, row in enumerate(rows[1:], start=2)
        if len(row) != len(header)
    ]

    if bad_rows:
        raise ValueError(
            f"{csv_path} has malformed rows with wrong column counts at lines: "
            f"{bad_rows}"
        )

    return [dict(zip(header, row, strict=True)) for row in rows[1:]]


def known_person_from_row(row: dict[str, str], *, source_kind: str) -> KnownPerson:
    full_name = f"{row['Prénom']} {row['Nom']}".strip()
    identifier = row["identifiant"].strip()
    kind = (
        "deputy" if source_kind == "deputy" else person_kind_from_identifier(identifier)
    )
    role = "Député" if kind == "deputy" else row["Profession"].strip() or None

    return KnownPerson(
        identity=PersonIdentity(
            id=identifier,
            name=full_name,
            role=role,
            kind=kind,
        ),
        normalized_name=normalize_name(full_name),
    )


def load_known_people(csv_path: Path, *, source_kind: str) -> list[KnownPerson]:
    return [
        known_person_from_row(row, source_kind=source_kind)
        for row in validate_ground_truth_csv(csv_path)
    ]


def load_all_known_people(
    deputies_csv_path: Path,
    ministers_csv_path: Path,
) -> list[KnownPerson]:
    return [
        *load_known_people(deputies_csv_path, source_kind="deputy"),
        *load_known_people(ministers_csv_path, source_kind="minister"),
    ]


def resolve_known_person(
    normalized_name: str,
    known_people: list[KnownPerson],
    threshold: int = FUZZY_MATCH_THRESHOLD,
) -> PersonResolution:
    known_names = [person.normalized_name for person in known_people]
    name_to_people: defaultdict[str, list[KnownPerson]] = defaultdict(list)

    for person in known_people:
        name_to_people[person.normalized_name].append(person)

    if not known_names:
        return raw_person_resolution(normalized_name)

    query_tokens = name_tokens(normalized_name)

    if len(query_tokens) < 2:
        exact_matches = name_to_people.get(normalized_name, [])

        if len(exact_matches) == 1:
            return PersonResolution(
                identity=exact_matches[0].identity,
                match_score=100.0,
                is_known_person=True,
            )

        return raw_person_resolution(normalized_name)

    match, score, _ = process.extractOne(
        normalized_name,
        known_names,
        scorer=fuzz.WRatio,
    )

    candidate_tokens = name_tokens(match)

    if score >= threshold and tokens_have_strong_overlap(query_tokens, candidate_tokens):
        known_person = name_to_people[match][0]
        return PersonResolution(
            identity=known_person.identity,
            match_score=float(score),
            is_known_person=True,
        )

    return raw_person_resolution(normalized_name, float(score))
