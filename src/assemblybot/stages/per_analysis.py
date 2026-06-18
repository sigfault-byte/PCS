from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable

from assemblybot.models.turn_document import PersonIdentity, Turn, TurnAnalysis
from assemblybot.stages.per_identity import (
    KnownPerson,
    PersonResolution,
    normalize_name,
    resolve_known_person,
)

PER_CONFIDENCE_THRESHOLD = 0.8
CONTEXT_RADIUS = 80
SENTENCE_BOUNDARY_CHARS = ".?!"
SPEAKER_SEARCH_WINDOW = 5
NEXT_SPEAKER_WEIGHT = 2
PREVIOUS_SPEAKER_WEIGHT = 1
CURRENT_SPEAKER_SOURCE_NEXT = "inferred_from_next_speaker"
CURRENT_SPEAKER_SOURCE_PREVIOUS = "inferred_from_previous_speaker"
CURRENT_SPEAKER_SOURCE_ASSEMBLY_CHAIR = "hardcoded_assembly_chair"
ASSEMBLY_CHAIR_IDENTITY = PersonIdentity(
    id="assembly_chair:yael-braun-pivet",
    name="Yaël Braun-Pivet",
    role="Présidente de l'Assemblée nationale",
    kind="assembly_chair",
)

NEXT_SPEAKER_PATTERNS = [
    "la parole est à",
    "je donne la parole à",
    "vous avez la parole",
]

PREVIOUS_SPEAKER_PATTERNS: list[str] = []

GENERIC_PERSON_MENTIONS = {
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

ASSEMBLY_CHAIR_TURN_PATTERNS = [
    *NEXT_SPEAKER_PATTERNS,
    "la séance est ouverte",
    "la séance est levée",
    "l'ordre du jour appelle",
    "le scrutin est ouvert",
    "le scrutin est clos",
    "je vais mettre aux voix",
    "je mets aux voix",
]

NERCallable = Callable[[str], list[dict[str, Any]]]


@dataclass(frozen=True)
class SpeakerPersonPrediction:
    source_turn_id: int
    predicted_turn_id: int
    speaker_raw: str
    speaker_normalized: str
    resolution: PersonResolution
    role: str
    weight: int

    @property
    def identity_key(self) -> tuple[str, str | None, str]:
        return self.resolution.identity_key


def is_generic_person_mention(normalized_name: str) -> bool:
    return normalized_name in GENERIC_PERSON_MENTIONS


def context_window(
    text: str,
    start: int,
    end: int,
    radius: int = CONTEXT_RADIUS,
) -> str:
    return text[max(0, start - radius) : min(len(text), end + radius)].lower()


def sentence_context(text: str, start: int, end: int) -> tuple[str, int, int]:
    previous_boundaries = [
        text.rfind(boundary, 0, start) for boundary in SENTENCE_BOUNDARY_CHARS
    ]
    previous_boundary = max(previous_boundaries)
    span_start = previous_boundary + 1 if previous_boundary >= 0 else 0

    next_boundaries = [
        boundary_index
        for boundary in SENTENCE_BOUNDARY_CHARS
        if (boundary_index := text.find(boundary, end)) >= 0
    ]
    span_end = min(next_boundaries) if next_boundaries else len(text)

    return (
        text[span_start:span_end].lower(),
        start - span_start,
        end - span_start,
    )


def anchor_before_entity(
    context: str,
    entity_start: int,
    patterns: list[str],
) -> bool:
    prefix = context[:entity_start]
    return any(pattern in prefix for pattern in patterns)


def next_speaker_anchor_position(context: str, entity_start: int) -> int | None:
    prefix = context[:entity_start]
    positions = []

    for pattern in NEXT_SPEAKER_PATTERNS:
        position = prefix.rfind(pattern)

        if position >= 0:
            positions.append(position)

    return max(positions) if positions else None


def infer_person_role(text: str, start: int, end: int) -> str | None:
    context, entity_start, _ = sentence_context(text, start, end)

    if next_speaker_anchor_position(context, entity_start) is not None:
        return "probable_next_speaker"

    return None


def next_speaker_anchor_key(text: str, start: int, end: int) -> tuple[int, int] | None:
    context, entity_start, _ = sentence_context(text, start, end)
    anchor_position = next_speaker_anchor_position(context, entity_start)

    if anchor_position is None:
        return None

    return start - entity_start, anchor_position


def is_assembly_chair_turn(text: str) -> bool:
    normalized_text = text.lower()
    return any(pattern in normalized_text for pattern in ASSEMBLY_CHAIR_TURN_PATTERNS)


def anchor_weight(role: str) -> int:
    if role == "probable_next_speaker":
        return NEXT_SPEAKER_WEIGHT
    return PREVIOUS_SPEAKER_WEIGHT


def current_speaker_source(role: str) -> str:
    if role == "probable_next_speaker":
        return CURRENT_SPEAKER_SOURCE_NEXT
    return CURRENT_SPEAKER_SOURCE_PREVIOUS


def collect_person_mentions(
    turns: list[Turn],
    ner: NERCallable,
    threshold: float = PER_CONFIDENCE_THRESHOLD,
) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []

    for turn in turns:
        for entity in ner(turn.text):
            if entity.get("entity_group") != "PER":
                continue

            if float(entity.get("score", 0.0)) <= threshold:
                continue

            normalized_name = normalize_name(str(entity.get("word", "")))

            if is_generic_person_mention(normalized_name):
                continue

            mentions.append(
                {
                    "entity": entity,
                    "normalized_name": normalized_name,
                    "text": turn.text,
                    "turn_id": turn.turn_id,
                }
            )

    return mentions


def find_predicted_turn_id(
    turns_by_id: dict[int, Turn],
    source_turn_id: int,
    role: str,
    search_window: int = SPEAKER_SEARCH_WINDOW,
) -> int | None:
    current_turn = turns_by_id[source_turn_id]
    current_speaker = current_turn.speaker_id

    direction = 1 if role == "probable_next_speaker" else -1
    fallback_turn_id = source_turn_id + direction
    predicted_turn_id: int | None = None

    for offset in range(1, search_window + 1):
        candidate_turn_id = source_turn_id + (direction * offset)
        candidate = turns_by_id.get(candidate_turn_id)

        if candidate is None:
            continue

        if candidate.speaker_id != current_speaker:
            return candidate.turn_id

        predicted_turn_id = fallback_turn_id

    return predicted_turn_id if predicted_turn_id in turns_by_id else None


def predict_person_turns(
    turns: list[Turn],
    mentions: list[dict[str, Any]],
    known_people: list[KnownPerson],
) -> list[SpeakerPersonPrediction]:
    predictions: list[SpeakerPersonPrediction] = []
    turns_by_id = {turn.turn_id: turn for turn in turns}
    first_next_mentions: dict[tuple[int, int, int], dict[str, Any]] = {}

    for mention in mentions:
        entity = mention["entity"]
        key = next_speaker_anchor_key(
            mention["text"],
            int(entity["start"]),
            int(entity["end"]),
        )

        if key is None:
            continue

        first_next_key = (int(mention["turn_id"]), *key)
        existing = first_next_mentions.get(first_next_key)

        if existing is None or int(entity["start"]) < int(existing["entity"]["start"]):
            first_next_mentions[first_next_key] = mention

    for mention in mentions:
        entity = mention["entity"]
        role = infer_person_role(
            mention["text"],
            int(entity["start"]),
            int(entity["end"]),
        )

        if role is None:
            continue

        key = next_speaker_anchor_key(
            mention["text"],
            int(entity["start"]),
            int(entity["end"]),
        )

        if key is None:
            continue

        if first_next_mentions[(int(mention["turn_id"]), *key)] is not mention:
            continue

        source_turn_id = int(mention["turn_id"])
        predicted_turn_id = find_predicted_turn_id(
            turns_by_id,
            source_turn_id,
            role,
        )

        if predicted_turn_id is None:
            continue

        resolution = resolve_known_person(mention["normalized_name"], known_people)
        predictions.append(
            SpeakerPersonPrediction(
                source_turn_id=source_turn_id,
                predicted_turn_id=predicted_turn_id,
                speaker_raw=str(entity.get("word", "")),
                speaker_normalized=mention["normalized_name"],
                resolution=resolution,
                role=role,
                weight=anchor_weight(role),
            )
        )

    return predictions


def build_mentioned_persons_by_turn(
    mentions: list[dict[str, Any]],
    known_people: list[KnownPerson],
) -> dict[int, list[PersonIdentity]]:
    mentioned_persons_by_turn: dict[int, list[PersonIdentity]] = defaultdict(list)
    seen_by_turn: defaultdict[int, set[tuple[str, str | None, str]]] = defaultdict(set)

    for mention in mentions:
        turn_id = int(mention["turn_id"])
        resolution = resolve_known_person(mention["normalized_name"], known_people)
        identity_key = resolution.identity_key

        if identity_key in seen_by_turn[turn_id]:
            continue

        mentioned_persons_by_turn[turn_id].append(resolution.identity)
        seen_by_turn[turn_id].add(identity_key)

    return mentioned_persons_by_turn


def build_speaker_person_summary(
    turns: list[Turn],
    predictions: list[SpeakerPersonPrediction],
) -> dict[str, dict[str, float | int | str | PersonIdentity]]:
    matrix: defaultdict[str, Counter[tuple[str, str | None, str]]] = defaultdict(
        Counter
    )
    source_matrix: defaultdict[
        str,
        defaultdict[tuple[str, str | None, str], Counter[str]],
    ] = defaultdict(lambda: defaultdict(Counter))
    identities_by_key: dict[tuple[str, str | None, str], PersonIdentity] = {}
    turns_by_id = {turn.turn_id: turn for turn in turns}

    for prediction in predictions:
        turn = turns_by_id.get(prediction.predicted_turn_id)

        if turn is None or turn.speaker_id is None:
            continue

        identities_by_key[prediction.identity_key] = prediction.resolution.identity
        matrix[turn.speaker_id][prediction.identity_key] += prediction.weight
        source_matrix[turn.speaker_id][prediction.identity_key][
            current_speaker_source(prediction.role)
        ] += prediction.weight

    speaker_id_to_person: dict[str, dict[str, float | int | str | PersonIdentity]] = {}

    for speaker_id, counts in matrix.items():
        identity_key, count = counts.most_common(1)[0]
        total = sum(counts.values())
        source_counts = source_matrix[speaker_id][identity_key]
        source = max(
            source_counts,
            key=lambda item: (
                source_counts[item],
                item == CURRENT_SPEAKER_SOURCE_NEXT,
            ),
        )
        speaker_id_to_person[speaker_id] = {
            "person": identities_by_key[identity_key],
            "source": source,
            "count": count,
            "total": total,
            "purity": count / total,
        }

    return speaker_id_to_person


def build_turn_analysis(
    turns: list[Turn],
    speaker_id_to_person: dict[str, dict[str, float | int | str | PersonIdentity]],
    mentioned_persons_by_turn: dict[int, list[PersonIdentity]],
) -> list[TurnAnalysis]:
    turns_analysis: list[TurnAnalysis] = []

    for turn in turns:
        mentioned_persons = mentioned_persons_by_turn.get(turn.turn_id, [])
        hardcoded_current_speaker = (
            ASSEMBLY_CHAIR_IDENTITY if is_assembly_chair_turn(turn.text) else None
        )
        speaker_data = (
            speaker_id_to_person.get(turn.speaker_id)
            if turn.speaker_id is not None
            else None
        )

        if hardcoded_current_speaker is not None:
            turns_analysis.append(
                TurnAnalysis(
                    turn_id=turn.turn_id,
                    current_speaker=hardcoded_current_speaker,
                    current_speaker_source=CURRENT_SPEAKER_SOURCE_ASSEMBLY_CHAIR,
                    current_speaker_purity=1.0,
                    mentioned_persons=mentioned_persons,
                )
            )
            continue

        if speaker_data is None:
            turns_analysis.append(
                TurnAnalysis(
                    turn_id=turn.turn_id,
                    mentioned_persons=mentioned_persons,
                )
            )
            continue

        turns_analysis.append(
            TurnAnalysis(
                turn_id=turn.turn_id,
                current_speaker=speaker_data["person"],  # type: ignore[arg-type]
                current_speaker_source=str(speaker_data["source"]),
                current_speaker_purity=float(speaker_data["purity"]),
                mentioned_persons=mentioned_persons,
            )
        )

    return turns_analysis
