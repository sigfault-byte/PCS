from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable

from assemblybot.models.turn_document import (
    PersonIdentity,
    SpeakerIdentityEvidence,
    Turn,
    TurnAnalysis,
)
from assemblybot.per_config import DEFAULT_PER_CONFIG, PerConfig
from assemblybot.stages.per_identity import (
    KnownPerson,
    PersonResolution,
    normalize_name,
    resolve_known_person,
)

PER_CONFIDENCE_THRESHOLD = DEFAULT_PER_CONFIG.per_confidence_threshold
CONTEXT_RADIUS = DEFAULT_PER_CONFIG.context_radius
SENTENCE_BOUNDARY_CHARS = DEFAULT_PER_CONFIG.sentence_boundary_chars
SPEAKER_SEARCH_WINDOW = DEFAULT_PER_CONFIG.speaker_search_window
NEXT_SPEAKER_WEIGHT = DEFAULT_PER_CONFIG.next_speaker_weight
PREVIOUS_SPEAKER_WEIGHT = DEFAULT_PER_CONFIG.previous_speaker_weight
CURRENT_SPEAKER_SOURCE_CHAIR_NEXT_CALL = "chair_next_speaker_call"
CURRENT_SPEAKER_SOURCE_NEXT = CURRENT_SPEAKER_SOURCE_CHAIR_NEXT_CALL
CURRENT_SPEAKER_SOURCE_PREVIOUS = "inferred_from_previous_speaker"
CURRENT_SPEAKER_SOURCE_ASSEMBLY_CHAIR = "hardcoded_assembly_chair"
CURRENT_SPEAKER_SOURCE_PROPAGATED = "propagated_from_speaker_cluster"
ASSEMBLY_CHAIR_IDENTITY = DEFAULT_PER_CONFIG.assembly_chair_identity

NEXT_SPEAKER_PATTERNS = list(DEFAULT_PER_CONFIG.next_speaker_patterns)

PREVIOUS_SPEAKER_PATTERNS = list(DEFAULT_PER_CONFIG.previous_speaker_patterns)

GENERIC_PERSON_MENTIONS = set(DEFAULT_PER_CONFIG.generic_person_mentions)

ASSEMBLY_CHAIR_TURN_PATTERNS = list(DEFAULT_PER_CONFIG.assembly_chair_turn_patterns)

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


def is_generic_person_mention_with_config(
    normalized_name: str,
    config: PerConfig,
) -> bool:
    return normalized_name in config.generic_person_mentions


def context_window(
    text: str,
    start: int,
    end: int,
    radius: int | None = None,
    config: PerConfig = DEFAULT_PER_CONFIG,
) -> str:
    radius = config.context_radius if radius is None else radius
    return text[max(0, start - radius) : min(len(text), end + radius)].lower()


def sentence_context(
    text: str,
    start: int,
    end: int,
    config: PerConfig = DEFAULT_PER_CONFIG,
) -> tuple[str, int, int]:
    previous_boundaries = [
        text.rfind(boundary, 0, start)
        for boundary in config.sentence_boundary_chars
    ]
    previous_boundary = max(previous_boundaries)
    span_start = previous_boundary + 1 if previous_boundary >= 0 else 0

    next_boundaries = [
        boundary_index
        for boundary in config.sentence_boundary_chars
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
    return next_speaker_anchor_position_with_config(
        context,
        entity_start,
        DEFAULT_PER_CONFIG,
    )


def next_speaker_anchor_position_with_config(
    context: str,
    entity_start: int,
    config: PerConfig,
) -> int | None:
    prefix = context[:entity_start]
    positions = []

    for pattern in config.next_speaker_patterns:
        position = prefix.rfind(pattern)

        if position >= 0:
            positions.append(position)

    return max(positions) if positions else None


def infer_person_role(
    text: str,
    start: int,
    end: int,
    config: PerConfig = DEFAULT_PER_CONFIG,
) -> str | None:
    context, entity_start, _ = sentence_context(text, start, end, config)

    if (
        next_speaker_anchor_position_with_config(context, entity_start, config)
        is not None
    ):
        return "probable_next_speaker"

    return None


def next_speaker_anchor_key(
    text: str,
    start: int,
    end: int,
    config: PerConfig = DEFAULT_PER_CONFIG,
) -> tuple[int, int] | None:
    context, entity_start, _ = sentence_context(text, start, end, config)
    anchor_position = next_speaker_anchor_position_with_config(
        context,
        entity_start,
        config,
    )

    if anchor_position is None:
        return None

    return start - entity_start, anchor_position


def is_assembly_chair_turn(
    text: str,
    config: PerConfig = DEFAULT_PER_CONFIG,
) -> bool:
    normalized_text = text.lower()
    return any(
        pattern in normalized_text
        for pattern in config.assembly_chair_turn_patterns
    )


def anchor_weight(role: str, config: PerConfig = DEFAULT_PER_CONFIG) -> int:
    if role == "probable_next_speaker":
        return config.next_speaker_weight
    return config.previous_speaker_weight


def current_speaker_source(role: str) -> str:
    if role == "probable_next_speaker":
        return CURRENT_SPEAKER_SOURCE_CHAIR_NEXT_CALL
    return CURRENT_SPEAKER_SOURCE_PREVIOUS


def prediction_evidence_source(
    prediction: SpeakerPersonPrediction,
    source_turn: Turn,
    config: PerConfig = DEFAULT_PER_CONFIG,
) -> tuple[str, bool]:
    if (
        prediction.role == "probable_next_speaker"
        and is_assembly_chair_turn(source_turn.text, config)
    ):
        return CURRENT_SPEAKER_SOURCE_CHAIR_NEXT_CALL, True

    return current_speaker_source(prediction.role), False


def collect_person_mentions(
    turns: list[Turn],
    ner: NERCallable,
    threshold: float | None = None,
    config: PerConfig = DEFAULT_PER_CONFIG,
) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []
    threshold = config.per_confidence_threshold if threshold is None else threshold

    for turn in turns:
        for entity in ner(turn.text):
            if entity.get("entity_group") != "PER":
                continue

            if float(entity.get("score", 0.0)) <= threshold:
                continue

            normalized_name = normalize_name(str(entity.get("word", "")))

            if is_generic_person_mention_with_config(normalized_name, config):
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
    search_window: int | None = None,
    config: PerConfig = DEFAULT_PER_CONFIG,
) -> int | None:
    search_window = (
        config.speaker_search_window if search_window is None else search_window
    )
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
    config: PerConfig = DEFAULT_PER_CONFIG,
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
            config,
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
            config,
        )

        if role is None:
            continue

        key = next_speaker_anchor_key(
            mention["text"],
            int(entity["start"]),
            int(entity["end"]),
            config,
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
            config=config,
        )

        if predicted_turn_id is None:
            continue

        resolution = resolve_known_person(
            mention["normalized_name"],
            known_people,
            config=config,
        )
        predictions.append(
            SpeakerPersonPrediction(
                source_turn_id=source_turn_id,
                predicted_turn_id=predicted_turn_id,
                speaker_raw=str(entity.get("word", "")),
                speaker_normalized=mention["normalized_name"],
                resolution=resolution,
                role=role,
                weight=anchor_weight(role, config),
            )
        )

    return predictions


def build_mentioned_persons_by_turn(
    mentions: list[dict[str, Any]],
    known_people: list[KnownPerson],
    config: PerConfig = DEFAULT_PER_CONFIG,
) -> dict[int, list[PersonIdentity]]:
    mentioned_persons_by_turn: dict[int, list[PersonIdentity]] = defaultdict(list)
    seen_by_turn: defaultdict[int, set[tuple[str, str | None, str]]] = defaultdict(set)

    for mention in mentions:
        turn_id = int(mention["turn_id"])
        resolution = resolve_known_person(
            mention["normalized_name"],
            known_people,
            config=config,
        )
        identity_key = resolution.identity_key

        if identity_key in seen_by_turn[turn_id]:
            continue

        mentioned_persons_by_turn[turn_id].append(resolution.identity)
        seen_by_turn[turn_id].add(identity_key)

    return mentioned_persons_by_turn


def build_speaker_person_summary(
    turns: list[Turn],
    predictions: list[SpeakerPersonPrediction],
    config: PerConfig = DEFAULT_PER_CONFIG,
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
        source_turn = turns_by_id.get(prediction.source_turn_id)

        if turn is None or turn.speaker_id is None or source_turn is None:
            continue

        source, _ = prediction_evidence_source(prediction, source_turn, config)
        identities_by_key[prediction.identity_key] = prediction.resolution.identity
        matrix[turn.speaker_id][prediction.identity_key] += prediction.weight
        source_matrix[turn.speaker_id][prediction.identity_key][source] += (
            prediction.weight
        )

    speaker_id_to_person: dict[str, dict[str, float | int | str | PersonIdentity]] = {}

    for speaker_id, counts in matrix.items():
        identity_key, count = counts.most_common(1)[0]
        total = sum(counts.values())
        source_counts = source_matrix[speaker_id][identity_key]
        source = max(
            source_counts,
            key=lambda item: (
                source_counts[item],
                item == CURRENT_SPEAKER_SOURCE_CHAIR_NEXT_CALL,
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


def build_speaker_identity_evidence_by_turn(
    turns: list[Turn],
    predictions: list[SpeakerPersonPrediction],
    config: PerConfig = DEFAULT_PER_CONFIG,
) -> dict[int, list[SpeakerIdentityEvidence]]:
    evidence_by_turn: defaultdict[int, list[SpeakerIdentityEvidence]] = defaultdict(list)
    turns_by_id = {turn.turn_id: turn for turn in turns}

    for prediction in predictions:
        source_turn = turns_by_id.get(prediction.source_turn_id)
        target_turn = turns_by_id.get(prediction.predicted_turn_id)

        if source_turn is None or target_turn is None:
            continue

        source, eligible = prediction_evidence_source(
            prediction,
            source_turn,
            config,
        )
        evidence_by_turn[target_turn.turn_id].append(
            SpeakerIdentityEvidence(
                source=source,
                eligible_for_cluster_majority=eligible,
                person=prediction.resolution.identity,
                source_turn_id=source_turn.turn_id,
                target_turn_id=target_turn.turn_id,
                source_speaker_id=source_turn.speaker_id,
                target_speaker_id=target_turn.speaker_id,
                speaker_raw=prediction.speaker_raw,
                speaker_normalized=prediction.speaker_normalized,
                match_score=prediction.resolution.match_score,
                is_known_person=prediction.resolution.is_known_person,
            )
        )

    return dict(evidence_by_turn)


def choose_current_speaker_from_evidence(
    evidence: list[SpeakerIdentityEvidence],
) -> tuple[PersonIdentity, str, float]:
    matrix: Counter[tuple[str, str | None, str]] = Counter()
    source_matrix: defaultdict[tuple[str, str | None, str], Counter[str]] = defaultdict(
        Counter
    )
    identities_by_key: dict[tuple[str, str | None, str], PersonIdentity] = {}

    for item in evidence:
        identity_key = (item.person.kind, item.person.id, item.person.name)
        identities_by_key[identity_key] = item.person
        matrix[identity_key] += 1
        source_matrix[identity_key][item.source] += 1

    identity_key, count = matrix.most_common(1)[0]
    total = sum(matrix.values())
    source_counts = source_matrix[identity_key]
    source = source_counts.most_common(1)[0][0]

    return identities_by_key[identity_key], source, count / total


def build_turn_analysis(
    turns: list[Turn],
    speaker_id_to_person: dict[str, dict[str, float | int | str | PersonIdentity]],
    mentioned_persons_by_turn: dict[int, list[PersonIdentity]],
    speaker_identity_evidence_by_turn: dict[int, list[SpeakerIdentityEvidence]]
    | None = None,
    config: PerConfig = DEFAULT_PER_CONFIG,
) -> list[TurnAnalysis]:
    turns_analysis: list[TurnAnalysis] = []
    speaker_identity_evidence_by_turn = speaker_identity_evidence_by_turn or {}

    for turn in turns:
        mentioned_persons = mentioned_persons_by_turn.get(turn.turn_id, [])
        speaker_identity_evidence = speaker_identity_evidence_by_turn.get(
            turn.turn_id,
            [],
        )
        hardcoded_current_speaker = (
            config.assembly_chair_identity
            if is_assembly_chair_turn(turn.text, config)
            else None
        )
        speaker_data = (
            speaker_id_to_person.get(turn.speaker_id)
            if turn.speaker_id is not None
            else None
        )

        if hardcoded_current_speaker is not None:
            hardcoded_evidence = [
                SpeakerIdentityEvidence(
                    source=CURRENT_SPEAKER_SOURCE_ASSEMBLY_CHAIR,
                    eligible_for_cluster_majority=True,
                    person=hardcoded_current_speaker,
                    source_turn_id=turn.turn_id,
                    target_turn_id=turn.turn_id,
                    source_speaker_id=turn.speaker_id,
                    target_speaker_id=turn.speaker_id,
                    speaker_raw=hardcoded_current_speaker.name,
                    speaker_normalized=normalize_name(hardcoded_current_speaker.name),
                    match_score=100.0,
                    is_known_person=True,
                )
            ]
            turns_analysis.append(
                TurnAnalysis(
                    turn_id=turn.turn_id,
                    current_speaker=hardcoded_current_speaker,
                    current_speaker_source=CURRENT_SPEAKER_SOURCE_ASSEMBLY_CHAIR,
                    current_speaker_purity=1.0,
                    speaker_identity_evidence=hardcoded_evidence,
                    mentioned_persons=mentioned_persons,
                )
            )
            continue

        if speaker_identity_evidence:
            person, source, purity = choose_current_speaker_from_evidence(
                speaker_identity_evidence
            )
            turns_analysis.append(
                TurnAnalysis(
                    turn_id=turn.turn_id,
                    current_speaker=person,
                    current_speaker_source=source,
                    current_speaker_purity=purity,
                    speaker_identity_evidence=speaker_identity_evidence,
                    mentioned_persons=mentioned_persons,
                )
            )
            continue

        if speaker_data is None:
            turns_analysis.append(
                TurnAnalysis(
                    turn_id=turn.turn_id,
                    speaker_identity_evidence=speaker_identity_evidence,
                    mentioned_persons=mentioned_persons,
                )
            )
            continue

        turns_analysis.append(
            TurnAnalysis(
                turn_id=turn.turn_id,
                current_speaker=speaker_data["person"],  # type: ignore[arg-type]
                current_speaker_source=CURRENT_SPEAKER_SOURCE_PROPAGATED,
                current_speaker_purity=float(speaker_data["purity"]),
                speaker_identity_evidence=speaker_identity_evidence,
                mentioned_persons=mentioned_persons,
            )
        )

    return turns_analysis
