from pathlib import Path

from assemblybot.models.time import TimeRange
from assemblybot.models.turn_document import PersonIdentity, Turn
from assemblybot.stages.per_identity import GROUND_TRUTH_HEADERS, KnownPerson, normalize_name


def make_turn(turn_id: int, speaker_id: str, text: str) -> Turn:
    return Turn(
        turn_id=turn_id,
        audio_time=TimeRange.from_seconds(turn_id, turn_id + 1),
        text=text,
        speaker_id=speaker_id,
        speaker_confidence=1.0,
        transcript_segment_ids=[],
        diarization_segment_ids=[],
    )


def write_ground_truth_csv(path: Path, rows: list[list[str]]) -> None:
    lines = [",".join(f'"{value}"' for value in GROUND_TRUTH_HEADERS)]
    lines.extend(",".join(f'"{value}"' for value in row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def known_person(
    identifier: str,
    name: str,
    role: str | None,
    kind: str,
) -> KnownPerson:
    return KnownPerson(
        identity=PersonIdentity(
            id=identifier,
            name=name,
            role=role,
            kind=kind,
        ),
        normalized_name=normalize_name(name),
    )
