import argparse
from dataclasses import dataclass, field
from pathlib import Path

from assemblybot.helper.directory import build_default_output_path
from assemblybot.helper.document import load_document, save_turn_document
from assemblybot.models.document import CanonicalDocument
from assemblybot.models.flags import SegmentFlag
from assemblybot.models.time import TimeRange
from assemblybot.models.turn_document import Turn, TurnDocument

MAX_TURN_SILENCE_SECONDS = 5.0


@dataclass
class _TurnBuilderState:
    text_parts: list[str] = field(default_factory=list)
    transcript_segment_ids: list[int] = field(default_factory=list)
    diarization_segment_ids: list[int] = field(default_factory=list)
    speaker_id: str | None = None
    speaker_confidence: float = 0.0
    start_seconds: float | None = None
    end_seconds: float | None = None
    flags: SegmentFlag = SegmentFlag.NONE

    @property
    def is_empty(self) -> bool:
        return self.start_seconds is None or self.end_seconds is None

    @classmethod
    def from_alignment(
        cls,
        *,
        text: str,
        speaker_id: str | None,
        speaker_confidence: float,
        transcript_segment_id: int,
        diarization_segment_ids: list[int],
        start_seconds: float,
        end_seconds: float,
        flags: SegmentFlag,
    ) -> "_TurnBuilderState":
        return cls(
            text_parts=[text],
            transcript_segment_ids=[transcript_segment_id],
            diarization_segment_ids=list(diarization_segment_ids),
            speaker_id=speaker_id,
            speaker_confidence=speaker_confidence,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            flags=flags,
        )

    def can_merge(self, *, speaker_id: str | None, start_seconds: float) -> bool:
        return (
            self.speaker_id == speaker_id
            and speaker_id is not None
            and self.end_seconds is not None
            and start_seconds - self.end_seconds <= MAX_TURN_SILENCE_SECONDS
        )

    def merge_alignment(
        self,
        *,
        text: str,
        speaker_confidence: float,
        transcript_segment_id: int,
        diarization_segment_ids: list[int],
        start_seconds: float,
        end_seconds: float,
        flags: SegmentFlag,
    ) -> None:
        if self.start_seconds is None or self.end_seconds is None:
            raise ValueError("Cannot merge into an empty turn state.")

        self.text_parts.append(text)
        self.transcript_segment_ids.append(transcript_segment_id)
        for diarization_segment_id in diarization_segment_ids:
            if diarization_segment_id not in self.diarization_segment_ids:
                self.diarization_segment_ids.append(diarization_segment_id)

        self.speaker_confidence = speaker_confidence
        self.start_seconds = min(self.start_seconds, start_seconds)
        self.end_seconds = max(self.end_seconds, end_seconds)
        self.flags |= flags

    def to_turn(self, turn_id: int) -> Turn:
        if self.start_seconds is None or self.end_seconds is None:
            raise ValueError("Cannot create a turn from an empty turn state.")

        return Turn(
            turn_id=turn_id,
            audio_time=TimeRange.from_seconds(
                self.start_seconds,
                self.end_seconds,
            ),
            text=" ".join(self.text_parts).strip(),
            speaker_id=self.speaker_id,
            speaker_confidence=self.speaker_confidence,
            transcript_segment_ids=self.transcript_segment_ids,
            diarization_segment_ids=self.diarization_segment_ids,
            flags=self.flags,
        )


def consolidate_turns(document: CanonicalDocument) -> list[Turn]:
    alignments = document.alignment.transcript_diarization_matches
    transcript_by_id = {
        segment.segment_id: segment for segment in document.transcript.raw_segments
    }
    diarization_by_id = {
        segment.segment_id: segment for segment in document.diarization.raw_segments
    }

    turns: list[Turn] = []
    current_turn = _TurnBuilderState()

    for alignment in alignments:
        transcript_segment = transcript_by_id[alignment.transcript_segment_id]
        diarization_segments = [
            diarization_by_id[segment_id]
            for segment_id in alignment.diarization_segment_ids
        ]

        alignment_start_seconds = min(
            segment.time.start_seconds for segment in diarization_segments
        )
        alignment_end_seconds = max(
            segment.time.end_seconds for segment in diarization_segments
        )
        alignment_speaker_id = alignment.probable_speaker_id
        alignment_speaker_confidence = alignment.speaker_confidence or 0.0
        alignment_flags = SegmentFlag(alignment.flags) | transcript_segment.flags

        for diarization_segment in diarization_segments:
            alignment_flags |= diarization_segment.flags

        if current_turn.can_merge(
            speaker_id=alignment_speaker_id,
            start_seconds=alignment_start_seconds,
        ):
            current_turn.merge_alignment(
                text=transcript_segment.raw_text,
                speaker_confidence=alignment_speaker_confidence,
                transcript_segment_id=alignment.transcript_segment_id,
                diarization_segment_ids=alignment.diarization_segment_ids,
                start_seconds=alignment_start_seconds,
                end_seconds=alignment_end_seconds,
                flags=alignment_flags,
            )
            continue

        if not current_turn.is_empty:
            turns.append(current_turn.to_turn(turn_id=len(turns) + 1))

        current_turn = _TurnBuilderState.from_alignment(
            text=transcript_segment.raw_text,
            speaker_id=alignment_speaker_id,
            speaker_confidence=alignment_speaker_confidence,
            transcript_segment_id=alignment.transcript_segment_id,
            diarization_segment_ids=alignment.diarization_segment_ids,
            start_seconds=alignment_start_seconds,
            end_seconds=alignment_end_seconds,
            flags=alignment_flags,
        )

    if not current_turn.is_empty:
        turns.append(current_turn.to_turn(turn_id=len(turns) + 1))

    return turns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Creates a new JSON document from the final stage alignment data.")
    )
    parser.add_argument(
        "--input-json",
        required=True,
        help="Existing canonical document JSON to load the alignment segments from",
    )
    parser.add_argument(
        "--output-json",
        help="Optional output JSON path (default: generated in interim directory)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_json_path = Path(args.input_json).resolve()
    document = load_document(input_json_path)

    output_json_path = (
        Path(args.output_json).resolve()
        if args.output_json
        else build_default_output_path(
            Path(document.source.input_path or input_json_path),
            "_01_turns",
            "json",
        )
    )

    turns = consolidate_turns(document)

    turn_document = TurnDocument(
        turns=turns,
        turns_analysis=[],
    )

    # STEP 3 -- Save document.

    save_turn_document(turn_document, output_json_path)


if __name__ == "__main__":
    main()
