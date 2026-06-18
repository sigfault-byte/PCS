import contextlib
import io
import unittest

from assemblybot.models.turn_document import PersonIdentity, TurnDocument
from assemblybot.stages.PER_extraction import enrich_turn_document, parse_args
from assemblybot.stages.per_analysis import CURRENT_SPEAKER_SOURCE_NEXT
from tests.per_test_helpers import known_person, make_turn


class PerExtractionStageTest(unittest.TestCase):
    def test_enrich_turn_document_orchestrates_analysis(self) -> None:
        turns = [
            make_turn(1, "CHAIR", "La parole est à Madame Alice Dupont."),
            make_turn(2, "S_A", "Bonjour."),
        ]

        def ner(text: str) -> list[dict[str, object]]:
            if "Alice Dupont" not in text:
                return []

            start = text.index("Alice")
            return [
                {
                    "entity_group": "PER",
                    "score": 0.99,
                    "word": "Alice Dupont",
                    "start": start,
                    "end": start + len("Alice Dupont"),
                }
            ]

        alice = PersonIdentity(
            id="123",
            name="Alice Dupont",
            role="Député",
            kind="deputy",
        )
        document = TurnDocument(turns=turns, turns_analysis=[])
        enriched = enrich_turn_document(
            document,
            [known_person("123", "Alice Dupont", "Député", "deputy")],
            ner,
        )

        self.assertIs(enriched.turns, turns)
        self.assertEqual(enriched.turns_analysis[1].current_speaker, alice)
        self.assertEqual(
            enriched.turns_analysis[1].current_speaker_source,
            CURRENT_SPEAKER_SOURCE_NEXT,
        )

    def test_generic_title_does_not_drift_next_speaker_anchor(self) -> None:
        turns = [
            make_turn(
                1,
                "CHAIR",
                "Merci beaucoup Monsieur le Ministre. La parole est à présent à "
                "Monsieur Laurent Marcangeli, président du groupe Horizon et "
                "Indépendant. Monsieur le Président.",
            ),
            make_turn(2, "S_LAURENT", "Merci Madame la Présidente."),
        ]
        minister_start = turns[0].text.index("Monsieur le Ministre")
        laurent_start = turns[0].text.index("Laurent Marcangeli")

        def ner(text: str) -> list[dict[str, object]]:
            if "Laurent Marcangeli" not in text:
                return []

            return [
                {
                    "entity_group": "PER",
                    "score": 0.99,
                    "word": "Monsieur le Ministre",
                    "start": minister_start,
                    "end": minister_start + len("Monsieur le Ministre"),
                },
                {
                    "entity_group": "PER",
                    "score": 0.99,
                    "word": "Laurent Marcangeli",
                    "start": laurent_start,
                    "end": laurent_start + len("Laurent Marcangeli"),
                },
            ]

        laurent = PersonIdentity(
            id="605782",
            name="Laurent Marcangeli",
            role="Député",
            kind="deputy",
        )
        marine = known_person("720614", "Marine Le Pen", "Député", "deputy")
        enriched = enrich_turn_document(
            TurnDocument(turns=turns, turns_analysis=[]),
            [
                marine,
                known_person("605782", "Laurent Marcangeli", "Député", "deputy"),
            ],
            ner,
        )

        self.assertEqual(enriched.turns_analysis[0].mentioned_persons, [laurent])
        self.assertEqual(enriched.turns_analysis[1].current_speaker, laurent)
        self.assertEqual(
            enriched.turns_analysis[1].current_speaker_source,
            CURRENT_SPEAKER_SOURCE_NEXT,
        )
        self.assertNotEqual(
            enriched.turns_analysis[1].current_speaker,
            marine.identity,
        )

    def test_parse_args_requires_minister_ground_truth(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--input-json",
                        "turns.json",
                        "--csv-ground-truth-PER",
                        "deputies.csv",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
