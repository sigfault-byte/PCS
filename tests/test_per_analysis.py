import unittest

from assemblybot.models.turn_document import PersonIdentity
from assemblybot.per_config import PerConfig
from assemblybot.stages.per_analysis import (
    ASSEMBLY_CHAIR_IDENTITY,
    CURRENT_SPEAKER_SOURCE_ASSEMBLY_CHAIR,
    CURRENT_SPEAKER_SOURCE_NEXT,
    CURRENT_SPEAKER_SOURCE_PREVIOUS,
    CURRENT_SPEAKER_SOURCE_PROPAGATED,
    SpeakerPersonPrediction,
    build_mentioned_persons_by_turn,
    build_speaker_person_summary,
    build_speaker_identity_evidence_by_turn,
    build_turn_analysis,
    collect_person_mentions,
    find_predicted_turn_id,
    infer_person_role,
    is_assembly_chair_turn,
    is_generic_person_mention,
    current_speaker_source,
    predict_person_turns,
)
from assemblybot.stages.per_identity import resolve_known_person
from tests.per_test_helpers import known_person, make_turn


class PerAnalysisTest(unittest.TestCase):
    def test_infer_person_role(self) -> None:
        next_text = "La parole est à Madame Alice Dupont."
        previous_text = "Merci Madame Alice Dupont."
        mentioned_text = "Alice Dupont était présente."

        self.assertEqual(
            infer_person_role(next_text, next_text.index("Alice"), len(next_text)),
            "probable_next_speaker",
        )
        self.assertIsNone(
            infer_person_role(
                previous_text,
                previous_text.index("Alice"),
                len(previous_text),
            )
        )
        self.assertIsNone(
            infer_person_role(
                mentioned_text,
                mentioned_text.index("Alice"),
                len(mentioned_text),
            )
        )

    def test_sentence_boundary_blocks_previous_anchor(self) -> None:
        text = "Merci madame la présidente. Monsieur Alice Dupont prend la parole."

        self.assertIsNone(
            infer_person_role(
                text,
                text.index("Alice"),
                text.index("Dupont") + len("Dupont"),
            )
        )

    def test_same_sentence_merci_does_not_anchor_previous_speaker(self) -> None:
        text = "Merci Monsieur Alice Dupont."

        self.assertIsNone(
            infer_person_role(
                text,
                text.index("Alice"),
                text.index("Dupont") + len("Dupont"),
            )
        )

    def test_same_sentence_next_anchor_still_matches(self) -> None:
        text = "La parole est à Monsieur Laurent Marcangeli."

        self.assertEqual(
            infer_person_role(
                text,
                text.index("Laurent"),
                text.index("Marcangeli") + len("Marcangeli"),
            ),
            "probable_next_speaker",
        )

    def test_generic_person_mention_detection(self) -> None:
        self.assertTrue(is_generic_person_mention("monsieur le ministre"))
        self.assertTrue(is_generic_person_mention("madame la presidente"))
        self.assertFalse(is_generic_person_mention("laurent marcangeli"))

    def test_current_speaker_source_from_role(self) -> None:
        self.assertEqual(
            current_speaker_source("probable_next_speaker"),
            CURRENT_SPEAKER_SOURCE_NEXT,
        )
        self.assertEqual(
            current_speaker_source("probable_previous_speaker"),
            CURRENT_SPEAKER_SOURCE_PREVIOUS,
        )

    def test_assembly_chair_turn_detection(self) -> None:
        self.assertTrue(
            is_assembly_chair_turn("La parole est à Monsieur Laurent Nunez.")
        )
        self.assertTrue(is_assembly_chair_turn("La séance est levée."))
        self.assertFalse(is_assembly_chair_turn("Bonjour à toutes et à tous."))

    def test_speaker_matrix_and_turn_analysis(self) -> None:
        turns = [
            make_turn(1, "CHAIR", "La parole est à Alice Dupont."),
            make_turn(2, "S_A", "Bonjour."),
            make_turn(3, "CHAIR", "Merci Alice Dupont."),
        ]
        people = [known_person("123", "Alice Dupont", "Député", "deputy")]
        predictions = predict_person_turns(
            turns,
            [
                {
                    "entity": {
                        "word": "Alice Dupont",
                        "start": 15,
                        "end": 27,
                    },
                    "normalized_name": "alice dupont",
                    "text": turns[0].text,
                    "turn_id": 1,
                },
                {
                    "entity": {
                        "word": "Alice Dupont",
                        "start": 6,
                        "end": 18,
                    },
                    "normalized_name": "alice dupont",
                    "text": turns[2].text,
                    "turn_id": 3,
                },
            ],
            people,
        )

        summary = build_speaker_person_summary(turns, predictions)
        self.assertEqual(
            summary["S_A"]["person"],
            PersonIdentity(
                id="123",
                name="Alice Dupont",
                role="Député",
                kind="deputy",
            ),
        )
        self.assertEqual(summary["S_A"]["count"], 2)
        self.assertEqual(summary["S_A"]["total"], 2)
        self.assertEqual(summary["S_A"]["purity"], 1.0)
        self.assertEqual(summary["S_A"]["source"], CURRENT_SPEAKER_SOURCE_NEXT)

        alice = PersonIdentity(
            id="123",
            name="Alice Dupont",
            role="Député",
            kind="deputy",
        )
        analyses = build_turn_analysis(
            turns,
            summary,
            {
                1: [alice],
                3: [alice],
            },
        )
        self.assertEqual([analysis.turn_id for analysis in analyses], [1, 2, 3])
        self.assertEqual(analyses[0].current_speaker, ASSEMBLY_CHAIR_IDENTITY)
        self.assertEqual(
            analyses[0].current_speaker_source,
            CURRENT_SPEAKER_SOURCE_ASSEMBLY_CHAIR,
        )
        self.assertEqual(analyses[0].current_speaker_purity, 1.0)
        self.assertEqual(analyses[0].mentioned_persons, [alice])
        self.assertEqual(analyses[1].current_speaker, alice)
        self.assertEqual(
            analyses[1].current_speaker_source,
            CURRENT_SPEAKER_SOURCE_PROPAGATED,
        )
        self.assertEqual(analyses[1].current_speaker_purity, 1.0)
        self.assertEqual(analyses[1].mentioned_persons, [])

        evidence_by_turn = build_speaker_identity_evidence_by_turn(turns, predictions)
        analyses_with_evidence = build_turn_analysis(
            turns,
            summary,
            {
                1: [alice],
                3: [alice],
            },
            evidence_by_turn,
        )
        self.assertEqual(
            analyses_with_evidence[1].current_speaker_source,
            CURRENT_SPEAKER_SOURCE_NEXT,
        )
        self.assertEqual(
            analyses_with_evidence[1].speaker_identity_evidence[0].source,
            CURRENT_SPEAKER_SOURCE_NEXT,
        )
        self.assertTrue(
            analyses_with_evidence[1]
            .speaker_identity_evidence[0]
            .eligible_for_cluster_majority
        )

    def test_adjacent_turn_fallback_matches_experiment(self) -> None:
        turns = [
            make_turn(1, "CHAIR", "La parole est à Alice Dupont."),
            make_turn(2, "CHAIR", "Suite de la presidence."),
            make_turn(3, "CHAIR", "Encore la presidence."),
        ]
        turns_by_id = {turn.turn_id: turn for turn in turns}

        self.assertEqual(
            find_predicted_turn_id(
                turns_by_id,
                source_turn_id=1,
                role="probable_next_speaker",
            ),
            2,
        )

    def test_assembly_chair_hardcoded_identity_overrides_speaker_matrix(self) -> None:
        turns = [
            make_turn(1, "CHAIR", "La parole est à Alice Dupont."),
        ]
        wrong_identity = PersonIdentity(
            id="123",
            name="Alice Dupont",
            role="Député",
            kind="deputy",
        )

        analyses = build_turn_analysis(
            turns,
            {
                "CHAIR": {
                    "person": wrong_identity,
                    "count": 1,
                    "total": 1,
                    "purity": 1.0,
                }
            },
            {},
        )

        self.assertEqual(analyses[0].current_speaker, ASSEMBLY_CHAIR_IDENTITY)
        self.assertEqual(
            analyses[0].current_speaker_source,
            CURRENT_SPEAKER_SOURCE_ASSEMBLY_CHAIR,
        )
        self.assertEqual(analyses[0].current_speaker_purity, 1.0)

    def test_only_first_per_after_next_speaker_anchor_is_predicted(self) -> None:
        turns = [
            make_turn(1, "CHAIR", "La parole est à Alice Dupont avec Bob Martin."),
            make_turn(2, "S_A", "Bonjour."),
        ]
        alice_start = turns[0].text.index("Alice")
        bob_start = turns[0].text.index("Bob")
        people = [
            known_person("123", "Alice Dupont", "Député", "deputy"),
            known_person("456", "Bob Martin", "Député", "deputy"),
        ]

        predictions = predict_person_turns(
            turns,
            [
                {
                    "entity": {
                        "word": "Alice Dupont",
                        "start": alice_start,
                        "end": alice_start + len("Alice Dupont"),
                    },
                    "normalized_name": "alice dupont",
                    "text": turns[0].text,
                    "turn_id": 1,
                },
                {
                    "entity": {
                        "word": "Bob Martin",
                        "start": bob_start,
                        "end": bob_start + len("Bob Martin"),
                    },
                    "normalized_name": "bob martin",
                    "text": turns[0].text,
                    "turn_id": 1,
                },
            ],
            people,
        )

        summary = build_speaker_person_summary(turns, predictions)

        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0].speaker_normalized, "alice dupont")
        self.assertEqual(
            summary["S_A"]["person"],
            PersonIdentity(
                id="123",
                name="Alice Dupont",
                role="Député",
                kind="deputy",
            ),
        )
        self.assertEqual(summary["S_A"]["count"], 2)
        self.assertEqual(summary["S_A"]["total"], 2)
        self.assertEqual(summary["S_A"]["purity"], 1.0)
        self.assertEqual(summary["S_A"]["source"], CURRENT_SPEAKER_SOURCE_NEXT)

    def test_merci_does_not_predict_previous_speaker(self) -> None:
        turns = [
            make_turn(1, "S_A", "Bonjour."),
            make_turn(2, "CHAIR", "Merci Alice Dupont."),
        ]
        start = turns[1].text.index("Alice")

        self.assertEqual(
            predict_person_turns(
                turns,
                [
                    {
                        "entity": {
                            "word": "Alice Dupont",
                            "start": start,
                            "end": start + len("Alice Dupont"),
                        },
                        "normalized_name": "alice dupont",
                        "text": turns[1].text,
                        "turn_id": 2,
                    }
                ],
                [known_person("123", "Alice Dupont", "Député", "deputy")],
            ),
            [],
        )

    def test_mentioned_persons_deduplicate_and_resolve(self) -> None:
        mentions = [
            {
                "normalized_name": "alice dupont",
                "turn_id": 1,
            },
            {
                "normalized_name": "alice dupond",
                "turn_id": 1,
            },
            {
                "normalized_name": "unknown person",
                "turn_id": 1,
            },
        ]
        alice = PersonIdentity(
            id="123",
            name="Alice Dupont",
            role="Député",
            kind="deputy",
        )

        self.assertEqual(
            build_mentioned_persons_by_turn(
                mentions,
                [known_person("123", "Alice Dupont", "Député", "deputy")],
            ),
            {
                1: [
                    alice,
                    PersonIdentity(
                        id=None,
                        name="unknown person",
                        role=None,
                        kind="raw_per",
                    ),
                ]
            },
        )

    def test_single_token_false_positive_stays_raw_mention(self) -> None:
        mentions = [
            {
                "normalized_name": "fleuristes",
                "turn_id": 52,
            }
        ]

        self.assertEqual(
            build_mentioned_persons_by_turn(
                mentions,
                [known_person("456", "Sandrine Le Feur", "Député", "deputy")],
            ),
            {
                52: [
                    PersonIdentity(
                        id=None,
                        name="fleuristes",
                        role=None,
                        kind="raw_per",
                    )
                ]
            },
        )

    def test_collect_person_mentions_threshold(self) -> None:
        turns = [make_turn(1, "CHAIR", "Alice Dupont")]

        def ner(_: str) -> list[dict[str, object]]:
            return [
                {
                    "entity_group": "PER",
                    "score": 0.79,
                    "word": "Low Score",
                    "start": 0,
                    "end": 9,
                },
                {
                    "entity_group": "PER",
                    "score": 0.99,
                    "word": "Alice Dupont",
                    "start": 0,
                    "end": 12,
                },
            ]

        mentions = collect_person_mentions(turns, ner)
        self.assertEqual(len(mentions), 1)
        self.assertEqual(mentions[0]["normalized_name"], "alice dupont")

    def test_custom_per_confidence_threshold_changes_mentions(self) -> None:
        turns = [make_turn(1, "CHAIR", "Alice Dupont")]

        def ner(_: str) -> list[dict[str, object]]:
            return [
                {
                    "entity_group": "PER",
                    "score": 0.9,
                    "word": "Alice Dupont",
                    "start": 0,
                    "end": 12,
                }
            ]

        self.assertEqual(len(collect_person_mentions(turns, ner)), 1)
        self.assertEqual(
            collect_person_mentions(
                turns,
                ner,
                config=PerConfig(per_confidence_threshold=0.95),
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
