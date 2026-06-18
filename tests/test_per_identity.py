import tempfile
import unittest
from pathlib import Path

from assemblybot.models.turn_document import PersonIdentity
from assemblybot.stages.per_identity import (
    GROUND_TRUTH_HEADERS,
    load_known_people,
    normalize_name,
    person_kind_from_identifier,
    resolve_known_person,
    validate_ground_truth_csv,
)
from tests.per_test_helpers import known_person, write_ground_truth_csv


class PerIdentityTest(unittest.TestCase):
    def test_normalize_name(self) -> None:
        self.assertEqual(
            normalize_name("  Élodie   Députée  "),
            "elodie deputee",
        )

    def test_load_deputy_ground_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "deputies.csv"
            write_ground_truth_csv(
                csv_path,
                [
                    [
                        "123",
                        "Élodie",
                        "Durand",
                        "",
                        "",
                        "",
                        "Avocate",
                        "Socialistes et apparentés",
                        "SOC",
                    ]
                ],
            )

            self.assertEqual(
                load_known_people(csv_path, source_kind="deputy"),
                [
                    known_person(
                        "123",
                        "Élodie Durand",
                        "Député",
                        "deputy",
                    )
                ],
            )

    def test_load_minister_and_assembly_chair_ground_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "ministers.csv"
            write_ground_truth_csv(
                csv_path,
                [
                    [
                        "minister:laurent_nunez",
                        "Laurent",
                        "Nunez",
                        "",
                        "",
                        "",
                        "Ministre de l'interieur",
                        "Gouvernement",
                        "GOUV",
                    ],
                    [
                        "assembly_chari:yael-braun-pivet",
                        "Yaël",
                        "Braun-Pivet",
                        "",
                        "",
                        "",
                        "Présidente de l'Assemblée nationale",
                        "Présidente de l'Assemblée nationale",
                        "",
                    ],
                ],
            )

            self.assertEqual(
                load_known_people(csv_path, source_kind="minister"),
                [
                    known_person(
                        "minister:laurent_nunez",
                        "Laurent Nunez",
                        "Ministre de l'interieur",
                        "minister",
                    ),
                    known_person(
                        "assembly_chari:yael-braun-pivet",
                        "Yaël Braun-Pivet",
                        "Présidente de l'Assemblée nationale",
                        "assembly_chair",
                    ),
                ],
            )

    def test_assembly_chair_typo_variants_are_supported(self) -> None:
        self.assertEqual(
            person_kind_from_identifier("assembly_chair:yael-braun-pivet"),
            "assembly_chair",
        )
        self.assertEqual(
            person_kind_from_identifier("assembly_chari:yael-braun-pivet"),
            "assembly_chair",
        )
        self.assertEqual(
            person_kind_from_identifier("assembly_chaii:yael-braun-pivet"),
            "assembly_chair",
        )

    def test_malformed_ground_truth_csv_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "bad.csv"
            csv_path.write_text(
                ",".join(f'"{value}"' for value in GROUND_TRUTH_HEADERS)
                + '\n"minister:bad","Bad"\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "malformed rows"):
                validate_ground_truth_csv(csv_path)

    def test_resolve_known_person(self) -> None:
        people = [
            known_person("123", "Alice Dupont", "Député", "deputy"),
        ]

        resolved = resolve_known_person("alice dupond", people)
        self.assertTrue(resolved.is_known_person)
        self.assertEqual(
            resolved.identity,
            PersonIdentity(
                id="123",
                name="Alice Dupont",
                role="Député",
                kind="deputy",
            ),
        )

        unresolved = resolve_known_person("zzzzzz", people)
        self.assertFalse(unresolved.is_known_person)
        self.assertEqual(
            unresolved.identity,
            PersonIdentity(
                id=None,
                name="zzzzzz",
                role=None,
                kind="raw_per",
            ),
        )

    def test_single_token_false_positive_stays_raw(self) -> None:
        people = [
            known_person("456", "Sandrine Le Feur", "Député", "deputy"),
        ]

        resolved = resolve_known_person("fleuristes", people)

        self.assertFalse(resolved.is_known_person)
        self.assertEqual(
            resolved.identity,
            PersonIdentity(
                id=None,
                name="fleuristes",
                role=None,
                kind="raw_per",
            ),
        )

    def test_multi_token_name_resolves(self) -> None:
        people = [
            known_person("720614", "Laurent Marcangeli", "Député", "deputy"),
        ]

        resolved = resolve_known_person("laurent marcangeli", people)

        self.assertTrue(resolved.is_known_person)
        self.assertEqual(resolved.identity.id, "720614")

    def test_close_multi_token_spelling_variant_resolves(self) -> None:
        people = [
            known_person("720614", "Laurent Marcangeli", "Député", "deputy"),
        ]

        resolved = resolve_known_person("laurent marcangelli", people)

        self.assertTrue(resolved.is_known_person)
        self.assertEqual(resolved.identity.id, "720614")

    def test_noisy_multi_token_name_resolves_with_exact_first_name(self) -> None:
        people = [
            known_person("795950", "Bernard Chaix", "Député", "deputy"),
        ]

        resolved = resolve_known_person("bernard chex", people)

        self.assertTrue(resolved.is_known_person)
        self.assertEqual(
            resolved.identity,
            PersonIdentity(
                id="795950",
                name="Bernard Chaix",
                role="Député",
                kind="deputy",
            ),
        )


if __name__ == "__main__":
    unittest.main()
