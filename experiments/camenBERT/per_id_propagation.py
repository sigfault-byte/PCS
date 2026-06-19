import csv
import json
import unicodedata
from collections import Counter, defaultdict

from rapidfuzz import fuzz, process
from transformers import pipeline

file = "data/interim/1ere-seance--questions-au-gouvernement--simplification-de-la-vie-economique-cmp--renforcer-la-s-14-avril-2026_02_per_extraction.json"

THRESHOLD = 0.8
THRESHOLD2 = 80

NEXT_SPEAKER_PATTERNS = [
    "la parole est à",
    "je donne la parole à",
    "vous avez la parole",
    "est à vous",
]

PREVIOUS_SPEAKER_PATTERNS = [
    "merci",
    "merci madame",
    "merci monsieur",
]

ner = pipeline(
    "token-classification",
    model="Jean-Baptiste/camembert-ner",
    aggregation_strategy="simple",
)

with open(file, "r", encoding="utf-8") as f:
    turns = json.load(f)["turns"]


def normalize_name(name: str) -> str:
    name = name.lower().strip()
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = " ".join(name.split())
    return name


deputies = []


def resolve_deputy_name(normalized_name: str, threshold: int = THRESHOLD2):
    match, score, _ = process.extractOne(
        normalized_name,
        known_names,
        # scorer=fuzz.token_sort_ratio,
        scorer=fuzz.WRatio,
    )

    if score >= threshold:
        deputy = name_to_deputy[match]
        return {
            "speaker_resolved": deputy["normalized_name"],
            "is_deputy": True,
            "deputy_id": deputy["id"],
            "match_score": score,
        }

    return {
        "speaker_resolved": normalized_name,
        "is_deputy": False,
        "deputy_id": None,
        "match_score": score,
    }


with open("docs/liste_deputes_libre_office_2026-06.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        full_name = f"{row['Prénom']} {row['Nom']}"
        deputies.append(
            {
                "id": row["identifiant"],
                "full_name": full_name,
                "normalized_name": normalize_name(full_name),
                "group": row["Groupe politique (abrégé)"],
            }
        )

known_names = [d["normalized_name"] for d in deputies]
name_to_deputy = {d["normalized_name"]: d for d in deputies}


def context_window(text, start, end, radius=80):
    return text[max(0, start - radius) : min(len(text), end + radius)].lower()


entities = []
for turn in turns:
    for ent in ner(turn["text"]):
        if ent["entity_group"] == "PER" and ent["score"] > THRESHOLD:
            entities.append(
                {
                    "entity": ent,
                    "normalized_name": normalize_name(ent["word"]),
                    "text": turn["text"],
                    "id": turn["turn_id"],
                }
            )
        if turn["speaker_id"] == "SPEAKER_55":
            print("----SPEAKER_55-----")
            print(f"Speaker_id = {turn['speaker_id']}")
            print(
                entities[-2]["entity"],
                entities[-2]["normalized_name"],
                entities[-2]["id"],
            )
            print(
                entities[-1]["entity"],
                entities[-1]["normalized_name"],
                entities[-1]["id"],
            )


previous_speaker = 0
next_speaker = 0
turn_to_infer_speaker = []

for item in entities:
    ent = item["entity"]
    text = item["text"]
    turn = item["id"]

    ctx = context_window(
        text,
        ent["start"],
        ent["end"],
    )
    if any(p in ctx for p in NEXT_SPEAKER_PATTERNS):
        next_speaker += 1
        role = "probable_next_speaker"
    elif any(p in ctx for p in PREVIOUS_SPEAKER_PATTERNS):
        previous_speaker += 1
        role = "probable_previous_speaker"
    else:
        continue

    item["role"] = role
    turn_to_infer_speaker.append(item)

turn_prediction = []
turns_by_id = {turn["turn_id"]: turn for turn in turns}

for i in turn_to_infer_speaker:
    turn_id = i["id"]
    role = i["role"]

    current_turn = turns_by_id[turn_id]
    current_speaker = current_turn["speaker_id"]

    predicted_turn_id = None

    if role == "probable_next_speaker":
        for offset in range(1, 6):
            candidate = turns_by_id.get(turn_id + offset)

            if candidate is None:
                continue

            if candidate["speaker_id"] != current_speaker:
                predicted_turn_id = candidate["turn_id"]
                break

            predicted_turn_id = turn_id + 1

    elif role == "probable_previous_speaker":
        for offset in range(1, 6):
            candidate = turns_by_id.get(turn_id - offset)

            if candidate is None:
                continue

            if candidate["speaker_id"] != current_speaker:
                predicted_turn_id = candidate["turn_id"]
                break

            predicted_turn_id = turn_id - 1

    else:
        continue

    if predicted_turn_id is None:
        continue

    resolved = resolve_deputy_name(i["normalized_name"])

    turn_prediction.append(
        {
            "source_turn_id": turn_id,
            "predicted_turn_id": predicted_turn_id,
            "speaker_raw": i["entity"]["word"],
            "speaker_normalized": i["normalized_name"],
            **resolved,
            "role": role,
        }
    )


matrix = defaultdict(Counter)

turns_by_id = {turn["turn_id"]: turn for turn in turns}

for pred in turn_prediction:
    turn_id = pred["predicted_turn_id"]

    if pred["is_deputy"]:
        person = f"deputy:{pred['deputy_id']}:{pred['speaker_resolved']}"
    else:
        person = f"raw_per:{pred['speaker_resolved']}"

    turn = turns_by_id.get(turn_id)

    if turn is None:
        continue

    speaker_id = turn["speaker_id"]
    matrix[speaker_id][person] += 1

for speaker_id, counts in matrix.items():
    print("=" * 80)
    print(speaker_id)

    for person, count in counts.most_common():
        print(f"  {person}: {count}")

speaker_id_to_person = {}

for speaker_id, counts in matrix.items():
    person, count = counts.most_common(1)[0]
    total = sum(counts.values())
    purity = count / total

    speaker_id_to_person[speaker_id] = {
        "person": person,
        "count": count,
        "total": total,
        "purity": purity,
    }
# {'SPEAKER_21': {'person': 'raw_per:laurent nunez', 'count': 2, 'total': 3, 'purity': 0.6666666666666666}
depute_match_nb = 0

print("\n")
print("//" * 100)

for speaker_id, data in speaker_id_to_person.items():
    if "deputy" in data["person"]:
        depute_match_nb += 1
    print("=" * 80)
    print(f"speaker_id: {speaker_id}")
    print(f"person:     {data['person']}")
    print(f"count:      {data['count']}")
    print(f"total:      {data['total']}")
    print(f"purity:     {data['purity']:.2%}")

print(f"{len(speaker_id_to_person)} number of speaker")
print(f"{depute_match_nb} number of deputues")

identified_speakers = len(speaker_id_to_person)

high_purity = sum(1 for v in speaker_id_to_person.values() if v["purity"] >= THRESHOLD)

print(f"{high_purity} of High purity (90+)")

high_purity_id = sum(
    1
    for v in speaker_id_to_person.values()
    if v["purity"] >= THRESHOLD and "deputy" in v["person"]
)

print(f"{high_purity_id} of High purity (90+) ID'd deputy")

# high-purity speaker_id → collect clean diarization segments → recompute centroid → compare / merge duplicated speakers
#
