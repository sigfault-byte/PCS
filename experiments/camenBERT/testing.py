import json
import os

from transformers import pipeline

token = os.environ["HF_TOKEN"]

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

ADDRESS_PATTERNS = [
    "monsieur le ministre",
    "madame la ministre",
    "madame la présidente",
    "monsieur le président",
]


file = "data/interim/1ere-seance--questions-au-gouvernement--simplification-de-la-vie-economique-cmp--renforcer-la-s-14-avril-2026_01_turns.json"

ner = pipeline(
    "token-classification",
    model="Jean-Baptiste/camembert-ner",
    aggregation_strategy="simple",
    token=token,
)

with open(file, "r", encoding="utf-8") as f:
    turns = json.load(f)["turns"]


def context_window(text, start, end, radius=80):
    return text[max(0, start - radius) : min(len(text), end + radius)].lower()


entities = []

turn42 = turns[41]
print("*" * 5)
print(ner(turn42["text"]))
print("*" * 5)

for turn in turns[:25]:
    for ent in ner(turn["text"]):
        if ent["entity_group"] == "PER" and ent["score"] > 0.90:
            entities.append(
                {"entity": ent, "text": turn["text"], "id": turn["turn_id"]}
            )


probable_previous_speaker = 0
next_speaker = 0

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
        probable_previous_speaker += 1
        role = "probable_previous_speaker"
    else:
        role = "person_mentioned"

    if role == "person_mentioned":
        continue
    print(f"ent: {ent}", f"role is : {role}")

    print("TURN", turn)
    print("PER:", ent["word"], "| score:", float(ent["score"]))
    print("ROLE:", role)
    print("CTX:", ctx)
    print("=" * 80)

print(f"number of PER: {len(entities)}")
print(f"number of probable_next_speaker = {next_speaker}")
print(f"number of probable_previous_speaker = {probable_previous_speaker}")


test = ner(
    "Merci madame la députée, mesdames et messieurs les députés. Monsieur le député, vous le savez, depuis le début, nous avons les yeux rivés sur ce qui se passe au Moyen-Orient. Votre question aurait probablement dû commencer par cela. On est dans une situation extrêmement volatile, avec des prix qui fluctuent en fonction des prix. C'est la prise de parole des grandes puissances, et en premier lieu de Donald Trump. On s'est adapté depuis les premiers jours. Dès les premiers jours du conflit, nous avons décidé, à la demande du Premier ministre, d'être aux côtés des agriculteurs, des pêcheurs, parce que vous l'avez souligné et vous avez raison, ils traversent une situation extrêmement difficile, d'être aux côtés des transporteurs et d'apporter une aide d'urgence qui rende soutenable l'activité économique, parce que derrière, c'est des emplois, c'est des salaires, et évidemment des enjeux sociaux extrêmement importants. Nous adapterons ce dispositif autant que nécessaire dans les jours et dans les semaines à venir, en fonction d'une situation qui nous dépasse, en tout cas qui dépasse le cadre purement franco-français. Nous avons dit depuis le départ que ces aides pouvaient être amplifiées, pouvaient être reconduites, autant que de besoin, encore une fois, en fonction de l'évolution du conflit. Je note quand même que vous évoquez parmi les pistes pour faire baisser durablement le carburant, alors la baisse de la TVA à 5,5%, j'ai déjà eu l'occasion d'y répondre, 12,5%. C'est une baisse de 12 milliards d'euros qui aujourd'hui ne sont pas financées dans vos propositions. Et ensuite, je note l'exploration, pardonnez-moi, de nouveaux gisements d'hydrocarbures. Et là, on a un désaccord effectivement fondamental sur l'approche qu'on a de l'énergie, parce que nous, on considère qu'elle doit être décarbonée, qu'il faut électrifier, et que pour ça, on a besoin des renouvelables et du nucléaire. Certainement pas d'aller chercher de nouveaux gisements de pétrole ou de gaz de schiste, comme j'ai pu l'entendre dans vos rangs. Donc voilà, j'en conclurai là-dessus. Néanmoins, de façon très républicaine, je voudrais aussi vous souhaiter bon vent pour la suite de vos fonctions."
)

for i in test:
    print(i)
