import time

import spacy

start = time.time()
nlp = spacy.load("fr_core_news_sm")
after_load = time.time()

IMPORTANT_LEMMAS = {
    "entendre",
    "dire",
    "répondre",
    "être",
    "poser",
    "appeler",
    "donner",
}

corpus = [
    "La première va être posée par monsieur Paul Christophe, président du groupe Horizon.",
    "Hugo c'est l'heure d'aller a l'école.",
    # "La parole est à présent à Monsieur Marcelin Nadeau, pour le groupe GDR.",
    # "Merci madame la présidente, ma question s'adresse à monsieur le premier ministre. Alors que la France connaît d'importantes turbulences financières, le prix de l'énergie n'échappe pas à l'instabilité grandissante de notre économie.",
]
start_2 = time.time()
for text in corpus:
    doc = nlp(text)

    print("\nTEXT:", text)

    print("ENTITIES:")
    for ent in doc.ents:
        print("  ", ent.text, ent.label_)

    print("TOKENS:")

    # Header with explicit labels
    print(
        f"{'IDX':>3}  {'TEXT':<15} {'LEMMA':<15} {'POS':<6} "
        f"{'DEP':<10} → {'HEAD':<15} {'ENT_TYPE':<8} {'MORPH'}"
    )
    print("-" * 110)

    for token in doc:
        print(
            f"{token.i:>3}  {token.text:<15} {token.lemma_:<15} {token.pos_:<6} "
            f"{token.dep_:<10} → {token.head.text:<15} "
            f"{token.ent_type_:<8} {str(token.morph)}"
        )

    print()
    for token in doc:
        if token.lemma_ in IMPORTANT_LEMMAS:
            print("ROOT VERB:", token.text)

            # agent
            for child in token.children:
                if child.dep_ == "obl:agent" or "obl:arg":
                    print("AGENT:", child.text)

                    # appos (Paul)
                    for app in child.children:
                        if app.dep_ == "appos" or "flat:name":
                            print("APPOS:", app.text)

                            # flat:name (Christophe)
                            for name in app.children:
                                if name.dep_ == "flat:name":
                                    print("NAME:", name.text)


end = time.time()
print(f"Load took {after_load - start} s")
print(f"Token spacy took {end - start_2} s")
