import time

import spacy

start = time.time()
nlp = spacy.load("fr_core_news_sm")
after_load = time.time()


corpus = [
    "Bonjour à tous, je vous ai entendu monsieur Cordier, la séance est ouverte et pas de photo effectivement. L'ordre du jour appelle les questions au gouvernement. La première va être posée par monsieur Paul Christophe, président du groupe Horizon",
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
    for token in doc:
        print(
            f"{token.i:>3}  {token.text:<15} {token.lemma_:<15} {token.pos_:<6} "
            f"{token.dep_:<10} → {token.head.text:<15} ENT:{token.ent_type_:<5} MORPH:{str(token.morph)}"
        )
end = time.time()
print(f"Load took {after_load - start} s")
print(f"Token spacy took {end - start_2} s")
