import re

import numpy as np
from sentence_transformers import SentenceTransformer


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def cosine(a, b) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def build_semantic_chunks(sentences, embeddings, drop_threshold=-0.10):
    cosarr = []

    for i in range(len(sentences) - 1):
        cosarr.append(cosine(embeddings[i], embeddings[i + 1]))

    breakpoints = []

    for i in range(1, len(cosarr)):
        delta = cosarr[i] - cosarr[i - 1]

        if delta < drop_threshold:
            # drop at pair i means new chunk starts at sentence i + 1
            breakpoints.append(i + 1)

    chunks = []
    start = 0

    for bp in breakpoints:
        chunks.append(" ".join(sentences[start:bp]))
        start = bp

    chunks.append(" ".join(sentences[start:]))

    return chunks, cosarr, breakpoints


def search_chunks(query, chunks, model, top_k=3):
    chunk_embeddings = model.encode(chunks)
    query_embedding = model.encode(query)

    scores = []

    for i, emb in enumerate(chunk_embeddings):
        scores.append((i, cosine(query_embedding, emb)))

    scores.sort(key=lambda x: x[1], reverse=True)

    return scores[:top_k]


raw_text = """
Merci madame la présidente, messieurs les ministres, monsieur le rapporteur, monsieur le président de la commission, monsieur le président de la commission spéciale. La simplification de notre vie économique n'est pas un sujet que l'on peut balayer d'un revers de main. Dans le contexte qui est le nôtre, compétitivité sous pression, entrepreneurs qui peinent à se projeter, charges administratives qui pèsent sur nos entreprises, rejeter ce texte sans même le soumettre au débat est un choix politique lourd de sang, chers collègues de gauche. La simplification n'est pas un sujet technique parmi d'autres, elle conditionne directement la capacité de notre pays à créer des emplois, à attirer des investissements et à libérer l'énergie de celles et ceux qui y entreprennent. Voter cette motion, c'est refuser à nos acteurs économiques les réponses qu'ils attendent. Nos entrepreneurs, nos artisans, nos commerçants ont besoin de stabilité, de lisibilité, de prévisibilité. Ils ont besoin de règles claires et proportionnées qui leur permettent de se projeter, d'investir, de recruter, de transmettre. Ce projet de loi est le fruit d'un travail approfondi, conduit depuis plus de deux ans au Sénat, à l'Assemblée nationale et en commission mixte paritaire. Il mérite mieux qu'un rejet sans examen. C'est pourquoi nous voterons contre cette motion de rejet préalable.
"""


# Other option: "Lajavaness/sentence-camembert-base"
# "Lajavaness/sentence-camembert-large"
# model = SentenceTransformer(
#     "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# )
# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model = SentenceTransformer("h4c5/sts-camembert-base")

sentences = split_sentences(raw_text)


embeddings = model.encode(sentences)

chunks, cosarr, breakpoints = build_semantic_chunks(
    sentences,
    embeddings,
    drop_threshold=-0.10,
)

print("\n=== SENTENCE SIMILARITIES ===")
for i, value in enumerate(cosarr):
    print(f"{i} -> {i + 1}: {value:.4f}")

print("\n=== BREAKPOINTS ===")
print(breakpoints)

print("\n=== CHUNKS ===")
for i, chunk in enumerate(chunks):
    print("=" * 80)
    print(f"CHUNK {i}")
    print(chunk)

print("\n=== MINI RAG SEARCH ===")

query = "Les acteurs économiques"

results = search_chunks(
    query=query,
    chunks=chunks,
    model=model,
    top_k=3,
)

for chunk_id, score in results:
    print("=" * 80)
    print(f"CHUNK {chunk_id} | score={score:.4f}")
    print(chunks[chunk_id])
