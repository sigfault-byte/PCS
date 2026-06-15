import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model = SentenceTransformer("h4c5/sts-camembert-base")

raw_text = "Merci madame la présidente, messieurs les ministres, monsieur le rapporteur, monsieur le président de la commission, monsieur le président de la commission spéciale. La simplification de notre vie économique n'est pas un sujet que l'on peut balayer d'un revers de main. Dans le contexte qui est le nôtre, compétitivité sous pression, entrepreneurs qui peinent à se projeter, charges administratives qui pèsent sur nos entreprises, rejeter ce texte sans même le soumettre au débat est un choix politique lourd de sang, chers collègues de gauche. La simplification n'est pas un sujet technique parmi d'autres, elle conditionne directement la capacité de notre pays à créer des emplois, à attirer des investissements et à libérer l'énergie de celles et ceux qui y entreprennent. Voter cette motion, c'est refuser à nos acteurs économiques les réponses qu'ils attendent. Nos entrepreneurs, nos artisans, nos commerçants ont besoin de stabilité, de lisibilité, de prévisibilité. Ils ont besoin de règles claires et proportionnées qui leur permettent de se projeter, d'investir, de recruter, de transmettre. Ce projet de loi est le fruit d'un travail approfondi, conduit depuis plus de deux ans au Sénat, à l'Assemblée nationale et en commission mixte paritaire. Il mérite mieux qu'un rejet sans examen. C'est pourquoi nous voterons contre cette motion de rejet préalable."

sentences = [s.strip() for s in raw_text.split(".") if s.strip()]
index = 0
for i in sentences:
    print(f"INDEX: {index}")
    print(i)
    index += 1

# sentences = [
#     "J'aime le pain",
#     "J'aime la baguette",
#     "J'adore le pain",
#     "J'apprécie le pain",
#     "Je déteste le pain",
#     "Le chat dort sur le canapé",
# ]


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


embeddings = model.encode(sentences)

cosarr = []

for i in range(len(sentences) - 1):
    cos = cosine(embeddings[i], embeddings[i + 1])
    cosarr.append(cos)

    print()
    print(sentences[i][:50])
    print(sentences[i + 1][:50])
    print(f"cos={cos:.4f}")
    print(f"{i} -> {i + 1}: {cos:.4f}")


for i, cos in enumerate(cosarr):
    print("=" * 80)
    print(f"PAIR {i} -> {i + 1} | cosine={cos:.4f}")
    print("A:", sentences[i])
    print("B:", sentences[i + 1])

print("***===***" * 8, flush=True)

lowest = sorted(enumerate(cosarr), key=lambda x: x[1])

for i, cos in lowest:
    print("=" * 80)
    print(f"PAIR {i} -> {i + 1} | cosine={cos:.4f}")
    print("A:", sentences[i])
    print("B:", sentences[i + 1])

deltas = []

for i in range(1, len(cosarr)):
    deltas.append(cosarr[i] - cosarr[i - 1])

for i in deltas:
    print(i)

# plt.plot(deltas)
# plt.show()
plt.plot(cosarr)
# plt.xlabel("Sentence pair")

# plt.ylabel("Cosine similarity")

# plt.title("Similarity between consecutive sentences")

# plt.scatter(range(len(cosarr)), cosarr)

plt.show()

# for i in range(len(sentences)):
#     for j in range(i + 1, len(sentences)):
#         cos = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]

#         euc = euclidean(embeddings[i], embeddings[j])

#         print()
#         print(f"{sentences[i]!r}")
#         print(f"{sentences[j]!r}")
#         print(f"cosine   = {cos:.4f}")
#         print(f"euclidean= {euc:.4f}")


pca = PCA(n_components=2)


coords = pca.fit_transform(embeddings)

print("Explained variance:", pca.explained_variance_ratio_)

for sentence, (x, y) in zip(sentences, coords):
    plt.scatter(x, y)
    plt.text(x, y, sentence)

plt.show()
