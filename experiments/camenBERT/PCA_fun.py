import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer

INPUT_FILE = Path(
    "data/interim/1ere-seance--questions-au-gouvernement--simplification-de-la-vie-economique-cmp--renforcer-la-s-14-avril-2026_01_turns.json"
)
MODEL_NAME = "camembert-base"


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


with INPUT_FILE.open("r", encoding="utf-8") as f:
    turns = json.load(f)["turns"]

texts = [turn["text"] for turn in turns]
turn_ids = [turn["turn_id"] for turn in turns]
speaker_ids = [turn.get("speaker_id") for turn in turns]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

embeddings = []

batch_size = 16

with torch.no_grad():
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        outputs = model(**inputs)

        pooled = mean_pool(
            outputs.last_hidden_state,
            inputs["attention_mask"],
        )

        embeddings.append(pooled.cpu().numpy())

X = np.vstack(embeddings)

pca = PCA(n_components=2)
coords = pca.fit_transform(X)

print("X shape:", X.shape)
print("Explained variance:", pca.explained_variance_ratio_)

# colors = [
#     "red" if speaker_id in {"SPEAKER_40", "SPEAKER_36"} else "blue"
#     for speaker_id in speaker_ids
# ]

colors = [turn["audio_time"]["duration_seconds"] for turn in turns]


plt.figure(figsize=(10, 8))
plt.scatter(coords[:, 0], coords[:, 1], c=colors, s=30)

plt.colorbar()

for x, y, turn_id in zip(coords[:, 0], coords[:, 1], turn_ids):
    plt.text(x, y, str(turn_id), fontsize=7)

plt.title("CamemBERT turn embeddings PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()


dist = np.linalg.norm(coords, axis=1)

top = np.argsort(dist)[-10:]

for idx in top:
    print("=" * 80)
    print(turn_ids[idx])
    print(texts[idx][:500])
