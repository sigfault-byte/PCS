import numpy as np

seg = np.load(
    "../data/interim/assemblee_nov26_2024_02_segment_embeddings.npz", allow_pickle=True
)
cent = np.load(
    "../data/interim/assemblee_nov26_2024_02_speaker_centroids.npz", allow_pickle=True
)

print("SEG files:", seg.files)
print("CENT files:", cent.files)

print("segment_ids shape:", seg["segment_ids"].shape)
print("segment_speaker_ids shape:", seg["segment_speaker_ids"].shape)
print("segment_embeddings shape:", seg["segment_embeddings"].shape)

print("speaker_ids shape:", cent["speaker_ids"].shape)
print("speaker_centroids shape:", cent["speaker_centroids"].shape)

print("segment dtype:", seg["segment_embeddings"].dtype)
print("centroid dtype:", cent["speaker_centroids"].dtype)

seg_emb = seg["segment_embeddings"]
cent_emb = cent["speaker_centroids"]

print("segment NaN:", np.isnan(seg_emb).any())
print("segment Inf:", np.isinf(seg_emb).any())
print("centroid NaN:", np.isnan(cent_emb).any())
print("centroid Inf:", np.isinf(cent_emb).any())

seg_norms = np.linalg.norm(seg_emb, axis=1)
cent_norms = np.linalg.norm(cent_emb, axis=1)

print("segment norm min/max/mean:", seg_norms.min(), seg_norms.max(), seg_norms.mean())
print(
    "centroid norm min/max/mean:", cent_norms.min(), cent_norms.max(), cent_norms.mean()
)


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


speaker_ids = cent["speaker_ids"]
speaker_centroids = cent["speaker_centroids"]

speaker_to_centroid = {
    speaker_ids[i]: speaker_centroids[i] for i in range(len(speaker_ids))
}

scores = []
for seg_id, spk_id, emb in zip(seg["segment_ids"], seg["segment_speaker_ids"], seg_emb):
    c = speaker_to_centroid[spk_id]
    scores.append((seg_id, spk_id, cosine(emb, c)))

scores_sorted = sorted(scores, key=lambda x: x[2])

print("Worst 20:")
for row in scores_sorted[:20]:
    print(row)

vals = np.array([x[2] for x in scores])
print("min/max/mean:", vals.min(), vals.max(), vals.mean())

cent_unit = cent_emb / np.linalg.norm(cent_emb, axis=1, keepdims=True)
sim = cent_unit @ cent_unit.T

print(sim.shape)
print(sim[:5, :5])
