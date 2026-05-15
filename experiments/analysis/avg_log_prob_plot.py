import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FILE = Path(
    "data/interim/1ere-seance--questions-au-gouvernement--simplification-de-la-vie-economique-cmp--renforcer-la-s-14-avril-2026_02_transcription_VAD-1000_whisper_segment_audit.json"
)

with open(FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

segments = data["transcript"]["raw_segments"]

# x-axis: segment midpoint in seconds
x = np.array(
    [(s["time"]["start_seconds"] + s["time"]["end_seconds"]) / 2 for s in segments]
)

# y-axis: "badness" proxy
# avg_logprob closer to 0 is better, so -avg_logprob makes worse = higher
badness = np.array([-s["avg_logprob"] for s in segments])

segment_ids = [s["segment_id"] for s in segments]

# discrete difference: how much the signal changes between consecutive segments
delta = np.diff(badness)

# x values for delta: place each change between the two points
x_delta = (x[1:] + x[:-1]) / 2

plt.figure(figsize=(16, 6))
plt.plot(x, badness, linewidth=0.5, alpha=0.5)
plt.scatter(x, badness, s=4)
plt.xlabel("Time midpoint / seconds")
plt.ylabel("-avg_logprob  (higher = worse)")
plt.title("Whisper avg_logprob badness over time")
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(16, 6))
plt.plot(x_delta, delta, linewidth=0.8)
plt.axhline(0, linewidth=1)
plt.xlabel("Time midpoint / seconds")
plt.ylabel("Discrete difference")
plt.title("Change in Whisper confidence between segments")
plt.grid(True, alpha=0.3)
plt.show()

top_n = 20

idx = np.argsort(np.abs(delta))[-top_n:][::-1]

for i in idx:
    print(
        f"{segment_ids[i]} -> {segment_ids[i + 1]} | "
        f"time={x_delta[i]:.2f}s | "
        f"delta={delta[i]:+.3f} | "
        f"badness {badness[i]:.3f} -> {badness[i + 1]:.3f}"
    )
