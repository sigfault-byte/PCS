import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

file = Path(
    "data/audio-audit/1ere-seance--questions-au-gouvernement--simplification-de-la-vie-economique-cmp--renforcer-la-s-14-avril-2026_audio_audit.json"
)

with open(file, "r", encoding="utf-8") as f:
    frames = json.load(f)["frames"]

time = np.array([frame["time_seconds"] for frame in frames if frame["db"] > -50])

flatness = np.array(
    [frame["spectral_flatness"] for frame in frames if frame["db"] > -50]
)
mean_flatness = np.mean(flatness)
median_flatness = np.median(flatness)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=time,
        y=flatness,
        mode="lines",
        name="Spectral Flatness",
    )
)

fig.update_layout(
    title="Spectral Flatness Over Time",
    xaxis_title="Time (s)",
    yaxis_title="Spectral Flatness",
)

fig.add_hline(
    y=mean_flatness, line_dash="dash", annotation_text=f"Mean: {mean_flatness:.4f}"
)

fig.add_hline(
    y=median_flatness, line_dash="dot", annotation_text=f"Median: {median_flatness:.4f}"
)

fig.show()
