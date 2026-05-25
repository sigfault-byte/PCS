import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

file = "data/interim/1ere-seance--questions-au-gouvernement--simplification-de-la-vie-economique-cmp--renforcer-la-s-14-avril-2026_03_alignment.json"

with open(file, "r", encoding="utf-8") as f:
    data = json.load(f)["transcript"]["raw_segments"]

transcritpion_id = np.array([s["segment_id"] for s in data])
xs = np.array([s["avg_logprob"] for s in data])

mean_logprob = np.mean(xs)
median_logprob = np.median(xs)

print(f"mean : {mean_logprob}median : {median_logprob}")

log_prob_segment = []

# KDE = smooth empirical distribution

kde = gaussian_kde(xs)

grid = np.linspace(xs.min(), xs.max(), 1000)

density = kde(grid)

# derivative of density curve
derivative = np.gradient(density, grid)

# derivative of the derivative
derivative2 = np.gradient(derivative, grid)

plt.figure()

plt.hist(xs, bins=80, density=True, alpha=0.4)

plt.plot(grid, density)

plt.title("avg_logprob distribution")

plt.xlabel("avg_logprob")

plt.ylabel("density")

plt.show()

plt.figure()

plt.plot(grid, derivative)

plt.axhline(0, linestyle="--")

plt.title("Derivative of avg_logprob density")

plt.xlabel("avg_logprob")

plt.ylabel("density slope")

plt.show()

plt.plot(grid, derivative2)

plt.show()
