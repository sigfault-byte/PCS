import json
from collections import Counter
from pathlib import Path

import numpy as np

from assemblybot.models.flags import SegmentFlag, flags_to_list  # type: ignore

FILE = Path(
    "data/interim/1ere-seance--questions-au-gouvernement--simplification-de-la-vie-economique-cmp--renforcer-la-s-14-avril-2026_02_transcription_whisper_segment_audit.json"
)

with open(FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

segments = data["transcript"]["raw_segments"]

# Keep only segments with flags
flagged_segments = [s for s in segments if s["flags"] != 0]

print(f"Total segments        : {len(segments)}")
print(f"Flagged segments      : {len(flagged_segments)}")
print()

# Count individual flags
flag_counter = Counter()

for segment in flagged_segments:
    active_flags = flags_to_list(segment["flags"])

    for flag_name in active_flags:
        flag_counter[flag_name] += 1

print("----=== FLAG SUMMARY ===0----")
print()

total_flagged = len(flagged_segments)
for flag_name, count in flag_counter.most_common():
    percentage = (count / total_flagged) * 100

    print(f"{flag_name:<40} {count:<5} ({percentage:.2f}%)")

partial_overlap = [
    s for s in flagged_segments if s["flags"] & SegmentFlag.PARTIAL_VAD_OVERLAP
]

durations = np.array([s["time"]["duration_seconds"] for s in partial_overlap])

print(f"mean = {durations.mean():.2f}")
print(
    f"median= {np.median(durations):.2f}"
)  # someday you ll remember np.array().median does not exist this is not panda
print(f"p10  = {np.percentile(durations, 10):.2f}")
print(f"p90  = {np.percentile(durations, 90):.2f}")
print(f"min  = {durations.min():.2f}")
print(f"max  = {durations.max():.2f}")
print(f"std  = {durations.std():.2f}")


combo_counter = Counter()

for segment in flagged_segments:
    active_flags = flags_to_list(segment["flags"])

    combo_key = tuple(sorted(active_flags))

    combo_counter[combo_key] += 1


print("------=== FLAG COMBINATIONS ===------")
print()

for combo, count in combo_counter.most_common():
    combo_str = " | ".join(combo)

    print(f"{combo_str:<80} {count}")
