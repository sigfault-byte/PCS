import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

FILE_PATH = "../data/interim/assemblee_nov26_2024_02_transcription.json"
AUDIO_DURATION_SEC = 4 * 3600 + 19 * 60 + 8
RESOLUTION = 0.1


def sec_to_hms(x, pos=None):
    x = int(x)
    hours = x // 3600
    minutes = (x % 3600) // 60
    seconds = x % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def sec_to_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def build_segment_df(segments, text_key=None, id_key="segment_id"):
    rows = []

    for seg in segments:
        start = seg["time"]["start_seconds"]
        end = seg["time"]["end_seconds"]

        row = {
            id_key: seg[id_key],
            "start_sec": round(start, 3),
            "end_sec": round(end, 3),
            "total_duration": round(end - start, 3),
        }

        if text_key is not None:
            row["is_empty"] = not seg[text_key].strip()

        rows.append(row)

    return pd.DataFrame(rows)


def build_token_df(tokens):
    rows = []

    for tok in tokens:
        start = tok["start_seconds"]
        end = tok["end_seconds"]

        rows.append(
            {
                "token_id": tok["token_id"],
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "total_duration": round(end - start, 3),
                "is_empty": not tok["raw_token"].strip(),
            }
        )

    return pd.DataFrame(rows)


def is_time_in_any_segment(t, df):
    return ((df["start_sec"] <= t) & (df["end_sec"] >= t)).any()


with open(FILE_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)


# -------------------------
# Build dataframes
# -------------------------

diarization = data["diarization"]["raw_segments"]
transcription_seg = data["transcript"]["raw_segments"]
transcription_tok = data["transcript"]["raw_tokens"]

d_df = build_segment_df(diarization)
ts_df = build_segment_df(transcription_seg, text_key="raw_text")
tt_df = build_token_df(transcription_tok)

# Optional: only non-empty transcript segments count as transcription activity
ts_nonempty_df = ts_df[~ts_df["is_empty"]].copy()

# -------------------------
# Summary prints
# -------------------------

diar_entire_duration = d_df["total_duration"].sum()
trans_seg_entire_duration = ts_df["total_duration"].sum()
trans_tok_entire_duration = tt_df["total_duration"].sum()

empty_seg = ts_df["is_empty"].sum()
empty_tok = tt_df["is_empty"].sum()

print("=== Diarization ===")
print(d_df.head())
print(d_df.tail())
print(d_df.shape)
print(f"Diarization total: {sec_to_ts(diar_entire_duration)}")

print("\n=== Transcript segments ===")
print(ts_df.head())
print(ts_df.tail())
print(ts_df.shape)
print(f"Transcript segment total: {sec_to_ts(trans_seg_entire_duration)}")
print(f"Empty transcript segments: {empty_seg}")

print("\n=== Transcript tokens ===")
print(tt_df.head())
print(tt_df.tail())
print(tt_df.shape)
print(f"Transcript token total: {sec_to_ts(trans_tok_entire_duration)}")
print(f"Empty transcript tokens: {empty_tok}")

# -------------------------
# Build sampled timeline
# -------------------------

timeline = pd.DataFrame(
    {"time_sec": np.arange(0, AUDIO_DURATION_SEC + RESOLUTION, RESOLUTION)}
)

timeline["diar_active"] = timeline["time_sec"].apply(
    lambda t: is_time_in_any_segment(t, d_df)
)

timeline["trans_active"] = timeline["time_sec"].apply(
    lambda t: is_time_in_any_segment(t, ts_nonempty_df)
)

# State encoding:
# 0 = neither
# 1 = diar only
# 2 = trans only
# 3 = both
timeline["state"] = timeline["diar_active"].astype(int) + 2 * timeline[
    "trans_active"
].astype(int)

print("\n=== Timeline ===")
print(timeline.head(20))
print(timeline["state"].value_counts().sort_index())

non_zero = timeline[timeline["state"] != 0]
print(non_zero.head(20))

# -------------------------
# Build interval lists
# -------------------------

ts_nonempty_df = ts_df[~ts_df["is_empty"]].copy()

diar_intervals = [
    (row["start_sec"], row["end_sec"] - row["start_sec"]) for _, row in d_df.iterrows()
]

trans_intervals = [
    (row["start_sec"], row["end_sec"] - row["start_sec"])
    for _, row in ts_nonempty_df.iterrows()
]

timeline = pd.DataFrame(
    {"time_sec": np.arange(0, AUDIO_DURATION_SEC + RESOLUTION, RESOLUTION)}
)


def is_time_in_any_segment(t, df):
    return ((df["start_sec"] <= t) & (df["end_sec"] >= t)).any()


timeline["diar_active"] = timeline["time_sec"].apply(
    lambda t: is_time_in_any_segment(t, d_df)
)

timeline["trans_active"] = timeline["time_sec"].apply(
    lambda t: is_time_in_any_segment(t, ts_nonempty_df)
)


def classify_row(row):
    if row["diar_active"] and row["trans_active"]:
        return "both"
    if row["diar_active"] and not row["trans_active"]:
        return "diar_only"
    if row["trans_active"] and not row["diar_active"]:
        return "trans_only"
    return "neither"


timeline["state_label"] = timeline.apply(classify_row, axis=1)

state_df = timeline[timeline["state_label"] != "neither"].copy()

state_df["prev_time"] = state_df["time_sec"].shift()
state_df["prev_label"] = state_df["state_label"].shift()

state_df["new_group"] = (state_df["state_label"] != state_df["prev_label"]) | (
    (state_df["time_sec"] - state_df["prev_time"]).round(6) > RESOLUTION
)

state_df["group_id"] = state_df["new_group"].cumsum()

intervals_df = (
    state_df.groupby("group_id")
    .agg(
        state_label=("state_label", "first"),
        start_sec=("time_sec", "min"),
        end_sec=("time_sec", "max"),
    )
    .reset_index(drop=True)
)

intervals_df["end_sec"] = intervals_df["end_sec"] + RESOLUTION
intervals_df["duration"] = (intervals_df["end_sec"] - intervals_df["start_sec"]).round(
    3
)

both_intervals = [
    (row["start_sec"], row["duration"])
    for _, row in intervals_df[intervals_df["state_label"] == "both"].iterrows()
]

diar_only_intervals = [
    (row["start_sec"], row["duration"])
    for _, row in intervals_df[intervals_df["state_label"] == "diar_only"].iterrows()
]

trans_only_intervals = [
    (row["start_sec"], row["duration"])
    for _, row in intervals_df[intervals_df["state_label"] == "trans_only"].iterrows()
]

# -------------------------
# Plot
# -------------------------

fig, ax = plt.subplots(figsize=(16, 5))

# Lane 1: transcription
ax.broken_barh(trans_intervals, (10, 6), facecolors="tab:orange", label="Transcription")

# Lane 2: diarization
ax.broken_barh(diar_intervals, (20, 6), facecolors="tab:blue", label="Diarization")

# Lane 3: overlap / disagreement
ax.broken_barh(both_intervals, (30, 6), facecolors="blue", label="Both")
ax.broken_barh(diar_only_intervals, (30, 6), facecolors="green", label="Diar only")
ax.broken_barh(trans_only_intervals, (30, 6), facecolors="red", label="Trans only")

ax.set_xlim(0, AUDIO_DURATION_SEC)
ax.margins(x=0)

ax.set_ylim(5, 40)
ax.set_yticks([13, 23, 33])
ax.set_yticklabels(["Transcription", "Diarization", "Overlap"])

tick_step = 15 * 60
ticks = np.arange(0, AUDIO_DURATION_SEC + tick_step, tick_step)
ax.set_xticks(ticks)
ax.xaxis.set_major_formatter(FuncFormatter(sec_to_hms))

ax.set_xlabel("Time")
ax.set_title("Diarization / Transcription / Overlap")
ax.grid(True, axis="x", alpha=0.3)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# =============

timeline = pd.DataFrame(
    {"time_sec": np.arange(0, AUDIO_DURATION_SEC + RESOLUTION, RESOLUTION)}
)

timeline["diar_active"] = timeline["time_sec"].apply(
    lambda t: is_time_in_any_segment(t, d_df)
)

timeline["trans_active"] = timeline["time_sec"].apply(
    lambda t: is_time_in_any_segment(t, ts_nonempty_df)
)


def classify_row(row):
    if row["diar_active"] and not row["trans_active"]:
        return "diar_only"
    if row["trans_active"] and not row["diar_active"]:
        return "trans_only"
    if row["diar_active"] and row["trans_active"]:
        return "both"
    return "neither"


timeline["state_label"] = timeline.apply(classify_row, axis=1)

disagree_df = timeline[timeline["state_label"].isin(["diar_only", "trans_only"])].copy()

disagree_df["prev_time"] = disagree_df["time_sec"].shift()
disagree_df["prev_label"] = disagree_df["state_label"].shift()

disagree_df["new_group"] = (disagree_df["state_label"] != disagree_df["prev_label"]) | (
    (disagree_df["time_sec"] - disagree_df["prev_time"]).round(6) > RESOLUTION
)

disagree_df["group_id"] = disagree_df["new_group"].cumsum()

intervals_df = (
    disagree_df.groupby("group_id")
    .agg(
        state_label=("state_label", "first"),
        start_sec=("time_sec", "min"),
        end_sec=("time_sec", "max"),
    )
    .reset_index(drop=True)
)

intervals_df["end_sec"] = intervals_df["end_sec"] + RESOLUTION
intervals_df["duration"] = (intervals_df["end_sec"] - intervals_df["start_sec"]).round(
    3
)

intervals_df["start_ts"] = intervals_df["start_sec"].apply(sec_to_ts)
intervals_df["end_ts"] = intervals_df["end_sec"].apply(sec_to_ts)

print(intervals_df.head(20))
intervals_df.sort_values("duration", ascending=False).head(20)
print(intervals_df)
intervals_df.groupby("state_label")["duration"].sum()
print(intervals_df)
