import json

import pandas as pd


# ---------- helpers ----------
def sec_to_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def classify_position(
    dist_to_left: float, dist_to_right: float, edge_tol: float = 0.25
) -> str:
    if dist_to_left <= edge_tol:
        return "near_prev"
    elif dist_to_right <= edge_tol:
        return "near_next"
    return "deep_gap"


file = "../data/interim/assemblee_nov26_2024_03_diarization_collapsed.json"

# ---------- load json ----------
with open(file, "r", encoding="utf-8") as f:
    data = json.load(f)


# ---------- diarization: collapsed segments ----------
collapsed = data["diarization"]["collapsed_segments"]

ddf = pd.DataFrame(collapsed)
ddf_time = pd.json_normalize(ddf["time"])  # type: ignore
ddf = pd.concat([ddf.drop(columns=["time"]), ddf_time], axis=1)
ddf = ddf.sort_values("start_seconds").reset_index(drop=True)

print("\n=== COLLAPSED DIARIZATION SEGMENTS ===")
print(ddf.head())
print(ddf.columns.tolist())
print(f"count: {len(ddf)}")


# ---------- compute gaps between diarization segments ----------
gaps = pd.DataFrame(
    {
        "prev_segment_id": ddf["segment_id"].shift(1),
        "next_segment_id": ddf["segment_id"],
        "gap_start": ddf["end_seconds"].shift(1),
        "gap_end": ddf["start_seconds"],
    }
)

gaps["gap_seconds"] = gaps["gap_end"] - gaps["gap_start"]
gaps = gaps[gaps["gap_seconds"] > 0].reset_index(drop=True)

gaps["gap_start_ts"] = gaps["gap_start"].apply(sec_to_ts)
gaps["gap_end_ts"] = gaps["gap_end"].apply(sec_to_ts)

print("\n=== DIARIZATION GAPS ===")
print(gaps.head(10))
print(f"gap count: {len(gaps)}")
print("\nGap duration stats:")
print(gaps["gap_seconds"].describe())


# ---------- optional: keep only meaningful gaps ----------
MIN_GAP_SECONDS = 0.50
meaningful_gaps = gaps[gaps["gap_seconds"] >= MIN_GAP_SECONDS].reset_index(drop=True)

print(f"\nMeaningful gaps (>= {MIN_GAP_SECONDS:.2f}s): {len(meaningful_gaps)}")


# ---------- whisper raw tokens ----------
raw_tokens = data["transcript"]["raw_tokens"]

wdf = pd.DataFrame(raw_tokens)
# idiot
wdf = wdf.rename(columns={"raw_token": "text"})

# flatten nested "time" dict if present
if "time" in wdf.columns:
    wdf_time = pd.json_normalize(wdf["time"])
    wdf = pd.concat([wdf.drop(columns=["time"]), wdf_time], axis=1)

# sort by token start time
wdf = wdf.sort_values("start_seconds").reset_index(drop=True)

# midpoint = strongest temporal anchor for assignment
wdf["mid_seconds"] = (wdf["start_seconds"] + wdf["end_seconds"]) / 2

print("\n=== WHISPER RAW TOKENS ===")
print(wdf.head())
print(wdf.columns.tolist())
print(f"count: {len(wdf)}")


# ---------- whisper tokens overlapping diarization gaps ----------
overlap_matches = []

for _, gap in meaningful_gaps.iterrows():
    gap_start = gap["gap_start"]
    gap_end = gap["gap_end"]

    overlapping = wdf[
        (wdf["start_seconds"] < gap_end) & (wdf["end_seconds"] > gap_start)
    ].copy()

    if overlapping.empty:
        continue

    overlapping["prev_segment_id"] = gap["prev_segment_id"]
    overlapping["next_segment_id"] = gap["next_segment_id"]
    overlapping["gap_start"] = gap_start
    overlapping["gap_end"] = gap_end
    overlapping["gap_seconds"] = gap["gap_seconds"]
    overlapping["gap_start_ts"] = gap["gap_start_ts"]
    overlapping["gap_end_ts"] = gap["gap_end_ts"]

    overlapping["overlap_start"] = overlapping["start_seconds"].clip(lower=gap_start)
    overlapping["overlap_end"] = overlapping["end_seconds"].clip(upper=gap_end)
    overlapping["overlap_seconds"] = (
        overlapping["overlap_end"] - overlapping["overlap_start"]
    )

    overlap_matches.append(overlapping)

gap_overlap_df = (
    pd.concat(overlap_matches, ignore_index=True) if overlap_matches else pd.DataFrame()
)

print("\n=== WHISPER TOKENS OVERLAPPING GAPS ===")
if gap_overlap_df.empty:
    print("No Whisper tokens overlap meaningful diarization gaps.")
else:
    cols_to_show = [
        c
        for c in [
            "token_id",
            "text",
            "token",
            "start_seconds",
            "end_seconds",
            "mid_seconds",
            "gap_start_ts",
            "gap_end_ts",
            "gap_seconds",
            "overlap_seconds",
            "prev_segment_id",
            "next_segment_id",
        ]
        if c in gap_overlap_df.columns
    ]

    print(gap_overlap_df[cols_to_show].head(20))
    print(f"count: {len(gap_overlap_df)}")


# ---------- midpoint inside gap: stronger suspicion ----------
mid_matches = []

for _, gap in meaningful_gaps.iterrows():
    gap_start = gap["gap_start"]
    gap_end = gap["gap_end"]

    inside = wdf[
        (wdf["mid_seconds"] >= gap_start) & (wdf["mid_seconds"] <= gap_end)
    ].copy()

    if inside.empty:
        continue

    inside["prev_segment_id"] = gap["prev_segment_id"]
    inside["next_segment_id"] = gap["next_segment_id"]
    inside["gap_start"] = gap_start
    inside["gap_end"] = gap_end
    inside["gap_seconds"] = gap["gap_seconds"]
    inside["gap_start_ts"] = gap["gap_start_ts"]
    inside["gap_end_ts"] = gap["gap_end_ts"]

    inside["dist_to_left"] = inside["mid_seconds"] - gap_start
    inside["dist_to_right"] = gap_end - inside["mid_seconds"]
    inside["position_label"] = inside.apply(
        lambda row: classify_position(row["dist_to_left"], row["dist_to_right"]),
        axis=1,
    )

    mid_matches.append(inside)

gap_mid_df = (
    pd.concat(mid_matches, ignore_index=True) if mid_matches else pd.DataFrame()
)

print("\n=== WHISPER TOKEN MIDPOINTS INSIDE GAPS ===")
if gap_mid_df.empty:
    print("No Whisper token midpoints fall inside meaningful gaps.")
else:
    cols_to_show = [
        c
        for c in [
            "token_id",
            "text",
            "token",
            "start_seconds",
            "end_seconds",
            "mid_seconds",
            "gap_start_ts",
            "gap_end_ts",
            "gap_seconds",
            "dist_to_left",
            "dist_to_right",
            "position_label",
            "prev_segment_id",
            "next_segment_id",
        ]
        if c in gap_mid_df.columns
    ]

    print(gap_mid_df[cols_to_show].head(30))
    print(f"count: {len(gap_mid_df)}")

    print("\nPosition labels:")
    print(gap_mid_df["position_label"].value_counts())


# ---------- whisper token midpoints inside diarization gaps ----------
mid_matches = []

for _, gap in meaningful_gaps.iterrows():
    gap_start = gap["gap_start"]
    gap_end = gap["gap_end"]

    inside = wdf[
        (wdf["mid_seconds"] >= gap_start) & (wdf["mid_seconds"] <= gap_end)
    ].copy()

    if inside.empty:
        continue

    inside["prev_segment_id"] = gap["prev_segment_id"]
    inside["next_segment_id"] = gap["next_segment_id"]
    inside["gap_start"] = gap_start
    inside["gap_end"] = gap_end
    inside["gap_seconds"] = gap["gap_seconds"]
    inside["gap_start_ts"] = gap["gap_start_ts"]
    inside["gap_end_ts"] = gap["gap_end_ts"]

    inside["dist_to_left"] = inside["mid_seconds"] - gap_start
    inside["dist_to_right"] = gap_end - inside["mid_seconds"]
    inside["position_label"] = inside.apply(
        lambda row: classify_position(row["dist_to_left"], row["dist_to_right"]),
        axis=1,
    )

    mid_matches.append(inside)

gap_mid_df = (
    pd.concat(mid_matches, ignore_index=True) if mid_matches else pd.DataFrame()
)

print("\n=== WHISPER TOKEN MIDPOINTS INSIDE GAPS ===")
print(wdf.columns.tolist())
print(wdf.head(3).to_dict(orient="records"))
if gap_mid_df.empty:
    print("No Whisper token midpoints fall inside meaningful gaps.")
else:
    cols_to_show = [
        c
        for c in [
            "token_id",
            "text",
            "token",
            "start_seconds",
            "end_seconds",
            "mid_seconds",
            "gap_start_ts",
            "gap_end_ts",
            "gap_seconds",
            "dist_to_left",
            "dist_to_right",
            "position_label",
            "prev_segment_id",
            "next_segment_id",
        ]
        if c in gap_mid_df.columns
    ]

    print(gap_mid_df[cols_to_show].head(30))
    print(f"count: {len(gap_mid_df)}")

    print("\nPosition labels:")
    print(gap_mid_df["position_label"].value_counts())

    print("\n=== ONLY DEEP GAP TOKENS ===")

    deep_gap_df = gap_mid_df[gap_mid_df["position_label"] == "deep_gap"].copy()

    if deep_gap_df.empty:
        print("No deep gap tokens found.")
    else:
        print(deep_gap_df[cols_to_show].head(50))
        print(f"deep_gap count: {len(deep_gap_df)}")

    # print(deep_gap_df[cols_to_show].tail(70).to_string(index=False))


# ---------- contextual midpoint using current token start + next token end ----------
wdf = wdf.sort_values("start_seconds").reset_index(drop=True)

wdf["next_start_seconds"] = wdf["start_seconds"].shift(-1)
wdf["next_end_seconds"] = wdf["end_seconds"].shift(-1)
wdf["next_text"] = wdf["text"].shift(-1)

# midpoint of the larger contextual span:
# current token start  -> next token end
wdf["context_mid_seconds"] = (wdf["start_seconds"] + wdf["next_end_seconds"]) / 2

print("\n=== TOKENS WITH CONTEXT MIDPOINT ===")
print(
    wdf[
        [
            "token_id",
            "text",
            "start_seconds",
            "end_seconds",
            "next_text",
            "next_end_seconds",
            "context_mid_seconds",
        ]
    ].head(10)
)

# ---------- contextual midpoint inside diarization gaps ----------
context_mid_matches = []

for _, gap in meaningful_gaps.iterrows():
    gap_start = gap["gap_start"]
    gap_end = gap["gap_end"]

    inside = wdf[
        (wdf["context_mid_seconds"] >= gap_start)
        & (wdf["context_mid_seconds"] <= gap_end)
    ].copy()

    if inside.empty:
        continue

    inside["prev_segment_id"] = gap["prev_segment_id"]
    inside["next_segment_id"] = gap["next_segment_id"]
    inside["gap_start"] = gap_start
    inside["gap_end"] = gap_end
    inside["gap_seconds"] = gap["gap_seconds"]
    inside["gap_start_ts"] = gap["gap_start_ts"]
    inside["gap_end_ts"] = gap["gap_end_ts"]

    inside["dist_to_left"] = inside["context_mid_seconds"] - gap_start
    inside["dist_to_right"] = gap_end - inside["context_mid_seconds"]
    inside["position_label"] = inside.apply(
        lambda row: classify_position(row["dist_to_left"], row["dist_to_right"]),
        axis=1,
    )

    context_mid_matches.append(inside)

gap_context_mid_df = (
    pd.concat(context_mid_matches, ignore_index=True)
    if context_mid_matches
    else pd.DataFrame()
)

print("\n=== TOKEN CONTEXT MIDPOINTS INSIDE GAPS ===")
if gap_context_mid_df.empty:
    print("No token context midpoints fall inside meaningful gaps.")
else:
    cols_to_show = [
        c
        for c in [
            "token_id",
            "text",
            "start_seconds",
            "end_seconds",
            "next_text",
            "next_end_seconds",
            "context_mid_seconds",
            "gap_start_ts",
            "gap_end_ts",
            "gap_seconds",
            "dist_to_left",
            "dist_to_right",
            "position_label",
            "prev_segment_id",
            "next_segment_id",
        ]
        if c in gap_context_mid_df.columns
    ]

    print(gap_context_mid_df[cols_to_show].head(30))
    print(f"context-mid count: {len(gap_context_mid_df)}")

    print("\nPosition labels:")
    print(gap_context_mid_df["position_label"].value_counts())

print("\n=== ONLY DEEP GAP TOKENS (CONTEXT MIDPOINT) ===")

deep_context_gap_df = gap_context_mid_df[
    gap_context_mid_df["position_label"] == "deep_gap"
].copy()

if deep_context_gap_df.empty:
    print("No deep gap tokens found with context midpoint.")
else:
    # print(deep_context_gap_df[cols_to_show].tail(70).to_string(index=False))
    # print(deep_context_gap_df[cols_to_show[:7]].tail(70).to_string(index=False))
    print(f"deep_gap count: {len(deep_context_gap_df)}")

# add human-readable timestamp columns for easier inspection
deep_context_gap_df["start_ts"] = deep_context_gap_df["start_seconds"].apply(sec_to_ts)
deep_context_gap_df["end_ts"] = deep_context_gap_df["end_seconds"].apply(sec_to_ts)
deep_context_gap_df["context_mid_ts"] = deep_context_gap_df[
    "context_mid_seconds"
].apply(sec_to_ts)

print(
    deep_context_gap_df[
        [
            "token_id",
            "text",
            "start_ts",
            "end_ts",
            "start_seconds",
            "end_seconds",
            "next_text",
            "next_end_seconds",
            "context_mid_ts",
            "context_mid_seconds",
        ]
    ]
    .tail(70)
    .to_string(index=False)
)
