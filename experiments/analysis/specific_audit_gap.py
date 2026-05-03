import json

import pandas as pd


# ---------- helpers ----------
def sec_to_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


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

# ---------- whisper raw tokens ----------
raw_tokens = data["transcript"]["raw_tokens"]
wdf = pd.DataFrame(raw_tokens)

# flatten nested "time" dict if present
if "time" in wdf.columns:
    wdf_time = pd.json_normalize(wdf["time"])
    wdf = pd.concat([wdf.drop(columns=["time"]), wdf_time], axis=1)

# sort by token start time
wdf = wdf.sort_values("start_seconds").reset_index(drop=True)

# next token end time
wdf["next_end_seconds"] = wdf["end_seconds"].shift(-1)

# contextual midpoint:
# for token i -> (start_i + end_{i+1}) / 2
# for last token -> fallback to (start_i + end_i) / 2
wdf["context_mid_seconds"] = (
    wdf["start_seconds"] + wdf["next_end_seconds"].fillna(wdf["end_seconds"])
) / 2

print("\n=== WHISPER RAW TOKENS ===")
print(wdf.head())
print(wdf.columns.tolist())
print(f"count: {len(wdf)}")


# ---------- mark tokens whose context midpoint falls inside a diarization gap ----------
wdf["is_in_gap"] = False
wdf["gap_start"] = pd.NA
wdf["gap_end"] = pd.NA
wdf["gap_seconds"] = pd.NA
wdf["gap_start_ts"] = pd.NA
wdf["gap_end_ts"] = pd.NA

for _, gap in gaps.iterrows():
    mask = (wdf["context_mid_seconds"] >= gap["gap_start"]) & (
        wdf["context_mid_seconds"] <= gap["gap_end"]
    )

    wdf.loc[mask, "is_in_gap"] = True
    wdf.loc[mask, "gap_start"] = gap["gap_start"]
    wdf.loc[mask, "gap_end"] = gap["gap_end"]
    wdf.loc[mask, "gap_seconds"] = gap["gap_seconds"]
    wdf.loc[mask, "gap_start_ts"] = gap["gap_start_ts"]
    wdf.loc[mask, "gap_end_ts"] = gap["gap_end_ts"]

print("\n=== TOKENS MARKED WITH GAP MEMBERSHIP ===")
print(wdf.head())
print("\nTokens in gap:", int(wdf["is_in_gap"].sum()))
print("Tokens not in gap:", int((~wdf["is_in_gap"]).sum()))


orphans_df = wdf[wdf["is_in_gap"]].copy()

print("\n=== TOKENS WHOSE CONTEXT MIDPOINT IS IN A GAP ===")
print(
    orphans_df[
        [
            "token_id",
            "raw_token",
            "start_seconds",
            "end_seconds",
            "context_mid_seconds",
            "gap_start_ts",
            "gap_end_ts",
            "gap_seconds",
        ]
    ]
    .head(50)
    .to_string(index=False)
)

print(f"\norphan token count: {len(orphans_df)}")

orphans_df = wdf[wdf["is_in_gap"]].copy()

orphans_df["dist_to_left"] = orphans_df["context_mid_seconds"] - orphans_df["gap_start"]

orphans_df["dist_to_right"] = orphans_df["gap_end"] - orphans_df["context_mid_seconds"]

orphans_df["dist_to_left"] = pd.to_numeric(orphans_df["dist_to_left"], errors="coerce")

orphans_df["dist_to_right"] = pd.to_numeric(
    orphans_df["dist_to_right"], errors="coerce"
)

print(orphans_df["gap_seconds"].describe())
print(orphans_df[["dist_to_left", "dist_to_right"]].describe())

print(orphans_df[["dist_to_left", "dist_to_right"]].describe())

orphans_df["token_norm"] = (
    orphans_df["raw_token"]
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace(r"[^\wÀ-ÿ']", "", regex=True)
)

print(orphans_df["token_norm"].value_counts().head(30))

print(orphans_df["raw_token"].str.strip().value_counts().head(20))

print(orphans_df["gap_seconds"].value_counts().head(20))
print(
    orphans_df[["gap_start_ts", "gap_end_ts", "gap_seconds"]]
    .drop_duplicates()
    .sort_values("gap_seconds", ascending=False)
    .head(20)
)
