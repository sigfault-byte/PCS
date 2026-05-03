import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    import json

    import pandas as pd

    file = "../data/interim/assemblee_nov26_2024_03_diarization_collapsed.json"

    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(type(data))
    return data, pd


@app.cell
def _(data):
    print(data.keys())
    diarization = data["diarization"]
    print(type(diarization))
    return (diarization,)


@app.cell
def _(diarization):
    if isinstance(diarization, dict):

        print(diarization.keys())

    elif isinstance(diarization, list):

        print(len(diarization))

        print(diarization[0])

    else:

        print(diarization)
    return


@app.cell
def _(diarization, pd):
    collapsed = diarization["collapsed_segments"]

    df = pd.DataFrame(collapsed)

    # flatten the nested "time" dict

    time_df = pd.json_normalize(df["time"])

    df = pd.concat([df.drop(columns=["time"]), time_df], axis=1)

    print(df.head())

    print(df.columns.tolist())
    return (df,)


@app.cell
def _(df):
    df2 = df.sort_values("start_seconds").reset_index(drop=True)

    df2["prev_end"] = df2["end_seconds"].shift(1)
    df2["gap_seconds"] = df2["start_seconds"] - df2["prev_end"]
    return (df2,)


@app.cell
def _(df2):
    gaps = df2[df2["gap_seconds"] > 0].copy()

    print(gaps[["segment_id", "prev_end", "start_seconds", "gap_seconds"]])
    return


@app.cell
def _(df, pd):
    missing_chunks = pd.DataFrame({
        "gap_start": df["end_seconds"].shift(1),
        "gap_end": df["start_seconds"],
    })

    missing_chunks["gap_seconds"] = missing_chunks["gap_end"] - missing_chunks["gap_start"]
    missing_chunks = missing_chunks[missing_chunks["gap_seconds"] > 0].reset_index(drop=True)

    print(missing_chunks)
    return (missing_chunks,)


@app.cell
def _(missing_chunks):
    def sec_to_ts(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:05.2f}"

    missing_chunks["gap_start_ts"] = missing_chunks["gap_start"].apply(sec_to_ts)
    missing_chunks["gap_end_ts"] = missing_chunks["gap_end"].apply(sec_to_ts)

    print(missing_chunks)
    return


@app.cell
def _(missing_chunks):
    missing_chunks2 = missing_chunks[missing_chunks["gap_seconds"] > 0.3]
    print(missing_chunks2)
    return


@app.cell
def _(data):
    print(data.keys())

    print(data["transcript"].keys())
    raw_segments = data["transcript"]["raw_segments"]

    print(type(raw_segments))

    print(len(raw_segments))

    print(raw_segments[0].keys())

    print(raw_segments[0])
    return


@app.cell
def _(data, pd):
    raw_segments_trans = data["transcript"]["raw_segments"]

    wdf = pd.DataFrame(raw_segments_trans)
    wtime_df = pd.json_normalize(wdf["time"])
    wdf = pd.concat([wdf.drop(columns=["time"]), wtime_df], axis=1)

    print(wdf.head())
    print(wdf.columns.tolist())
    return (wdf,)


@app.cell
def _(missing_chunks, pd, wdf):
    matches = []

    for _, gap in missing_chunks.iterrows():
        gap_start = gap["gap_start"]
        gap_end = gap["gap_end"]

        overlapping = wdf[
            (wdf["start_seconds"] < gap_end) &
            (wdf["end_seconds"] > gap_start)
        ].copy()

        if overlapping.empty:
            continue

        overlapping["gap_start"] = gap_start
        overlapping["gap_end"] = gap_end
        overlapping["gap_seconds"] = gap["gap_seconds"]

        overlapping["overlap_start"] = overlapping["start_seconds"].clip(lower=gap_start)
        overlapping["overlap_end"] = overlapping["end_seconds"].clip(upper=gap_end)
        overlapping["overlap_seconds"] = overlapping["overlap_end"] - overlapping["overlap_start"]

        matches.append(overlapping)

    gap_overlap_df = pd.concat(matches, ignore_index=True) if matches else pd.DataFrame()

    print(gap_overlap_df[[
        "segment_id",
        "start_seconds",
        "end_seconds",
        "gap_start",
        "gap_end",
        "gap_seconds",
        "overlap_seconds",
    ]])
    return


@app.cell
def _(wdf):
    wdf["mid_seconds"] = (wdf["start_seconds"] + wdf["end_seconds"]) / 2
    return


@app.cell
def _(missing_chunks, pd, wdf):
    mid_matches = []

    for _, gap2 in missing_chunks.iterrows():
        gap_start2 = gap2["gap_start"]
        gap_end2 = gap2["gap_end"]

        inside = wdf[
            (wdf["mid_seconds"] >= gap_start2) &
            (wdf["mid_seconds"] <= gap_end2)
        ].copy()

        if inside.empty:
            continue

        inside["gap_start"] = gap_start2
        inside["gap_end"] = gap_end2
        inside["gap_seconds"] = gap2["gap_seconds"]
        inside["dist_to_left"] = inside["mid_seconds"] - gap_start2
        inside["dist_to_right"] = gap_end2 - inside["mid_seconds"]

        mid_matches.append(inside)

    gap_mid_df = pd.concat(mid_matches, ignore_index=True) if mid_matches else pd.DataFrame()

    print(gap_mid_df[[
        "segment_id",
        "start_seconds",
        "end_seconds",
        "mid_seconds",
        "gap_start",
        "gap_end",
        "dist_to_left",
        "dist_to_right",
    ]])
    return


if __name__ == "__main__":
    app.run()
