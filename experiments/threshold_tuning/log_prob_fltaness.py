import json
import statistics
from pathlib import Path

ALIGNMENT_INPUT = Path(
    "data/interim/1ere-seance--questions-au-gouvernement--simplification-de-la-vie-economique-cmp--renforcer-la-s-14-avril-2026_03_alignment.json"
)

AUDIO_AUDIT_INPUT = Path(
    "data/audio-audit/1ere-seance--questions-au-gouvernement--simplification-de-la-vie-economique-cmp--renforcer-la-s-14-avril-2026_audio_audit.json"
)

DROP_THRESHOLD = -0.08
ABS_DELTA_THRESHOLD = 0.08
LOCAL_Z_THRESHOLD = -2.0
WINDOW = 5

MIN_DB = -50.0

FLATNESS_Z_THRESHOLD = 2.0
FLATNESS_DELTA_THRESHOLD = 0.15


def load_json(path):
    with path.open() as f:
        return json.load(f)


alignment = load_json(ALIGNMENT_INPUT)
audio_audit = load_json(AUDIO_AUDIT_INPUT)

segments = alignment["transcript"]["raw_segments"]
frames = audio_audit["frames"]


def lp(s):
    return s.get("avg_logprob")


def overlaps(a_start, a_end, b_start, b_end):
    return a_start < b_end and b_start < a_end


def segment_audio_stats(start, end):
    seg_frames = [
        f
        for f in frames
        if overlaps(
            start,
            end,
            f["frame_start_seconds"],
            f["frame_end_seconds"],
        )
    ]

    voiced_frames = [f for f in seg_frames if f.get("db", -999) >= MIN_DB]

    if not voiced_frames:
        return {
            "frame_count": len(seg_frames),
            "voiced_frame_count": 0,
            "flatness_mean": None,
            "flatness_max": None,
            "flatness_std": None,
            "flatness_delta_max": None,
            "flatness_spike": False,
            "audio_reasons": ["no_voiced_frames"],
        }

    flatness = [f["spectral_flatness"] for f in voiced_frames]

    deltas = [abs(flatness[i] - flatness[i - 1]) for i in range(1, len(flatness))]

    flatness_mean = statistics.mean(flatness)
    flatness_max = max(flatness)
    flatness_std = statistics.pstdev(flatness) if len(flatness) > 1 else 0.0
    flatness_delta_max = max(deltas) if deltas else 0.0

    audio_reasons = []

    if flatness_std > 0:
        z_max = (flatness_max - flatness_mean) / flatness_std
        if z_max >= FLATNESS_Z_THRESHOLD:
            audio_reasons.append(f"flatness_local_spike z={z_max:.2f}")

    if flatness_delta_max >= FLATNESS_DELTA_THRESHOLD:
        audio_reasons.append(f"flatness_delta_spike delta={flatness_delta_max:.3f}")

    return {
        "frame_count": len(seg_frames),
        "voiced_frame_count": len(voiced_frames),
        "flatness_mean": flatness_mean,
        "flatness_max": flatness_max,
        "flatness_std": flatness_std,
        "flatness_delta_max": flatness_delta_max,
        "flatness_spike": bool(audio_reasons),
        "audio_reasons": audio_reasons,
    }


rows = []

for i, curr in enumerate(segments):
    curr_lp = lp(curr)
    if curr_lp is None:
        continue

    reasons = []

    prev = segments[i - 1] if i > 0 else None
    next_ = segments[i + 1] if i + 1 < len(segments) else None

    if prev and lp(prev) is not None:
        prev_lp = lp(prev)
        delta = curr_lp - prev_lp
        abs_delta = abs(delta)

        if delta <= DROP_THRESHOLD:
            reasons.append(f"sharp_drop delta={delta:.3f}")

        if abs_delta >= ABS_DELTA_THRESHOLD:
            reasons.append(f"large_abs_delta abs_delta={abs_delta:.3f}")

    lo = max(0, i - WINDOW)
    hi = min(len(segments), i + WINDOW + 1)

    local_values = [
        lp(s)
        for j, s in enumerate(segments[lo:hi], start=lo)
        if j != i and lp(s) is not None
    ]

    if len(local_values) >= 3:
        mu = statistics.mean(local_values)
        sigma = statistics.pstdev(local_values)

        if sigma > 0:
            z = (curr_lp - mu) / sigma

            if z <= LOCAL_Z_THRESHOLD:
                reasons.append(f"local_low_z z={z:.2f} mu={mu:.3f} sigma={sigma:.3f}")

    if not reasons:
        continue

    start = curr.get("time", {}).get("start_seconds")
    end = curr.get("time", {}).get("end_seconds")

    audio = segment_audio_stats(start, end)

    rows.append(
        {
            "index": i,
            "segment_id": curr.get("segment_id"),
            "start": start,
            "end": end,
            "duration": curr.get("time", {}).get("duration_seconds"),
            "avg_logprob": curr_lp,
            "prev_avg_logprob": lp(prev) if prev else None,
            "next_avg_logprob": lp(next_) if next_ else None,
            "no_speech_prob": curr.get("no_speech_prob"),
            "compression_ratio": curr.get("compression_ratio"),
            "flags": curr.get("flags"),
            "logprob_reasons": reasons,
            "audio_reasons": audio["audio_reasons"],
            "flatness_spike": audio["flatness_spike"],
            "flatness_mean": audio["flatness_mean"],
            "flatness_max": audio["flatness_max"],
            "flatness_std": audio["flatness_std"],
            "flatness_delta_max": audio["flatness_delta_max"],
            "voiced_frame_count": audio["voiced_frame_count"],
            "text": curr.get("raw_text") or curr.get("text"),
        }
    )

print(f"Logprob candidates: {len(rows)}")
print(f"With flatness spike: {sum(1 for r in rows if r['flatness_spike'])}")

for r in rows:
    print("=" * 80)
    print(
        f"id={r['segment_id']} "
        f"{r['start']}->{r['end']} "
        f"lp={r['avg_logprob']:.3f} "
        f"prev={r['prev_avg_logprob']} "
        f"next={r['next_avg_logprob']}"
    )
    print("logprob:", ", ".join(r["logprob_reasons"]))

    if r["audio_reasons"]:
        print("audio:", ", ".join(r["audio_reasons"]))
    else:
        print("audio: no flatness spike")

    print(
        f"flatness mean={r['flatness_mean']} "
        f"max={r['flatness_max']} "
        f"std={r['flatness_std']} "
        f"delta_max={r['flatness_delta_max']} "
        f"voiced_frames={r['voiced_frame_count']}"
    )
    print(r["text"])
