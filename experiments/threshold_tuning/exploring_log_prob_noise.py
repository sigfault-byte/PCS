import json
import statistics
from pathlib import Path

INPUT = Path(
    "data/interim/1ere-seance--questions-au-gouvernement--simplification-de-la-vie-economique-cmp--renforcer-la-s-14-avril-2026_03_alignment.json"
)
INPUT = Path(
    "data/audio-audit/1ere-seance--questions-au-gouvernement--simplification-de-la-vie-economique-cmp--renforcer-la-s-14-avril-2026_audio_audit.json"
)

DROP_THRESHOLD = -0.08
ABS_DELTA_THRESHOLD = 0.08
LOCAL_Z_THRESHOLD = -2.0
WINDOW = 5

with INPUT.open() as f:
    data = json.load(f)

segments = data["transcript"]["raw_segments"]


def lp(s):
    return s.get("avg_logprob")


rows = []

for i, curr in enumerate(segments):
    curr_lp = lp(curr)
    if curr_lp is None:
        continue

    reasons = []

    prev = segments[i - 1] if i > 0 else None
    next_ = segments[i + 1] if i + 1 < len(segments) else None

    # 1. Simple delta from previous segment
    if prev and lp(prev) is not None:
        prev_lp = lp(prev)
        delta = curr_lp - prev_lp
        abs_delta = abs(delta)

        if delta <= DROP_THRESHOLD:
            reasons.append(f"sharp_drop delta={delta:.3f}")

        if abs_delta >= ABS_DELTA_THRESHOLD:
            reasons.append(f"large_abs_delta abs_delta={abs_delta:.3f}")

    # 2. Local z-score against neighboring context
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

    if reasons:
        rows.append(
            {
                "index": i,
                "segment_id": curr.get("segment_id"),
                "start": curr.get("time", {}).get("start_seconds"),
                "end": curr.get("time", {}).get("end_seconds"),
                "duration": curr.get("time", {}).get("duration_seconds"),
                "avg_logprob": curr_lp,
                "prev_avg_logprob": lp(prev) if prev else None,
                "next_avg_logprob": lp(next_) if next_ else None,
                "no_speech_prob": curr.get("no_speech_prob"),
                "compression_ratio": curr.get("compression_ratio"),
                "flags": curr.get("flags"),
                "reasons": reasons,
                "text": curr.get("raw_text") or curr.get("text"),
            }
        )

print(f"Candidates: {len(rows)}")

for r in rows[:80]:
    print("=" * 80)
    print(
        f"id={r['segment_id']} "
        f"{r['start']}->{r['end']} "
        f"lp={r['avg_logprob']:.3f} "
        f"prev={r['prev_avg_logprob']} "
        f"next={r['next_avg_logprob']}"
    )
    print("reasons:", ", ".join(r["reasons"]))
    print(r["text"])
