import json
from collections import Counter, defaultdict

file = "data/interim/1ere-seance--questions-au-gouvernement--simplification-de-la-vie-economique-cmp--renforcer-la-s-14-avril-2026_02_transcription_whisper_segment_audit.json"

with open(file, "r", encoding="utf-8") as f:
    data = json.load(f)

transcript = data["transcript"]["raw_segments"]
diarization = data["diarization"]["raw_segments"]

diarization_transcript_matches = []

for seg_dia in diarization:
    dia_id = seg_dia["segment_id"]

    dia_start = seg_dia["time"]["start_seconds"]
    dia_end = seg_dia["time"]["end_seconds"]
    dia_duration = seg_dia["time"]["duration_seconds"]
    speaker = seg_dia["speaker_id"]

    for seg_trans in transcript:
        trans_id = seg_trans["segment_id"]

        trans_start = seg_trans["time"]["start_seconds"]
        trans_end = seg_trans["time"]["end_seconds"]
        trans_duration = seg_trans["time"]["duration_seconds"]

        # Compute temporal overlap
        overlap = min(dia_end, trans_end) - max(dia_start, trans_start)

        # No overlap
        if overlap <= 0:
            continue

        whisper_coverage = overlap / trans_duration
        diarization_coverage = overlap / dia_duration

        diarization_transcript_matches.append(
            {
                "diarization_id": dia_id,
                "transcription_id": trans_id,
                "overlap_seconds": round(overlap, 3),
                "whisper_coverage": round(whisper_coverage, 3),
                "diarization_coverage": round(diarization_coverage, 3),
                "speaker": speaker,
            }
        )

print(f"Total matches: {len(diarization_transcript_matches)}")

# Example
# for match in diarization_transcript_matches[10:20]:
#     print(match)

counter = Counter()

for match in diarization_transcript_matches:
    counter[match["transcription_id"]] += 1

print(counter.most_common(5))

# target = 3206

# matches = [m for m in diarization_transcript_matches if m["transcription_id"] == target]

# for m in sorted(matches, key=lambda x: x["overlap_seconds"], reverse=True):
#     print(m)


# Optional: lookup diarization segment by id

diarization_by_id = {d["segment_id"]: d for d in diarization}

transcript_diarization_matches = []

for wseg in transcript:
    wid = wseg["segment_id"]
    tflag = wseg["flags"]
    related = []

    for m in diarization_transcript_matches:
        if m["transcription_id"] != wid:
            continue

        dseg = diarization_by_id[m["diarization_id"]]

        related.append(
            {
                "diarization_id": m["diarization_id"],
                "speaker": m["speaker"],
                "overlap_seconds": m["overlap_seconds"],
                "whisper_coverage": m["whisper_coverage"],
                "diarization_coverage": m["diarization_coverage"],
                "diarization_duration": dseg["time"]["duration_seconds"],
                "diarization_start": dseg["time"]["start_seconds"],
                "diarization_end": dseg["time"]["end_seconds"],
            }
        )

    transcript_diarization_matches.append(
        {
            "transcription_id": wid,
            "text": wseg.get("raw_text"),
            "time": wseg["time"],
            "diarization_matches": related,
            "speaker_ids": sorted(set(r["speaker"] for r in related)),
            "flags": tflag,
        }
    )
multi_speaker_candidates = []
for i in transcript_diarization_matches:
    if len(i["speaker_ids"]) > 1:
        multi_speaker_candidates.append(
            (
                i["transcription_id"],
                i["flags"],
                i["speaker_ids"],
                i["diarization_matches"],
            )
        )

sorted_candidates = sorted(multi_speaker_candidates, key=lambda x: x[1])
for i in sorted_candidates:
    print("==========")
    print(i)
    print("==========")

print(len(multi_speaker_candidates))
no_flag = [i for i in multi_speaker_candidates if i[1] == 0]
print(len(no_flag))
