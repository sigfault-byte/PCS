import json

file = "data/interim/1ere-seance--questions-au-gouvernement--simplification-de-la-vie-economique-cmp--renforcer-la-s-14-avril-2026_03_alignment.json"

with open(file, "r", encoding="utf-8") as f:
    data = json.load(f)

transcript = data["transcript"]["raw_segments"]
alignment = data["alignment"]["transcript_diarization_matches"]

turns = []
current_turn = None

for segment, match in zip(transcript, alignment):
    speaker_id = match["probable_speaker_id"]

    if current_turn is None or current_turn["speaker_id"] != speaker_id:
        if current_turn is not None:
            turns.append(current_turn)

        current_turn = {
            "speaker_id": speaker_id,
            "start_seconds": segment["time"]["start_seconds"],
            "end_seconds": segment["time"]["end_seconds"],
            "time_start": segment["time"]["start_ts"],
            "time_end": segment["time"]["start_ts"],
            "transcript_segments": [],
            "diarization_segments": [],
            "flags": 0,
            "raw_text": "",
        }

    current_turn["end_seconds"] = segment["time"]["end_seconds"]

    current_turn["transcript_segments"].append(
        {
            "segment_id": segment["segment_id"],
            "flags": segment.get("flags", 0),
        }
    )

    current_turn["diarization_segments"].extend(
        {
            "segment_id": diarization_segment_id,
            "flags": 0,
        }
        for diarization_segment_id in match.get("diarization_segment_ids", [])
    )

    current_turn["flags"] |= segment.get("flags", 0)

    text = segment.get("raw_text") or segment.get("text") or ""
    if text:
        current_turn["raw_text"] += (" " if current_turn["raw_text"] else "") + text

if current_turn is not None:
    turns.append(current_turn)

print(f"created {len(turns)} turns")

for turn in turns:
    print("=" * 80)
    print(f"{turn['start_seconds']} → {turn['end_seconds']}")
    print(f"{turn['time_start']} → {turn['time_end']}")
    print(f"speaker: {turn['speaker_id']}")
    print(f"flags: {turn['flags']}")
    print(f"transcript segments: {turn['transcript_segments']}")
    print(f"diarization segments: {turn['diarization_segments']}")
    print()
    print(turn["raw_text"])
