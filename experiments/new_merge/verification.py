import json

file = "data/interim/1ere-seance--questions-au-gouvernement--simplification-de-la-vie-economique-cmp--renforcer-la-s-14-avril-2026_03_alignment.json"

with open(file, "r", encoding="utf-8") as f:
    data = json.load(f)

transcript = data["transcript"]["raw_segments"]
alignment = data["alignment"]["transcript_diarization_matches"]


matches = {m["transcript_segment_id"]: m for m in alignment}

for segment in transcript:
    match = matches[segment["segment_id"]]

    if len(match["speaker_ids"]) <= 1:
        continue

    print("=" * 80)
    print(
        f"[{segment['segment_id']}] {segment['time']['start_ts']} → {segment['time']['end_ts']}"
    )
    print(f"speaker: {match['probable_speaker_id']}")
    print(f"confidence: {match['speaker_confidence']}")
    print(f"speakers: {', '.join(match['speaker_ids'])}")
    print(f"overlap: {match['speaker_overlap_seconds']}")
    print(f"evidence: {match['speaker_evidence_score']}")
    print()
    print(segment["raw_text"])
