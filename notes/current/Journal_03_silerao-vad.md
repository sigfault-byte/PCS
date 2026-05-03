*Wed Apr 29 11:00:50 WEST 2026*

After exploring the data more carefully, an initial lightweight VAD pass appears necessary to establish a stronger temporal ground truth before diarization/transcription merging.

The goal is not to trust VAD as final truth, but to use it as a reliable “speech exists here” anchor, allowing later merge stages to identify high-confidence regions and better expose suspicious segments (hallucinations, repeated tokens inside silence gaps, boundary drift, etc.).
# Running Silero VAD to establish speech boundaries

```bash
jq ".vad.segments[0]" 1ere-seance--questions-au-gouvernement--simplification-de-la-vie-economique-cmp--renforcer-la-s-14-avril-2026_0_vad.json
{
  "segment_id": "vad_000001",
  "time": {
    "start_seconds": 913.5,
    "end_seconds": 916.1,
    "duration_seconds": 2.6000000000000227,
    "start_ts": "00:15:13.50",
    "end_ts": "00:15:16.10"
  },
  "confidence": null
}
```
The official *compte rendu* starts at `start_time": 914.06`
> This gives a difference of only ~0.56s, which strongly suggests the VAD stage is correctly identifying the true beginning of speech activity.

---
## Total detected speech regions

```bash
jq ".vad.segments | length"
3173
```
Silero produces a highly granular segmentation.

Most segments are short conversational units (1–5s), while still preserving realistic longer uninterrupted speeches.

---
## Duration distribution

```bash
jq '.vad.segments[]
| .time.duration_seconds' file.json \
| awk '
{
    if ($1 < 1) a++
    else if ($1 < 5) b++
    else if ($1 < 15) c++
    else if ($1 < 60) d++
    else e++
}
END {
    print "<1s:", a
    print "1-5s:", b
    print "5-15s:", c
    print "15-60s:", d
    print ">60s:", e
}'
<1s: 285
1-5s: 2025
5-15s: 718
15-60s: 138
>60s: 7
```
This distribution *looks* healthy:
- most segments fall into realistic human turn lengths
- very few ultra-long segments (>60s)
- limited number of ultra-short artifacts (<1s)

---
# Comparison with previous Pyannote-VAD approach:

### **First segment comparison**
```bash
jq ".diarization.raw_segments[0]" oldFile.json
{
  "segment_id": "dia_000001",
  "time": {
    "start_seconds": 913.49159375,
    "end_seconds": 915.9890937500002,
    "duration_seconds": 2.497500000000173,
    "start_ts": "00:15:13.49",
    "end_ts": "00:15:15.99"
  },
  "speaker_id": "SPEAKER_41"
}
```
> The first detected segment is nearly identical.

---
### **Total segment count**
```bash
Silero VAD:   3173
Pyannote VAD: 1369
Difference: +1804
```
> Silero produces significantly more speech regions.

This suggests the older pipeline was likely over-merging multiple speaking turns into large continuous regions.

---
## Previous duration distribution

```bash
<1s: 176
1-5s: 548
5-15s: 396
15-60s: 197
>60s: 52
```

---

|**Bucket**|**Change**|
|---|---|
|<1s|+61.9%|
|1–5s|+269.5%|
|5–15s|+81.3%|
|15–60s|-29.9%|
|>60s|-86.5%|
Notable signal is `>60s → -86.5%`

Major reduction in very long segments and strongly indicates the previous approach was over-merging speech regions.
The large increase in 1–5s segments (+269%) supports that Silero is producing more realistic conversational boundaries.
This should hopefully improve downstream transcript-to-speaker alignment and make confidence flagging more reliable.

---

# Pipeline change 

Silero VAD will be usedd as the initial temporal anchor of the pipeline:

```text
audio
- VAD (speech exists here)
-  diarization
-  transcription
-  temporal cross check
-  merge
-  confidence flagging
(...)
```

Rather than using diarization to define speech boundaries, diarization and transcription should now be evaluated against this VAD layer.

Let's see !

---

