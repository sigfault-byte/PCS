# Flagging Experiment.

Following the previous logic, an analysis was run with very basic, not yet tuned thresholds:

```python
vad_partial_coverage: float = 0.80
vad_long_segment_seconds: float = 10.0
vad_long_segment_min_coverage: float = 0.60
vad_internal_gap_seconds: float = 0.75
low_avg_logprob: float = -1.0 # way too permissive / not useful alone
high_no_speech_prob: float = 0.60
high_compression_ratio: float = 2.8
short_segment_seconds: float = 0.40
long_short_text_seconds: float = 8.0
long_short_text_min_words: int = 4
long_short_text_min_chars: int = 25
silence_event_max_seconds: float = 3.0
silence_event_median_db: float = -55.0
silence_event_db_delta_p95: float = 15.0
```

This logic is ran against the VAD-1000 transcript.

```bash
jq "[.transcript.raw_segments[]]| length " 14-avril-2026_02_transcription_VAD-1000_whisper_segment_audit.json 
5745

jq "[.transcript.raw_segments[] | select(.flags > 0)] | length " 14-avril-2026_02_transcription_VAD-1000_whisper_segment_audit.json 
619
```

---
## **Flag distribution**

Among the `619` flagged segments:
```bash
PARTIAL_VAD_OVERLAP                      558   (90.15%)
INSIDE_VAD_GAP                           133   (21.49%)
MULTI_SPEAKER_CANDIDATE                  42    (6.79%)
SHORT_SEGMENT                            12    (1.94%)
LONG_WHISPER_SEGMENT_LOW_VAD_COVERAGE    11    (1.78%)
LONG_DURATION_SHORT_TEXT                 9     (1.45%)
MOSTLY_SILENCE_WITH_SHORT_EVENT          1     (0.16%)
```

The `PARTIAL_VAD_OVERLAP` threshold may be too aggressive if interpreted as a text-quality warning.

For all segments where `flags & PARTIAL_VAD_OVERLAP`, the duration distribution is:
```bash
mean   = 4.21
median = 2.06
p10    = 1.20
p90    = 4.70
min    = 0.24
max    = 435.70
std    = 24.74
```

## Flag combinations

Per-segment flag combinations:
```bash
PARTIAL_VAD_OVERLAP                                                429
INSIDE_VAD_GAP | PARTIAL_VAD_OVERLAP                               113
MULTI_SPEAKER_CANDIDATE                                             40
SHORT_SEGMENT                                                       11
INSIDE_VAD_GAP                                                      10
INSIDE_VAD_GAP | LONG_DURATION_SHORT_TEXT | LONG_WHISPER_SEGMENT_LOW_VAD_COVERAGE | PARTIAL_VAD_OVERLAP          8
MULTI_SPEAKER_CANDIDATE | PARTIAL_VAD_OVERLAP                        2
LONG_WHISPER_SEGMENT_LOW_VAD_COVERAGE | PARTIAL_VAD_OVERLAP          2
INSIDE_VAD_GAP | LONG_DURATION_SHORT_TEXT | PARTIAL_VAD_OVERLAP      1
INSIDE_VAD_GAP | LONG_WHISPER_SEGMENT_LOW_VAD_COVERAGE | PARTIAL_VAD_OVERLAP   1
PARTIAL_VAD_OVERLAP | SHORT_SEGMENT                                  1
MOSTLY_SILENCE_WITH_SHORT_EVENT | PARTIAL_VAD_OVERLAP                1
```

---

# Observation

## 1. PARTIAL_VAD_OVERLAP

Unique PARTIAL_VAD_OVERLAP segments need investigation.

This flag currently means that the Whisper segment has less than 80% coverage by VAD speech intervals.

However, this should not automatically be interpreted as a text-quality failure.

Whisper timestamps are approximate. They are quantized, shifted by decoding behavior, and affected by segmentation. VAD boundaries are also not ground truth. Therefore, a low VAD coverage ratio can indicate boundary disagreement rather than hallucination.
Whisper segment:
```json
{
  "segment_id": "wseg_005733",
  "start_token_id": 46895,
  "end_token_id": 46896,
  "time": {
    "start_seconds": 19177.1,
    "end_seconds": 19178.18,
    "duration_seconds": 1.0800000000017462,
    "start_ts": "05:19:37.10",
    "end_ts": "05:19:38.18"
  },
  "raw_text": "votant 144",
  "avg_logprob": -0.12155330959050094,
  "no_speech_prob": 0.1473388671875,
  "compression_ratio": 1.6108949416342413,
  "flags": 64 # PARTIAL_VAD_OVERLAP
}
```
Corresponding VAD interval:
```json
{
  "segment_id": "vad_003172",
  "time": {
    "start_seconds": 19177.6,
    "end_seconds": 19185.4,
    "duration_seconds": 7.80000000000291,
    "start_ts": "05:19:37.60",
    "end_ts": "05:19:45.40"
  },
  "confidence": null
}
```

Coverage calculation:
```text
Whisper: 19177.10 -> 19178.18
VAD:     19177.60 -> 19185.40

overlap_start = max(19177.10, 19177.60) = 19177.60
overlap_end   = min(19178.18, 19185.40) = 19178.18

overlap_duration = 19178.18 - 19177.60 = 0.58s
segment_duration = 19178.18 - 19177.10 = 1.08s

coverage = 0.58 / 1.08 = 0.537
```

So the segment has only ~ 53.7% VAD coverage.

After listening, Whisper did not hallucinate. The transcription is correct. 
The speaker is just speaking very fast and abruptly.

PARTIAL_VAD_OVERLAP should be treated as an audio/timestamp alignment warning,
not as an automatic text-quality rejection flag.

A segment with this flag may still contain correct text.

>However, it may be a poor candidate for speaker embedding / voice centroid computation, because the timestamp region is not fully supported by VAD.

PARTIAL_VAD_OVERLAP
possible boundary mismatch
exclude from high-confidence audio sampling
keep text unless other quality flags disagree

---
## 2.INSIDE_VAD_GAP

The previous name `INSIDE_VAD_GAP` is misleading.
Example:
```json
{
  "segment_id": "wseg_005130",
  "start_seconds": 17637.54,
  "end_seconds": 17643.06,
  "duration_seconds": 5.520000000000437,
  "text_char_count": 76,
  "text_word_count": 14,
  "vad_coverage": 0.8369565217395387,
  "diarization_overlap_seconds": 0.0,
  "diarization_overlap_region_count": 0,
  "frame_count": 55,
  "db_mean": -33.095455169677734,
  "db_p10": -48.51819610595703,
  "db_p50": -30.626523971557617,
  "db_p90": -26.276065826416016,
  "db_delta_p95": 5.520583152770996,
  "rms_mean": 0.02937719225883484,
  "zcr_mean": 0.11195401102304459,
  "avg_logprob": -0.1273013565891473,
  "no_speech_prob": 0.015411376953125,
  "compression_ratio": 1.6824817518248176,
  "flags": 128,
  "flag_names": [
    "INSIDE_VAD_GAP"
  ]
}
```

The current logic is :

```python
max_internal_gap_seconds(overlaps) >= thresholds.vad_internal_gap_seconds
```
This does not mean that the segment is inside a VAD gap... !

It means that the Whisper segment overlaps multiple VAD speech regions separated by an internal gap.

The segment metrics are healthy:
```text
vad_coverage       = 0.837
avg_logprob        = -0.127
no_speech_prob     = 0.015
compression_ratio  = 1.68
db_p90             = -26.27
zcr_mean           = 0.112
```

Listening confirms that Whisper did not fail.

Therefore, the logic is not necessarily wrong, but the flag name is misleading.
Better name: `DISCONTIGUOUS_VAD_COVERAGE`

`DISCONTIGUOUS_VAD_COVERAGE`
Whisper segment spans multiple VAD speech islands
useful topology/debug signal
possible merge/speaker-turn concern
not automatically a text-quality failure

>This flag may be useful later during merge logic, especially if the segment also crosses a diarization boundary or speaker turn.

---

## 3. SHORT_SEGMENT

The current SHORT_SEGMENT flag is too naive if based only on duration.
Some short segments are legitimate short speech bursts.

Example:
```json
{
  "segment_id": "wseg_004213",
  "time": {
    "start_seconds": 13948.2,
    "end_seconds": 13948.6,
    "duration_seconds": 0.3999999999996362
  },
  "raw_text": "pardon",
  "avg_logprob": -0.0903846141237479,
  "no_speech_prob": 0.00225067138671875,
  "compression_ratio": 1.720524017467249,
  "flags": 1
}
```
Audit metrics:
```json
{
  "segment_id": "wseg_004213",
  "duration_seconds": 0.3999999999996362,
  "text_char_count": 6,
  "text_word_count": 1,
  "vad_coverage": 1.0,
  "frame_count": 4,
  "db_mean": -27.025121688842773,
  "db_p90": -26.29539680480957,
  "rms_mean": 0.044693682342767715,
  "zcr_mean": 0.10345458984375,
  "avg_logprob": -0.0903846141237479,
  "no_speech_prob": 0.00225067138671875,
  "compression_ratio": 1.720524017467249,
  "flag_names": [
    "SHORT_SEGMENT"
  ]
}
```

This looks like a valid short burst.

However, other short segments expose impossible timestamp geometry.

 Example:
 ```json
 {
  "segment_id": "wseg_003790",
  "start_token_id": 32275,
  "end_token_id": 32281,
  "time": {
    "start_seconds": 13086.43,
    "end_seconds": 13086.63,
    "duration_seconds": 0.1999999999989086,
    "start_ts": "03:38:06.43",
    "end_ts": "03:38:06.63"
  },
  "raw_text": "arrêtons de vouloir tout réjanter depuis Paris",
  "avg_logprob": -0.06635861012448625,
  "no_speech_prob": 0.0011882781982421875,
  "compression_ratio": 1.7398119122257054,
  "flags": 1
}
 ```
Previous segment:
```json
{
  "segment_id": "wseg_003789",
  "start_token_id": 32268,
  "end_token_id": 32274,
  "time": {
    "start_seconds": 13083.81,
    "end_seconds": 13086.43,
    "duration_seconds": 2.6200000000008004,
    "start_ts": "03:38:03.81",
    "end_ts": "03:38:06.43"
  },
  "raw_text": "arrêtons de vouloir tout réjanter depuis Paris",
  "avg_logprob": -0.06635861012448625,
  "no_speech_prob": 0.0011882781982421875,
  "compression_ratio": 1.7398119122257054,
  "flags": 0
}
```

The 0.20s segment cannot physically contain the full sentence. This is not merely “short”; it is impossible timing geometry.

So the current flag should probably be renamed or replaced.

Better flag: `IMPOSSIBLE_SPEECH_RATE` or `TEXT_TOO_LONG_FOR_DURATION`

The check should depend on both duration and amount of text.

Possible heuristic:
```python
words_per_second = text_word_count / duration_seconds
chars_per_second = text_char_count / duration_seconds

if duration_seconds < 0.25 and text_word_count >= 2:
    flags |= SegmentFlag.IMPOSSIBLE_SPEECH_RATE

if chars_per_second > 45 or words_per_second > 8:
    flags |= SegmentFlag.IMPOSSIBLE_SPEECH_RATE
```
This is essentially the opposite of `LONG_DURATION_SHORT_TEXT`.

> Re run with the new logic now `IMPOSSIBLE_SPEECH_RATE = 6` 

The first is the previous `wseg_003789`

Second is `wseg_004154` and it has excellent model confidence:

- `avg_logprob = -0.083`
- `no_speech_prob = 0.006`
- `compression_ratio = 1.64`
- `vad_coverage = 1.0`

However, the segment duration is only `0.02s`, has `frame_count = 0`, and contains 3 words / 19 non-space characters.

The phrase does not exist in the audio. It appears to be a plausible continuation hallucinated after the previous segment, which ended with “Merci Madame la présidente”.

Third is `wseg_004165`
It is not an hallucination, maybe the threshold needs more tuning. 
Threshold currently is 8 words per sec, 480 words per minutes.
Here the segment is :
```json
{
  "segment_id": "wseg_004165",
  "start_token_id": 34330,
  "end_token_id": 34332,
  "time": {
    "start_seconds": 13848.16,
    "end_seconds": 13848.5,
    "duration_seconds": 0.3400000000001455,
    "start_ts": "03:50:48.16",
    "end_ts": "03:50:48.50"
  },
  "raw_text": "et de Français",
  "avg_logprob": -0.08386600028836366,
  "no_speech_prob": 0.00614166259765625,
  "compression_ratio": 1.64,
  "flags": 1
}
```

If using the byte as the information rate this would give:
```text
et          = 2
space       = 1
de          = 2
space       = 1
Français    = 9   # ç is 2 bytes

total       = 15 bytes
non-space   = 13 bytes

15 / 0.34  = 44.1 bytes/s
13 / 0.34  = 38.2 non-space bytes/s
```

Where the first segment is :
```text
"Monsieur le Président"
duration = 0.02s

total       = 22 bytes
non-space   = 20 bytes

22 / 0.02 = 1100 bytes/s
20 / 0.02 = 1000 non-space bytes/s
```

Bytes per second may be the better metric here.

At this moment in the pipeline, the semantic meaning of the text is not the main concern.  
The question is whether the amount of text attached to a very short audio duration is physically plausible.
I ll start with `max_utf8_bytes_per_second = 80.0`

Fourth is `wseg_005360`
Still caught using the words per sec, using bytes as metric to confirm if the new heuristic would work.
```Text
"c'est une deuxième condition" => 22 bytes
Duration = 0.28

29/0.28 = 103 bytes/s
```
It is *wrong* because it is not the exact wording. 
The exact sentence would be "et deuxième condition".
On the segment the `avg_logprob": -0.14762270395741142` is confident.

Fifth is `wseg_005535`
Interestingly, it is in the middle of a short hallucination, both this segment and the one that precede it are wrong.
```text
"l'un ou l'autre" -> 15 bytes
duration = 0.22

15/0.22 = 68 bytes/s
```

Last is `wseg_005663`
 Hallucination , the text does not exist.
 ```text
 "par le président de la République" -> 35 bytes
 duration: 0.1
 
 350 bytes per sec
 ```
 
So 
IMPOSSIBLE_SPEECH_RATE using words/sec:
- catches real hallucinations
- but has at least one false positive: "et de Français"

UTF-8 bytes/sec with threshold ~80:
- keeps "et de Français" unflagged: 44 bytes/s
- catches "Monsieur le Président": 1100 bytes/s
- catches "par le président de la République": 350 bytes/s
- probably catches "c'est une deuxième condition": ~103 bytes/s
- does not catch "l'un ou l'autre": 68 bytes/s

---

# Some change in the flags system and new analysis:
```text
Total segments        : 5745
Flagged segments      : 615

----=== FLAG SUMMARY ===0----

PARTIAL_VAD_OVERLAP                      558   (90.73%)
DISCONTIGUOUS_VAD_COVERAGE               133   (21.63%)
MULTI_SPEAKER_CANDIDATE                  42    (6.83%)
LONG_WHISPER_SEGMENT_LOW_VAD_COVERAGE    11    (1.79%)
LONG_DURATION_SHORT_TEXT                 9     (1.46%)
INFORMATION_RATE_TOO_HIGH                6     (0.98%)
IMPOSSIBLE_SPEECH_RATE                   6     (0.98%)
MOSTLY_SILENCE_WITH_SHORT_EVENT          1     (0.16%)
mean = 4.21
median= 2.06
p10  = 1.20
p90  = 4.70
min  = 0.24
max  = 435.70
std  = 24.74
------=== FLAG COMBINATIONS ===------

PARTIAL_VAD_OVERLAP                                              429
DISCONTIGUOUS_VAD_COVERAGE | PARTIAL_VAD_OVERLAP                 113
MULTI_SPEAKER_CANDIDATE                                          40
DISCONTIGUOUS_VAD_COVERAGE                                       10
DISCONTIGUOUS_VAD_COVERAGE | LONG_DURATION_SHORT_TEXT | LONG_WHISPER_SEGMENT_LOW_VAD_COVERAGE | PARTIAL_VAD_OVERLAP       8
IMPOSSIBLE_SPEECH_RATE | INFORMATION_RATE_TOO_HIGH                4
MULTI_SPEAKER_CANDIDATE | PARTIAL_VAD_OVERLAP                     2
LONG_WHISPER_SEGMENT_LOW_VAD_COVERAGE | PARTIAL_VAD_OVERLAP       2
IMPOSSIBLE_SPEECH_RATE                                            2
INFORMATION_RATE_TOO_HIGH                                         1
INFORMATION_RATE_TOO_HIGH | PARTIAL_VAD_OVERLAP                   1
DISCONTIGUOUS_VAD_COVERAGE | LONG_DURATION_SHORT_TEXT | PARTIAL_VAD_OVERLAP    1
DISCONTIGUOUS_VAD_COVERAGE | LONG_WHISPER_SEGMENT_LOW_VAD_COVERAGE | PARTIAL_VAD_OVERLAP                                               1
MOSTLY_SILENCE_WITH_SHORT_EVENT | PARTIAL_VAD_OVERLAP             1
```

3 renames and one new flag *INFORMATION_RATE_TOO_HIGH*

## 4. INFORMATION_RATE_TOO_HIGH

The current threshold is 60 bytes per seconds. 
> accents, *weird* letters and weird UTF-8 double bytes characters.

When filtering to look for other segment with more than 50 bytes per sec this two new were found:
```json
{
  "segment_id": "wseg_000542",
  "start_token_id": 6419,
  "end_token_id": 6428,
  "time": {
    "start_seconds": 3171.46,
    "end_seconds": 3172.42,
    "duration_seconds": 0.9600000000000364,
    "start_ts": "00:52:51.46",
    "end_ts": "00:52:52.42"
  },
  "raw_text": "les moyens de la santé mentale, en termes de psychiatres.",
  "avg_logprob": -0.19741756217611348,
  "no_speech_prob": 0.001781463623046875,
  "compression_ratio": 1.7190332326283988,
  "flags": 0
}
```
Currently not flag because its rate is ~ 51 bytes per sec.
This segment first part is an hallucination, the rest after the comma isn't.

The second  is 
```json
{
  "segment_id": "wseg_000148",
  "start_token_id": 1715,
  "end_token_id": 1723,
  "time": {
    "start_seconds": 1576.42,
    "end_seconds": 1577.02,
    "duration_seconds": 0.599999999999909,
    "start_ts": "00:26:16.42",
    "end_ts": "00:26:17.02"
  },
  "raw_text": "C'est une baisse de 12 milliards d'euros",
  "avg_logprob": -0.10148437138646842,
  "no_speech_prob": 0.0016231536865234375,
  "compression_ratio": 1.6378378378378378,
  "flags": 0
}
```

Like the previous one, it is half correct and half confidently hallucinated.

I ll add a new pass, where segment adjacent to INFORMATION_RATE_TOO_HIGH segments will get a new flag "ADJACENT_INFORMATION_RATE_ANOMALY".

As with the other flags, this is not meant to reject the segment, but to expose the current assumptions and lower trust around suspicious regions.
