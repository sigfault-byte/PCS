# Merging Whisper and Pyannote Segments

## Initial approach

The original pipeline was designed to work in this order:

1. run **Whisper transcription**
2. use Whisper segment timestamps as the main temporal reference
3. run **Pyannote diarization**
4. merge diarization onto the Whisper timeline

At first glance, this looked reasonable. Whisper already outputs timestamped segments, so it was tempting to treat those timestamps as the most useful time reference for later speaker assignment.

The first practical issue was that Whisper and Pyannote did **not** produce matching boundaries. Small differences were expected, but the gap was not limited to rounding noise. More importantly, a single local example was misleading: the drift pattern changed over time, and it could accumulate enough to break naïve merge assumptions.

This means Whisper segment timestamps should **not** be treated as temporal ground truth for speaker alignment.

### Example: first segment mismatch

```json
{
  "segment_id": "dia_000001",
  "time": {
    "start_seconds": 10.15596875,
    "end_seconds": 14.644718750000003,
    "duration_seconds": 4.488750000000003,
    "start_ts": "00:00:10.16",
    "end_ts": "00:00:14.64"
  },
  "speaker_id": "SPEAKER_37"
}
```

*Pyannote first segment*

```json
{
  "segment_id": "Whisper_000001",
  "start_token_id": 0,
  "end_token_id": 39,
  "time": {
    "start_seconds": 9.81,
    "end_seconds": 26.71,
    "duration_seconds": 16.9,
    "start_ts": "00:00:09.81",
    "end_ts": "00:00:26.71"
  }
}
```

*Whisper first segment*

```json
{
  "segment_id": "cdia_000001",
  "time": {
    "start_seconds": 10.15596875,
    "end_seconds": 27.06471875,
    "duration_seconds": 16.90875,
    "start_ts": "00:00:10.16",
    "end_ts": "00:00:27.06"
  }
}
```

*Collapsed diarization segment*

After collapsing adjacent diarization segments from the same speaker, the first merge still looked straightforward:

1. loop over collapsed diarization spans
2. find overlapping Whisper segments
3. if one Whisper segment overlaps multiple speakers, flag it as ambiguous

That logic works for simple cases, but it breaks down once long speaker runs contain many internal Whisper segments and small diarization fractures.

---

## What was wrong in the initial assumptions

### 1. Whisper timestamps are usable, but not authoritative

The Whisper timestamps `9.81` and `26.71` are not “wrong” in the sense of being unusable, but they are not reliable enough to serve as the authoritative timeline for speaker attribution.

Listening to the audio suggests:

- the segment start is early: no actual word begins before roughly `10s`
- the segment end is closer to reality: the speaker does finish before `27s`

So the issue is not simply that Whisper timestamps are bad.  
The issue is that their boundary behavior has a **bias**, and that bias matters when they are used as merge anchors.

### 2. A single early example was misleading

The first merged segment was not problematic because the speaker assignment was still obvious. That made the overall approach look more robust than it really was.

The real problem appeared later, when the same kind of boundary mismatch repeated across many segments. In long stretches, the drift pattern becomes a structural issue rather than a local inconvenience.

### 3. Drift is not only cumulative error

At first, it was tempting to think in terms of “small drift accumulating over time.” 
That is only part of the story.

A more precise description is:

- Whisper segment boundaries are optimized for ASR segmentation, not diarization alignment
- Pyannote segment boundaries are optimized for voice activity and speaker turns, not linguistic units
- the two systems do not fail in the same way

This creates a mismatch of **segmentation regimes**, not just a mismatch of timestamps.

### Example of an implausible short segment

```json
{
  "segment_id": "seg_005206",
  "time": {
    "start_seconds": 13078.19,
    "end_seconds": 13078.45,
    "duration_seconds": 0.2600000000002183,
    "start_ts": "03:37:58.19",
    "end_ts": "03:37:58.45"
  },
  "text": {
    "raw": "C'est tout.",
    "normalized": null,
    "language": "fr"
  }
}
```

Taken literally, this implies that the phrase *“C'est tout.”* was spoken in about `0.26` seconds. That is a strong signal that raw segment boundaries should not be interpreted too literally.

---

## Word timestamps

Another early assumption was that Whisper `word_timestamps` could provide the smallest reliable unit of temporal precision, and that all later logic could be derived from them.

The idea was to approximate a token midpoint with something like:

`(token[i].start_time + token[i+1].end_time) / 2`

The goal was to use this midpoint as an anchor and map each token to the correct diarization segment.

That idea was useful as an exploration step, but it also turned out to be too optimistic.

---
### What was misunderstood

The mistake was not “using word timestamps at all”. The mistake was assuming that token- or word-level timestamps could serve as a clean atomic ground truth.

In practice:

1. token- or word-level timing can be noisy
2. some low-level timing artifacts are corrected later in Whisper’s reconstructed segment text
3. some words appear temporally inconsistent with diarization
4. timing at that granularity is still model output, not physical truth

So word timestamps are valuable as **anchors**, but not as an unquestionable base layer.

### Empirical observations

From dataframe exploration of diarization and transcription data:

```bash
orphan token count:
823 count
823.000000
unique 119.000000
```

Out of `37,946` tokens, `823` fell into diarization gaps.

That means roughly `2%` of tokens were “orphan” tokens: they were not covered by any diarized speech span.

Some examples were much worse than a boundary-edge mismatch. A token could appear deep inside an apparent silence span:

```bash
token_id text mid_seconds dist_to_left dist_to_right prev_segment_id next_segment_id
22 7569 réglementation 2586.54 4.600281 3.702219 cdia_000048 cdia_000049
```

The word *réglementation* appears about `4.6s` after the previous diarization segment and about `3.7s` before the next one. That is not a small edge effect. 
It means at least one of the timing signals is seriously misaligned for that local region.

---
### Why the punctuation heuristic only helped a little

A next idea was to improve the transcript before merging by using a sliding window and looking for final punctuation. That helped somewhat, but only at the surface level.

The deeper problem was not punctuation. The deeper problem was that the temporal signal being treated as authoritative was not reliable enough for the job.

---

## Refactor

The refactored pipeline now starts with diarization.

This is not because Pyannote timestamps are perfectly correct. 
They are not.
Pyannote often appears to start more accurately, but it tends to **lag at the end** of segments.

Still, diarization is a better foundation for the timeline, because it is closer to the actual acoustic question being asked:

- who is speaking
- when does speech start
- when does speech stop

Whisper remains essential, but it should be treated primarily as the source of linguistic content, not as the source of authoritative speaker-boundary timing.

---

## Important clarification: what each system is good at

The current understanding is:

### Whisper is better at
- recovering text
- preserving linguistic context
- reconstructing final phrasing better than raw token-level output
- providing approximate timing anchors

### Pyannote is better at
- detecting speaker activity
- detecting speaker transitions
- providing a more realistic start boundary for speech regions

### Neither should be treated as absolute ground truth

That point matters. The correct model is not:

- Whisper is wrong, Pyannote is right

The correct model is:

- Whisper and Pyannote encode different signals
- both are noisy
- both have systematic boundary biases
- the merge logic has to account for those biases explicitly

---

## The real structural issue

The hardest cases are long same-speaker stretches.

For example:

- one speaker talks for 6–7 minutes
- Whisper creates many internal transcription segments
- diarization may still split that same speaker into smaller spans
- short gaps or micro-pauses appear between those spans

If that speaker run is collapsed too early into one large interval, useful structure is lost.

This suggests that the problem is not simply “merge segment A into segment B.” It is closer to building a unified timeline that keeps track of:

- Whisper text spans
- Whisper empty spans
- diarization speech spans
- diarization gaps
- regions where both agree
- regions where only one system reports activity

In that framing, a gap is not missing data. A gap is a signal category.

---

## Absence is also information

One of the most important design corrections is that “nothing happening” must be kept explicitly.

There are at least three distinct cases:

1. **no speaker + no text**  
   likely real silence

2. **speaker + no text**  
   hesitation, failed ASR, low-information vocalization, noise, or diarization overhang

3. **text + no speaker**  
   diarization miss, Whisper timing drift, or segmentation mismatch

These cases should not be collapsed into one generic “gap.” They mean different things and should be preserved for later analysis and heuristics.

---

## Current conclusion

The mistake was treating one model’s timestamps as if they were precise enough to serve as the reference timeline for a different task.

The better interpretation is:

- Whisper provides the best text signal
- Pyannote provides the best speaker-activity signal
- merging them requires a timeline model that preserves disagreement, not one that erases it too early

The goal is therefore not to force one system to become the ground truth for the other.

The goal is to model the timeline as overlapping evidence, then derive speaker-attributed transcript segments from that evidence.

---

## Practical implications for the next steps

A better downstream representation would likely keep interval-level annotations such as:

- `Whisper_text_present`
- `Whisper_empty`
- `diarization_voice_present`
- `diarization_gap`
- `both_active`
- `Whisper_only`
- `diarization_only`

From there, heuristics can operate on a richer and more honest model of the data.

This should make it easier to:

- detect ambiguous merge zones
- explain orphan tokens
- understand long-speaker failure modes
- decide when collapsing is safe and when it destroys structure

---

# Exploring with pandas and matplotlib

The following plot uses diarization and transcription segments. 
Not using the less accurate token timestamp from Whisper.

![[Overlap-dairi-transcript.png]]

Four states exist: 
1. silence (nothing)
2. diarization only (green)
3. transcription only (red)
4. transcript and diarization match (deep blue)


After reading the [Whisper paper](https://cdn.openai.com/papers/whisper.pdf) some important points should be noted:
- The token stream is closer to the **decoder’s local generation process**, and the higher-level segment is a more **post-structured narrative unit**. This explains both ghost tokens and missing tokens when comparing the segment with the corresponding tokens.
- Inherently timestamps come from:
	- 20 ms quantization
	- timestamp discretization
	- decoding lag / local imprecision
	- chunk edge weirdness

This confirms the overall behavior noticed during the various runs. 

---
# New Merging Behavior Plan

- keep **gaps** as a core signal
- use **repetition** as a suspiciousness amplifier
- trust **overlap-first merging**
- treat **orphans as second-class objects**
- allow a small **rescue lag window** (~20–50 ms)
- flag uncertain leftovers rather than force-merging them

---

# Experiments

## Diarization Collapse Logic

Collapsed diarization segments follow this rule:

- two consecutive segments belong to the same speaker
- the gap between them is less than 2 seconds

-> merge them

```text
cdia_000001 SPEAKER_37 10.15596875 14.644718750000003 ['dia_000001']
cdia_000002 SPEAKER_37 17.074718750000002 27.06471875 ['dia_000002', 'dia_000003']
cdia_000003 SPEAKER_02 29.44409375 145.24034375000002 ['dia_000004', 'dia_000005', ..., 'dia_000031']
cdia_000004 SPEAKER_37 145.56096875 150.21846875 ['dia_000032']
```
---
## **Gap Tracking**

All meaningful gaps between collapsed diarization segments are preserved.

```text
True gaps (>2s): 97
(14.644718750000003, 17.074718750000002)
(27.06471875, 29.44409375)
(954.9365937500002, 957.53534375)
(1110.57471875, 1112.5828437500002)
(1762.1015937500001, 1765.2572187500002)
(1900.5272187500002, 1903.2947187500001)
...
```

---
 
## Transcript Segment Assignment

For each transcript segment:
1. compute its midpoint
2. check whether the midpoint falls inside a diarization gap

If yes:
-> do not merge it immediately
-> store its segment_id inside a dedicated orphan set for later analysis

Transcript segment ids in gap:
```text
[
  'wseg_001121', 'wseg_001409', 'wseg_001410', 'wseg_001411', 'wseg_001412', 'wseg_001413', 'wseg_001414', 'wseg_001415', 'wseg_001416', 'wseg_001417', 'wseg_001418', 'wseg_001419', 'wseg_001420', 'wseg_001421', 'wseg_001422', 'wseg_001423', 'wseg_001424', 'wseg_001425', 'wseg_001426', 'wseg_001427', 'wseg_001428', 'wseg_001429', 'wseg_001430', 'wseg_001431',
  ...
]
```

---
## **Repetition Detection**

The range: `wseg_001409` -> `wseg_001431`
turned out to be a repetition of the sentence:
“merci à tous”
during a break in the session.

This is exactly the kind of suspicious pattern that should be filtered early.

A consecutive block like this is rarely a merge issue, it usually indicates Whisper repetition or decoder drift.

However, `wseg_001121` revealed a legitimate issue.
This segment was classified as falling inside the gap: `(2473.33221875, 2475.40784375)`
Between:
```text
cdia_000047 SPEAKER_27 2412.34596875 2473.33221875
cdia_000048 SPEAKER_27 2475.40784375 2540.0390937500006
```
But both diarization segments belong to the same speaker.
The transcript midpoint is:
```text
(2472.98 + 2477.0) / 2 = 2474.99
```
which falls inside that gap.

This strongly suggests that midpoint-only assignment is too rigid here.
This case should probably be rescued by:
- overlap-first matching
- same-speaker continuity
- or a small lag tolerance window

because semantically, it clearly belongs to the same speech flow.
> It should be classified as a true orphan.

---
## **Whisper Failure Region**

While inspecting transcript continuity, a much larger issue appeared:
```text
wseg_004546 12338.08 12338.52 Pardon.
wseg_004725 13141.40 13144.70 Sous-titrage ST' 501
wseg_004726 13170.54 13174.70 Sous-titrage ST' 501
wseg_004727 13201.40 13204.70 Sous-titrage ST' 501
wseg_004728 13231.40 13234.70 Sous-titrage ST' 501
wseg_004729 13234.72 13264.70 Sous-titrage ST' 501
```
This phrase does not exist in the source audio.

It appeared repeatedly after a break in the recording, with a suspicious ~30-second rhythm.
After checking the actual audio timestamps:
-> Whisper was missing approximately 5 minutes of real transcript

Transcription failure caused upstream.

---
## **Transcription Fix**

The original Faster-Whisper settings were:
```python
beam_size = 5
vad_filter = True
vad_min_silence_duration_ms = 500
word_timestamps = True
```
Changed to `vad_min_silence_duration_ms = 2000`

-> the `"Sous-titrage ST' 501"` failure disappeared completely
The repetition heuristic still catches normal harmless repetitions like _“merci”_, but the catastrophic decoder failure is gone.

This confirms that the issue was caused by overly aggressive silence segmentation rather than beam search or merge logic.

---

## Final Validation of Remaining Orphans

After improving the assignment logic using:

- overlap-first matching
- midpoint fallback
- gap classification
- repetition filtering

only **two unmatched transcript segments** remained:

```text
Unmatched transcript segment ids:
{
  "wseg_006189",
  "wseg_006190"
}
```
Inspection showed:
```text
wseg_006189 → "Sous-titrage ST' 501"
wseg_006190 → "."
```
Neither of these contains meaningful parliamentary speech! 

---
### Remark **`wseg_006189`**

Whisper Hallucination:
The segment: `"Sous-titrage ST' 501"`
appears at the very end of the recording, as the final token before the audio ends.
(was appearing before in a long silence)

The decoder may fall back to highly probable patterns seen during training.

Since Whisper was trained on a very large amount of web video and broadcast material, subtitle-related artifacts can occasionally appear as false positives.
And this is somehow hilarious, it is generating a subtitle or broadcast caption marker learned from subtitled media.

---