
Initially, the merge strategy used diarization segments as the principal structure.
Whisper segments were attached to pyannote segments because pyannote timestamps were assumed to be more temporally accurate.

In practice, this produced unstable merges.

The issue was not timestamp precision itself, but the fact that diarization fragmentation was treated as structural truth. A single Whisper segment could overlap many small diarization fragments:
```text
[(3206, 16), (2534, 8), (163, 5), (1400, 5), (2196, 5)]
```
For example, Whisper segment 3206 overlapped 16  ( ! )diarization segments despite representing a short and semantically coherent sentence.

This revealed an architectural issue: speaker segmentation is acoustically unstable
text segmentation is semantically stable

As a result, the merge logic was inverted.

Instead of merging Whisper into diarization, the pipeline now:
```text
uses Whisper text segments as the principal structure
and treats diarization as a dynamic annotation layer
```

# Text as the **principal structure** and the diarization as a **dynamic annotation layer**.

After inversion of the merge strategy:
* 79 / 3300 Whisper segments contain multiple speaker candidates
* 44 / 79 were already flagged by previous audit stages

This means the current audit heuristics already detect ~55% of structurally noisy speaker regions independently from the new overlap analysis.

The flags are :
	- 4 segments have flag `2048` -> `2^11` -> PARTIAL_VAD_OVERLAP  
	- 4 segments have flag `6144` -> `2^11` and `2^12` -> PARTIAL_VAD_OVERLAP and DISCONTIGUOUS_VAD_COVERAGE  
	- 33 segments have flag `2147483648` -> `2^31` -> MULTI_SPEAKER_CANDIDATE  
	- 1 segment has flag `2147483651` ->`2^0` and `2^1` and `2^31` -> IMPOSSIBLE_SPEECH_RATE and INFORMATION_RATE_TOO_HIGH and MULTI_SPEAKER_CANDIDATE  
	- 1 segment has flag `14336` ->  -> `2^11` and `2^12` and `2^13` -> PARTIAL_VAD_OVERLAP and DISCONTIGUOUS_VAD_COVERAGE and LONG_WHISPER_SEGMENT_LOW_VAD_COVERAGE  
	- 1 segment has flag `2147497984` -> `2^11` and `2^12` and `2^13` and 2^31 -> PARTIAL_VAD_OVERLAP and DISCONTIGUOUS_VAD_COVERAGE and LONG_WHISPER_SEGMENT_LOW_VAD_COVERAGE and MULTI_SPEAKER_CANDIDATE  

## Structural observation

The remaining 35 unflagged cases were not random.

SPEAKER_40 dominates the first half of the unresolved overlaps.
SPEAKER_36 dominates the second half.
Manual inspection revealed:

* SPEAKER_40 is the assembly president
* SPEAKER_36 is likely an alias split of the same speaker produced by diarization instability

Many unresolved overlaps correspond to procedural speech handoffs such as:
```text
"Merci monsieur le député."
"Merci madame la présidente."
```
These short parliamentary formulas naturally occur at speaker transitions and therefore create local overlap ambiguity.

---

The 35 leftovers are ( tuple of whisper segment_id, flag, speaker_ids)
```text
(120, 0, ['SPEAKER_22', 'SPEAKER_40']), (163, 0, ['SPEAKER_37', 'SPEAKER_40']) (166, 0, ['SPEAKER_40', 'SPEAKER_56']), (214, 0, ['SPEAKER_40', 'SPEAKER_56']) (339, 0, ['SPEAKER_18', 'SPEAKER_40']), (391, 0, ['SPEAKER_05', 'SPEAKER_40']) (442, 0, ['SPEAKER_05', 'SPEAKER_40']), (472, 0, ['SPEAKER_40', 'SPEAKER_45']) (555, 0, ['SPEAKER_39', 'SPEAKER_40']), (742, 0, ['SPEAKER_26', 'SPEAKER_40']) (843, 0, ['SPEAKER_40', 'SPEAKER_63']), (868, 0, ['SPEAKER_40', 'SPEAKER_63']) (1017, 0, ['SPEAKER_01', 'SPEAKER_40']), (1071, 0, ['SPEAKER_04', 'SPEAKER_40']) (1094, 0, ['SPEAKER_04', 'SPEAKER_33']), (1099, 0, ['SPEAKER_06', 'SPEAKER_40']) (1499, 0, ['SPEAKER_19', 'SPEAKER_36']), (1531, 0, ['SPEAKER_25', 'SPEAKER_36']) (1532, 0, ['SPEAKER_36', 'SPEAKER_62']), (1549, 0, ['SPEAKER_36', 'SPEAKER_62']) (1550, 0, ['SPEAKER_35', 'SPEAKER_36']), (1924, 0, ['SPEAKER_23', 'SPEAKER_36']) (2405, 0, ['SPEAKER_36', 'SPEAKER_58']), (2503, 0, ['SPEAKER_36', 'SPEAKER_50']) (2539, 0, ['SPEAKER_06', 'SPEAKER_38']), (2540, 0, ['SPEAKER_06', 'SPEAKER_36']) (2720, 0, ['SPEAKER_03', 'SPEAKER_36']), (2882, 0, ['SPEAKER_02', 'SPEAKER_36']) (3141, 0, ['SPEAKER_03', 'SPEAKER_36']), (3206, 0, ['SPEAKER_29', 'SPEAKER_36']) (3235, 0, ['SPEAKER_21', 'SPEAKER_36']), (3240, 0, ['SPEAKER_30', 'SPEAKER_31']) (3262, 0, ['SPEAKER_30', 'SPEAKER_36']), (3298, 0, ['SPEAKER_36', 'SPEAKER_41']) (3300, 0, ['SPEAKER_06', 'SPEAKER_36'])
```

The segment 3206, previously associated with 16 diarization segment, is now a *simple* 2 speakers overlap.

Before investigation it, out of the 35 segments:
`SPEAKER_40` dominates the first half
`SPEAKER_36 `dominates the second half

`SPEAKER_40` is the assembly president. 
`SPEAKER_36` is a speaker alias split from `SPEAKER_40`

Segments `120`, `163`, `214` example:
```json
{
  "segment_id": 120,
...
  "raw_text": "Merci monsieur le député.",
...
  "flags": 0
},
...
{
  "segment_id": 163,
...
  "raw_text": "Merci beaucoup Madame la Ministre.",
...
  "flags": 0
},
...
{
  "segment_id": 214,
...
  "raw_text": "Merci beaucoup, madame la députée.",
...
  "flags": 0
}
```
All of them are from the president.

Segments `1532` , `2405` `3206`:
```json
{
  "segment_id": 1532,
...
  "raw_text": "Merci madame la présidente.",
...
  "flags": 0
},
...
{
  "segment_id": 2405,
...
  "raw_text": "Favorable. Monsieur Meurin.",
...
  "flags": 0
},
...
{
  "segment_id": 3206,
...
  "raw_text": "Merci monsieur le rapporteur. Monsieur le ministre.",
...
  "flags": 0
}

```

The segment `1532` is not from the president, but she was speaking right before it.
The second and the third are from her.

## Electing the probable speaker

Pyannote segment below `0.8` in the pipeline are not saved for embedding, because they are too noisy.

To elect the probable speaker on a multi-speaker segment here, the approach will be:

For each speaker, compute the total duration of the overlapping.
Look how many diarization segment are in, and who they belong to.
Sum the similar speaker segment duration.
But, weight the duration depending on the speaker segment continuous duration.

## Applying the logic with to various segments

### `120`
Whisper segment from pipeline:
```json
{
  "segment_id": 120,
  "start_token_id": 1444,
  "end_token_id": 1447,
  "time": {
    "start_seconds": 1494.6,
    "end_seconds": 1496.52,
    "duration_seconds": 1.9200000000000728,
    "start_ts": "00:24:54.60",
    "end_ts": "00:24:56.52"
  },
  "raw_text": "Merci monsieur le député.",
  "avg_logprob": -0.17717803536039411,
  "no_speech_prob": 0.009857177734375,
  "compression_ratio": 1.6812865497076024,
  "flags": 0
}
```

From current experiment:
```json
(120, 0, ['SPEAKER_22', 'SPEAKER_40'], [
{'diarization_id': 23, 'speaker': 'SPEAKER_22', 'overlap_seconds': 0.033, 'whisper_coverage': 0.017, 'diarization_coverage': 0.0, 'diarization_duration': 94.34812499999998, 'diarization_start': 1400.2847187500001, 'diarization_end': 1494.6328437500001}, 
{'diarization_id': 24, 'speaker': 'SPEAKER_40', 'overlap_seconds': 1.533, 'whisper_coverage': 0.798, 'diarization_coverage': 0.275, 'diarization_duration': 5.568749999999909, 'diarization_start': 1494.9872187500002, 'diarization_end': 1500.5559687500001}
])
```

Whisper start: 1494.6
Whisper end: 1496.52

Speaker 22 diarization starts at: 1400.28
Speaker 22 diarization ends at: 1494.63
Speaker 40 diarization starts at: 1494.98
Speaker 40 diarization ends at: 1500.55

Speaker 40 overlaps for 1.5 sec, with one diarization segment -> He is the most probable speaker

### `1532`
Whisper from pipeline:
```json
{
  "segment_id": 1532,
  "start_token_id": 21018,
  "end_token_id": 21021,
  "time": {
    "start_seconds": 9088.65,
    "end_seconds": 9090.17,
    "duration_seconds": 1.5200000000004366,
    "start_ts": "02:31:28.65",
    "end_ts": "02:31:30.17"
  },
  "raw_text": "Merci madame la présidente.",
  "avg_logprob": -0.08803879392558131,
  "no_speech_prob": 0.002216339111328125,
  "compression_ratio": 1.632258064516129,
  "flags": 0
}
```

From current experiment:
```json
(1532, 0, ['SPEAKER_36', 'SPEAKER_62'], [
{'diarization_id': 542, 'speaker': 'SPEAKER_36', 'overlap_seconds': 0.017, 'whisper_coverage': 0.011, 'diarization_coverage': 1.0, 'diarization_duration': 0.016875000001164153, 'diarization_start': 9088.855343750001, 'diarization_end': 9088.872218750003}, 
{'diarization_id': 543, 'speaker': 'SPEAKER_62', 'overlap_seconds': 1.298, 'whisper_coverage': 0.854, 'diarization_coverage': 0.02, 'diarization_duration': 63.93937499999811, 'diarization_start': 9088.872218750003, 'diarization_end': 9152.81159375}
])
```

Here it is the opposite. 
`SPEAKER_62` is the most probable speaker

### `3124`
Whisper from pipeline:
```json
{
  "segment_id": 3124,
  "start_token_id": 43413,
  "end_token_id": 43419,
  "time": {
    "start_seconds": 18060.7,
    "end_seconds": 18064.54,
    "duration_seconds": 3.8400000000001455,
    "start_ts": "05:01:00.70",
    "end_ts": "05:01:04.54"
  },
  "raw_text": "Je me doute. Le 71, madame Fossillon.",
  "avg_logprob": -0.1596765324734805,
  "no_speech_prob": 0.0005612373352050781,
  "compression_ratio": 1.5805243445692885,
  "flags": 2147483648
}
```

From current experiment:
```json
(3124, 2147483648, ['SPEAKER_07', 'SPEAKER_36'], [
{'diarization_id': 1300, 'speaker': 'SPEAKER_36', 'overlap_seconds': 0.036, 'whisper_coverage': 0.009, 'diarization_coverage': 0.014, 'diarization_duration': 2.5987499999973807, 'diarization_start': 18058.137218750002, 'diarization_end': 18060.73596875}, 
{'diarization_id': 1301, 'speaker': 'SPEAKER_07', 'overlap_seconds': 0.002, 'whisper_coverage': 0.001, 'diarization_coverage': 0.003, 'diarization_duration': 0.7256249999991269, 'diarization_start': 18059.976593749998, 'diarization_end': 18060.702218749997}, 
{'diarization_id': 1302, 'speaker': 'SPEAKER_07', 'overlap_seconds': 0.084, 'whisper_coverage': 0.022, 'diarization_coverage': 1.0, 'diarization_duration': 0.08437499999854481, 'diarization_start': 18060.73596875, 'diarization_end': 18060.820343749998}, 
{'diarization_id': 1303, 'speaker': 'SPEAKER_36', 'overlap_seconds': 1.451, 'whisper_coverage': 0.378, 'diarization_coverage': 1.0, 'diarization_duration': 1.4512499999982538, 'diarization_start': 18061.05659375, 'diarization_end': 18062.507843749998}, 
{'diarization_id': 1304, 'speaker': 'SPEAKER_36', 'overlap_seconds': 1.543, 'whisper_coverage': 0.402, 'diarization_coverage': 0.847, 'diarization_duration': 1.8224999999947613, 'diarization_start': 18062.997218750003, 'diarization_end': 18064.819718749997}])
```
`SPEAKER_07` has 2  segments : `[0.002, 0.084]
`SPEAKER_36` has 3 segments. `[0.036, 1.45, 1.5]`

Even if `SPEAKER_07` had 5 micro segment of `0.3`s summing up to `1.5` s
And `SPEAKER_36` had a single segment of `1.4` s 
The current algorithm would still chose `SPEAKER_36`

### `1362`
Whisper from pipeline:
```json
{
  "segment_id": 1362,
  "start_token_id": 18346,
  "end_token_id": 18363,
  "time": {
    "start_seconds": 8131.2,
    "end_seconds": 8161.93,
    "duration_seconds": 30.730000000000473,
    "start_ts": "02:15:31.20",
    "end_ts": "02:16:01.93"
  },
  "raw_text": "Madame la présidente, messieurs les ministres, monsieur le président, messieurs les rapporteurs de la commission spéciale, chers collègues.",
  "avg_logprob": -0.1009954634693361,
  "no_speech_prob": 0.004535675048828125,
  "compression_ratio": 1.6570397111913358,
  "flags": 2147497984
}
```

From current experiment:
```json
(1362, 2147497984, ['SPEAKER_00', 'SPEAKER_09', 'SPEAKER_36'], [
{'diarization_id': 453, 'speaker': 'SPEAKER_36', 'overlap_seconds': 0.134, 'whisper_coverage': 0.004, 'diarization_coverage': 0.01, 'diarization_duration': 13.078125, 'diarization_start': 8118.25596875, 'diarization_end': 8131.33409375}, 
{'diarization_id': 454, 'speaker': 'SPEAKER_09', 'overlap_seconds': 1.063, 'whisper_coverage': 0.035, 'diarization_coverage': 1.0, 'diarization_duration': 1.063124999998763, 'diarization_start': 8150.942843750001, 'diarization_end': 8152.00596875}, 
{'diarization_id': 455, 'speaker': 'SPEAKER_00', 'overlap_seconds': 0.304, 'whisper_coverage': 0.01, 'diarization_coverage': 1.0, 'diarization_duration': 0.3037499999991269, 'diarization_start': 8151.499718750001, 'diarization_end': 8151.80346875}, 
{'diarization_id': 456, 'speaker': 'SPEAKER_09', 'overlap_seconds': 7.224, 'whisper_coverage': 0.235, 'diarization_coverage': 0.991, 'diarization_duration': 7.289999999999054, 'diarization_start': 8154.7059687500005, 'diarization_end': 8161.9959687499995}])
```
Same logic applies for more than 2 speaker.

In case of a break even, which should be low probability, a specific flag will be used.

# Speaker election strategy

The objective of the merge stage is not to reconstruct pyannote segmentation.

The objective is: associate the most probable speaker to each semantically coherent Whisper segment

Diarization segments are therefore treated as speaker evidence rather than structural truth.
For each Whisper segment:

1. Collect all overlapping diarization segments
2. Group overlaps by speaker identity
3. Compute the cumulative overlap duration for each speaker
4. Weight long continuous diarization segments more heavily than multiple fragmented micro-segments
5. Elect the speaker with the strongest weighted evidence

The weighting step is important because diarization fragmentation is acoustically unstable.
For example:
```text
Speaker A:
5 segments of 0.3s
total overlap = 1.5s

Speaker B:
1 segment of 1.4s
total overlap = 1.4s
```

A naive cumulative-duration algorithm would elect Speaker A...

However, multiple micro-segments are less reliable evidence than a single continuous diarization region.

The merge therefore favors:
* long continuous speaker evidence
* low fragmentation
* stable overlap continuity

instead of purely maximizing cumulative overlap duration.

This prevents fragmented diarization artifacts from dominating speaker assignment.

# Clean segment extraction for centroid recomputation

After speaker election, Whisper segments associated with:

* a single dominant speaker
* stable overlap structure
* sufficiently long diarization support

can be reused as high-confidence speaker material.

Segments with:
* unique speaker ownership
* low ambiguity
* diarization support longer than ~1.5 seconds

will be used to recompute speaker centroids.

The objective is to:
* reduce diarization alias splits
* stabilize speaker identity across long sessions
* merge fragmented speaker identities such as SPEAKER_40 and SPEAKER_36

using cleaner, semantically validated speech regions.
