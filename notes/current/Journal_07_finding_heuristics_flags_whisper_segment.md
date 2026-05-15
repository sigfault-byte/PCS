 During experimentation, auditing very long regions showed that aggregate statistics can bury short but important acoustic events. Mean and median alone are often misleading on long windows; min/max and tail percentiles are needed to detect local issues.

The useful unit for audio-quality auditing is probably the Whisper raw segment window, enriched with:
- Whisper quality proxies
- VAD coverage
- diarization overlap
- librosa-derived local statistics

---
# Example

Lets consider the following segment from silero VAD :

```json
{
  "segment_id": "vad_002390",
  "time": {
    "start_seconds": 14722.5,
    "end_seconds": 14723.5,
    "duration_seconds": 1.0,
    "start_ts": "04:05:22.50",
    "end_ts": "04:05:23.50"
}
```

The whisper VAD-1000 segment that matches to the speech segment is:

```json
  {
    "segment_id": "wseg_004350",
    "start_token_id": 35345,
    "end_token_id": 35347,
    "time": {
      "start_seconds": 14287.46,
      "end_seconds": 14723.16,
      "duration_seconds": 435.7000000000007,
      "start_ts": "03:58:07.46",
      "end_ts": "04:05:23.16"
    },
    "raw_text": "mes chers collègues",
    "avg_logprob": -0.06647135528425376,
    "no_speech_prob": 0.01204681396484375,
    "compression_ratio": 1.8157894736842106,
    "flags": 0
  }
```

This behavior was already noticed in [[Journal_05_whisper_vad_value]]

Whisper is confident about the decoded text:
- `avg_logprob = -0.066` is good
- `no_speech_prob = 0.012` correctly indicates speech is present
- `compression_ratio = 1.81` is not obviously pathological

The problem is not the text itself, but the timestamp span. Whisper attached a short phrase to a 435.7s interval that is almost entirely silent.

Librosa for the whisper segments returns:

```json
{
  "time": {
    "start_seconds": 14287.46,
    "end_seconds": 14723.16,
    "duration_seconds": 435.7000000000007
  },
  "librosa_stats": {
    "rms": {
      "mean": 0.0001341848265779207,
      "median": 0.0,
      "std": 0.0018231047459456413,
      "min": 0.0,
      "max": 0.0480107702,
      "p10": 0.0,
      "p90": 0.0
    },
    "db": {
      "mean": -79.4468693867828,
      "median": -80.0,
      "std": 5.752580437865222,
      "min": -80.0,
      "max": -7.4478607178,
      "p10": -80.0,
      "p90": -80.0
    },
    "zcr": {
      "mean": 0.0011968325933669958,
      "median": 0.0,
      "std": 0.01343857888665155,
      "min": 0.0,
      "max": 0.3039550781,
      "p10": 0.0,
      "p90": 0.0
    },
    "spectral_centroid": {
      "mean": 17.174617798922817,
      "median": 0.0,
      "std": 186.51856259564852,
      "min": 0.0,
      "max": 3907.6513819248,
      "p10": 0.0,
      "p90": 0.0
    },
    "spectral_bandwidth": {
      "mean": 15.954104848467132,
      "median": 0.0,
      "std": 163.78865274440545,
      "min": 0.0,
      "max": 2401.9526884395,
      "p10": 0.0,
      "p90": 0.0
    },
    "spectral_flatness": {
      "mean": 0.9908992452661463,
      "median": 1.000000596,
      "std": 0.09346159808169903,
      "min": 0.005799992,
      "max": 1.000000596,
      "p10": 1.000000596,
      "p90": 1.000000596
    },
  }
}
```

So this 435.7s segment is: 
p90 RMS = 0
p90 dB = -80
p90 ZCR = 0
p90 centroid = 0
p90 bandwidth = 0

At least 90% of the sampled librosa frames in this interval are silent or zero-energy.
But the min/max value give a different story:
max dB = -7.45
max ZCR = 0.30
max centroid = 3907 Hz
flatness min = 0.0058

There is a short burst of structure acoustic event in the segment.

This segment should get a flag.

Some heuristics could be:
```python 
duration = end - start
vad_overlap_seconds = overlap(whisper_segment, vad_segments)
vad_coverage = vad_overlap_seconds / duration

if duration > 30 and vad_coverage < 0.05:
    flags |= LONG_WHISPER_SEGMENT_LOW_VAD_COVERAGE

if db_p90 <= -80 and db_max > -20:
    flags |= MOSTLY_SILENCE_WITH_SHORT_EVENT

if duration > 30 and len(text.split()) <= 5:
    flags |= LONG_DURATION_SHORT_TEXT
```

The last flag also represent the opposite of the " short segment with lots of words". That will be a different flag.

Silence gap are already found by masking VAD segment over the entire pipeline.
"applauses" segment are not marked by silero or diarization as speech segments. 
Unless they are overlapping speaker.

Current flags are:
```python
class SegmentFlag(IntFlag):
    NONE = 0

    # General quality
    SHORT_SEGMENT = auto()
    NEEDS_REVIEW = auto()
    GIBBERISH = auto()
    NON_FRENCH = auto()
    MOSTLY_SILENCE_WITH_SHORT_EVENT = auto()

    # VAD alignment
    OUTSIDE_VAD = auto()
    PARTIAL_VAD_OVERLAP = auto()
    INSIDE_VAD_GAP = auto()
    LONG_WHISPER_SEGMENT_LOW_VAD_COVERAGE = auto()

    # Whisper quality
    LOW_WHISPER_CONFIDENCE = auto()
    HIGH_NO_SPEECH_PROB = auto()
    HIGH_COMPRESSION_RATIO = auto()
    
    LONG_DURATION_SHORT_TEXT = auto()

    # Diarization quality
    DIARIZATION_OVERLAP = auto()
    MULTI_SPEAKER_CANDIDATE = auto()
    SPEAKER_CHANGE_NEARBY = auto()	

    # Merge integrity
    ORPHAN_TRANSCRIPT = auto()
    ORPHAN_DIARIZATION = auto()
```

For now, every flag is purely as triage. Nothing particular will be done with them.
The goal is to be able to have merged segments / diarization of "cleaned" segment.

A diarization segment is composed of one or multiple transcript segment. 

If all transcript segments inside a diarization region have `flags == NONE`, the region can be treated as a high-confidence candidate for downstream analysis.

This will likely improve precision at the cost of some recall. The goal is not to discard flagged material, but to create a reliable high-confidence subset while keeping all raw data available for review.
