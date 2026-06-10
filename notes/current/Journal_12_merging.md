A new class was added to store an alignment object representing the transcript <-> diarization match.

A new run was then generated using exactly the same parameters as the previous run. 

However, the transcription output is not byte-identical: Whisper produced slightly different segmentation.

[[Journal_10_new_parameters_and_failure|Previous run]]  Whisper audit:
```text
Total segments        : 3300
Flagged segments      : 270

----=== FLAG SUMMARY ===0----

PARTIAL_VAD_OVERLAP                      203   (75.19%)
DISCONTIGUOUS_VAD_COVERAGE               81    (30.00%)
MULTI_SPEAKER_CANDIDATE                  35    (12.96%)
LONG_WHISPER_SEGMENT_LOW_VAD_COVERAGE    6     (2.22%)
INFORMATION_RATE_TOO_HIGH                4     (1.48%)
IMPOSSIBLE_SPEECH_RATE                   3     (1.11%)
```

The new run :
```text
Total segments        : 3287
Flagged segments      : 273

----=== FLAG SUMMARY ===0----

PARTIAL_VAD_OVERLAP                      206   (75.46%)
DISCONTIGUOUS_VAD_COVERAGE               81    (29.67%)
MULTI_SPEAKER_CANDIDATE                  35    (12.82%)
LONG_WHISPER_SEGMENT_LOW_VAD_COVERAGE    6     (2.20%)
INFORMATION_RATE_TOO_HIGH                4     (1.47%)
IMPOSSIBLE_SPEECH_RATE                   3     (1.10%)
```

The audit system appears to be behaving correctly. 
The three additional PARTIAL_VAD_OVERLAP flags are consistent with Whisper merging some segments differently. When a longer Whisper segment spans regions that are only partially covered by VAD, the partial coverage flag is expected, especially given the timestamp drift between Whisper, pyannote, and Silero VAD.

The distribution of unique flag combinations is identical between both runs.

The speaker-overlap results are also identical:
```text
79 transcript segments overlap multiple speakers.
35 of those are not flagged.
```

> The audit system is statistically stable across runs despite segment boundary drift.

---

# Merging
