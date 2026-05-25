In order to have better reproducibility. The canonical JSON for the *engine* key*transcript* and *diarization* now saves a *option* dictionary, with the different parameters that was called for the current run.

> Forgetting to do this from day one was a massive mistake. Lesson learn reproductibility is more important than anything. What's the point of an experiment if it can not be reproduce? !

The diarization from *pyannote* does not have much parameters, but they are still saved ( except the HF_TOKEN of course)

But whisper is different. In the early test runs, the only tweaks to test on short videos where on **beam size**, **condition_on_previous_text**, **language** and later the **vad_min_silence_duration_ms**
All other value where the default, but this time, in order to save them,  **Temperature**, **vad_speech_pad_ms** were added.

# Whisper Temperature

The default temperature value for Whisper is not a single float, but a fallback schedule:
```text
[0.0, 0.2, 0.4, 0.6, 1.0]
```

Initially, the temperature was manually set to `0.0`, under the assumption that deterministic decoding would force Whisper to rely more strongly on the audio signal itself.

> This assumption was incorrect.

Ran against the audit module:
```text
Total segments        : 7542
Flagged segments      : 5185

----=== FLAG SUMMARY ===0----

HIGH_COMPRESSION_RATIO                   4881  (94.14%)
PARTIAL_VAD_OVERLAP                      816   (15.74%)
IMPOSSIBLE_SPEECH_RATE                   518   (9.99%)
INFORMATION_RATE_TOO_HIGH                369   (7.12%)
DISCONTIGUOUS_VAD_COVERAGE               229   (4.42%)
LONG_DURATION_SHORT_TEXT                 87    (1.68%)
OUTSIDE_VAD                              45    (0.87%)
MULTI_SPEAKER_CANDIDATE                  41    (0.79%)
LONG_WHISPER_SEGMENT_LOW_VAD_COVERAGE    14    (0.27%)
mean = 4.41
median= 1.82
p10  = 0.75
p90  = 4.64
min  = 0.08
max  = 754.27
std  = 33.70
```

for 
```json
      "name": "faster-whisper",
      "model": "large-v3",
      "device": "cuda",
      "compute_type": "float16",
      "options": {
        "language": "fr",
        "vad_filter": true,
        "vad_parameters": {
          "min_silence_duration_ms": 1000,
          "speech_pad_ms": 200
        },
        "beam_size": 5,
        "temperature": 0,
        "condition_on_previous_text": true,
        "word_timestamps": true
      }
```



A heavily simplified view of Whisper decoding is:
P(token=t | audio , token<t)

The next predicted token depends both on:
* the encoded audio representation
* the previously generated tokens

Because Whisper is autoregressive, previously generated text strongly influences future predictions.

If the model enters a bad decoding trajectory with high confidence, the linguistic prior progressively dominate the acoustic evidence.
This creates hallucination loops where the model keeps generating plausible text patterns while partially ignoring the underlying audio signal.

With fully deterministic decoding (temperature=0.0), these loops can become very stable...

The fallback temperature schedule exists to mitigate this behavior.

When Whisper detects suspicious decoding patterns from its confidence proxies :
* very high compression ratio
* extremely repetitive outputs
* low average log-probability

it retries decoding with a slightly higher temperature.

Increasing temperature does not *add creativity* directly. 
Instead, it rescales the token probability distribution before sampling, making lower-probability alternatives more reachable. This can help the decoder escape repetitive or degenerate autoregressive loops.

The new experimental configuration uses a shorter fallback schedule: `[0.0, 0.2, 0.6]`
The goal is to preserve deterministic-first decoding while still allowing limited fallback exploration when the decoder appears trapped in a pathological generation pattern.


# New parameters:

The future whisper runs will be:
```json
    "engine": {
      "name": "faster-whisper",
      "model": "large-v3",
      "device": "cuda",
      "compute_type": "float16",
      "options": {
        "language": "fr",
        "vad_filter": true,
        "vad_parameters": {
          "min_silence_duration_ms": 1000,
          "speech_pad_ms": 400
        },
        "beam_size": 5,
        "temperature": [
          0.0,
          0.2,
          0.4
        ],
        "condition_on_previous_text": true,
        "word_timestamps": true
      }
    },
```

Manual review of known noisy/failure regions:
- 400ms pad clearly cleaner than 200ms.
- Fallback temperature removed repetition/compression-ratio pathologies.
- No ultra-long pathological segments observed.
- Flag distribution is much healthier.

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

mean = 4.08
median= 3.30
p10  = 1.20
p90  = 7.45
min  = 0.32
max  = 30.73
std  = 3.89

------=== FLAG COMBINATIONS ===------

PARTIAL_VAD_OVERLAP                                                              147
DISCONTIGUOUS_VAD_COVERAGE | PARTIAL_VAD_OVERLAP                                 49
MULTI_SPEAKER_CANDIDATE                                                          33
DISCONTIGUOUS_VAD_COVERAGE                                                       29
INFORMATION_RATE_TOO_HIGH                                                        3
LONG_WHISPER_SEGMENT_LOW_VAD_COVERAGE | PARTIAL_VAD_OVERLAP                      3
DISCONTIGUOUS_VAD_COVERAGE | LONG_WHISPER_SEGMENT_LOW_VAD_COVERAGE | PARTIAL_VAD_OVERLAP 
2
DISCONTIGUOUS_VAD_COVERAGE | LONG_WHISPER_SEGMENT_LOW_VAD_COVERAGE | MULTI_SPEAKER_CANDIDATE | PARTIAL_VAD_OVERLAP 1
IMPOSSIBLE_SPEECH_RATE | INFORMATION_RATE_TOO_HIGH | MULTI_SPEAKER_CANDIDATE     1
IMPOSSIBLE_SPEECH_RATE | PARTIAL_VAD_OVERLAP                                     1
IMPOSSIBLE_SPEECH_RATE                                                           1

```