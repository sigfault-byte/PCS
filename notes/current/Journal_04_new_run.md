
# Transcription

```python
# Whisper confidence proxies
avg_logprob: float | None = None
no_speech_prob: float | None = None
compression_ratio: float | None = None
```
was added to the transcript segment class so these values can be captured during transcription.

1. **avg_logprob**:
Each generated token has a conditional probability given the audio and previous tokens.
`avg_logprob` is the average log-probability of those generated tokens over the segment.
Values are negative most of the time (`log(1) = 0`  & `log(<1) < 0`)
> High avg_logprob means Whisper was confident in its chosen words, not that those words are necessarily true.
> It could be 0 because Whisper hallucinated incorrectly but was confident in the hallucination, which is the whole point ... !

2. **no_speech_prob**
Probability estimated by Whisper that the current region contains no speech.
It behaves like an internal speech-presence detector and is closely related to VAD logic.
> The higher the value, the higher the chance the segment is a *bad* hallucination. 
> This is a complicated metric that would require diving into the Whisper code to understand properly. Like the previous one, it cannot be used as a truth metric. 

3. **compression_ratio**
This one is both interesting and cursed.
Alone it is not sufficient here and requires broader comparison across the dataset.
> This is basically, if I "zip" the current segment how much does it shrink
> In this specific example, `avg_logprob` and `no_speech_prob` are more informative than compression_ratio.

---
## Hypothesis: why running Whisper without using its VAD

Because Silero VAD is now in charge of VAD, Whisper can focus on its main purpose: transcription.

Whisper is fundamentally a decoder that tries to explain audio with the most plausible token sequence.
Allowing it to decode more freely can improve recall, but may increases hallucination ?
External Silero VAD can then be used as an independent audit layer to recover precision.

> Three runs : VAD = 2000 ms, VAD = 1000 ms, and VAD OFF

---
# First run with VAD OFF

Example of a segment on the (un)famous *bad* hallucinations from  [[Journal_01_Merging#Remark **`wseg_006189`**]]  `"Sous-titrage ST' 501"`

The audio file begins with a very long ~ 15 min of silence.
During this time Whisper VAD-OFF is in a hallucination loop typical of its decoder nature.
```text
[00:00:00.00 -> 00:00:29.98] Sous-titrage ST' 501
[00:00:30.00 -> 00:00:59.98] Sous-titrage ST' 501
[00:01:00.00 -> 00:01:29.98] Sous-titrage ST' 501
[00:01:30.00 -> 00:01:59.98] Sous-titrage ST' 501
```
> notice exact 30 sec window typical of Whisper

```text
[00:14:30.00 -> 00:14:59.98] Sous-titrage ST' 501
[00:15:19.52 -> 00:15:29.98] Sous-titrage ST' 501
```
> the second segment here triggered at 15:19. There is 19 second of gap between the two segments. 

```bash
jq ".transcript.raw_segments | .[30]" file.json
{
  "segment_id": "wseg_000031",
  "start_token_id": 120,
  "end_token_id": 123,
  "time": {
    "start_seconds": 919.5200000000001,
    "end_seconds": 929.98,
    "duration_seconds": 10.459999999999923,
    "start_ts": "00:15:19.52",
    "end_ts": "00:15:29.98"
  },
  "raw_text": "Sous-titrage ST' 501",
  "avg_logprob": -0.1774088591337204,
  "no_speech_prob": 0.0059051513671875,
  "compression_ratio": 0.7142857142857143,
  "flags": 0
}
```
Whisper "*knows*" there is speech in this segment. 
The `compression_ratio` is high
The `avg_logprob` is shy from 0.2. 

> The text is pure *bad* hallucination.

VAD = 1000ms in the same timeframe segment:
> Both VAD 1000ms and VAD 2000ms work fine here
```bash
{
  "segment_id": "wseg_000001",
  "start_token_id": 0,
  "end_token_id": 6,
  "time": {
    "start_seconds": 913.26,
    "end_seconds": 915.5,
    "duration_seconds": 2.240000000000009,
    "start_ts": "00:15:13.26",
    "end_ts": "00:15:15.50"
  },
  "raw_text": "Bonjour à tous, la séance est ouverte.",
  "avg_logprob": -0.10845170481638475,
  "no_speech_prob": 0.01445770263671875,
  "compression_ratio": 1.4705882352941178,
  "flags": 0
}
```
`avg_logprob` is closer to 0, so *overall* confidence is better regarding the generated text
`no_speech_prob` is higher, VAD is less certain there is speech at this moment, though it is only 1%
`compression_ratio` is bigger, very good sign, the text should be genuine.

## When does VAD OFF run catch up:

The VAD OFF catches up here:
`[00:17:03.30 -> 00:17:07.88] Voilà, tous les cinq ont été retenus en otages en Iran.`
So about 1 min 50 sec *late*, compared to Whisper with VAD detection.

```text
[00:17:03.30 -> 00:17:07.88] Voilà, tous les cinq ont été retenus en otages en Iran.
[00:17:08.12 -> 00:17:10.08] Ils se sont rencontrés aujourd'hui.
[00:17:10.60 -> 00:17:14.44] Et c'est beaucoup d'émotion pour la représentation nationale
[00:17:14.44 -> 00:17:18.68] de vous accueillir ici et de vous voir libres.
[00:17:18.72 -> 00:17:21.68] Et comme le dit Cécile, vive la vie !
[00:17:22.86 -> 00:17:23.38] Applaudissements
```
Notice Whisper with VAD_OFF detects and interprets noises. Which I am pretty sure is an *emergent* behavior, because that was not its purpose at all.

> Probably coming form its training data was from CC / SHD subtitles
--- 

# Precision and Recall

1. Precision
*Out of all predicted positives, how many were actually positive?*
Formula is : `(True positive) / (True positive + False positive)`

2. Recall
*How well does the model find all positive cases?*
Formula is: `(True positive) / (True positive + False negative)`

Looking at the previous data, VAD OFF looks like a high-recall but low-precision signal. 
It captures more real speech in some regions, but also introduces catastrophic hallucinations in silence, such as the repeated “Sous-titrage ST' 501” loop.

VAD OFF :
```text
[00:49:30.06 -> 00:49:31.94] mais quoi qu'il arrive nous devons avoir en tête
[00:49:31.94 -> 00:49:33.82] que collectivement et en particulier à Paris
[00:49:39.66 -> 00:49:41.50] la parole est à présent à monsieur Jean Claudereau
[00:49:41.50 -> 00:49:43.04] pour le groupe écologiste et social
[00:49:44.48 -> 00:49:44.92] merci
```
> gap of 6 seconds with missing text

VAD 2000 ms :
```text
[00:49:30.21 -> 00:49:31.95] mais quoi qu'il arrive nous ne devons avoir en tête
[00:49:31.95 -> 00:49:33.81] que collectivement et en particulier à Paris
[00:49:33.81 -> 00:49:35.95] nous sommes face à une chute démographique majeure
[00:49:35.95 -> 00:49:36.39] je vous remercie
[00:49:36.39 -> 00:49:38.49] Merci beaucoup Monsieur le Ministre
``` 
> looks fine

VAD 1000 ms:
```text
[00:49:30.11 -> 00:49:32.81] mais quoi qu'il arrive nous ne devons avoir en tête que collectivement,
[00:49:32.83 -> 00:49:36.39] et en particulier à Paris, nous sommes face à une chute démographique majeure. Je vous remercie.
[00:49:36.79 -> 00:49:38.47] Merci beaucoup Monsieur le Ministre.
```
> looks better

The VAD OFF segment also fails here to retrieve part of the text. The end of the sentence is important for understanding the speech.

> This is a local recall failure for VAD OFF. 
> It shows that VAD OFF is not universally better, even if it sometimes preserves more raw speech than stricter VAD settings.

VAD 2000 ms has some recall errors:
```
[00:49:22.11 -> 00:49:23.35] il y a toujours la possibilité d'ajuster
[00:49:23.59 -> 00:49:25.25] et au mois de juin et au mois d'août
[00:49:25.25 -> 00:49:26.97] pour ouvrir les classes là où c'est nécessaire
[00:49:30.21 -> 00:49:31.95] mais quoi qu'il arrive nous ne devons avoir en tête
```
> Gap of missing information of 4 seconds

Where VAD 1000 ms doesn't:
```text
[00:49:21.81 -> 00:49:26.97] il y a toujours la possibilité d'ajuster et au mois de juin et au mois d'août pour ouvrir les classes là où c'est nécessaire.
[00:49:27.11 -> 00:49:30.05] Donc nous verrons en cas d'espèce s'il y a lieu de le faire, si il y a lieu de le faire nous le ferons,
```

VAD OFF for reference: 
```text
[00:49:21.80 -> 00:49:23.36] il y a toujours la possibilité d'ajuster
[00:49:23.36 -> 00:49:25.26] et au mois de juin et au mois d'août
[00:49:25.26 -> 00:49:26.98] pour ouvrir les classes là où c'est nécessaire
[00:49:26.98 -> 00:49:28.92] nous verrons en cas d'espèce s'il y a lieu de faire
[00:49:28.92 -> 00:49:30.06] s'il y a lieu de faire nous le ferons
```
> Notice VAD OFF is missing a word "*donc*" compared to VAD 1000ms

Hence, it would appear that VAD 1000 does not have better precision than VAD OFF despite a better recall.
# Mandatory counter example

Both VAD OFF and VAD 2000 ms:
`[00:50:21.68 -> 00:50:23.64] 3000 personnes étaient réunies`

And VAD 1000 ms:
`[00:50:22.51 -> 00:50:26.51] plusieurs personnes étaient réunies pour une course organisée par FRV 100 jeunes,`

Hence, it would appear that VAD 1000 does not have better precision than VAD OFF despite a better recall.

---

# How to verify this further and more accurately

The current test will be:
Filter out all segments falling inside a "*silence*" gap detected by Silero VAD.
Filter out all segments falling into a "multi-speaker" overlap.

If the passages where VAD OFF or VAD 2000 miss text systematically fall inside multi-speaker overlap regions, this may suggest that the recall failures are caused less by the VAD setting itself and more by decoder instability during overlapping or low-energy speech.

If, however, the missed regions are clean single-speaker regions according to Pyannote and Silero, then the failures are more likely caused by the Whisper VAD/decoder behavior directly.

For now I'd say: 
VAD OFF: high recall, low precision because of hallucinated silence regions.
VAD 2000: higher precision, lower recall because it suppresses real speech.
VAD 1000: best compromise so far, but far from perfect.
> Way more tests are needed

---

Ideally, only segments that are supported by multiple independent signals should be treated as “*high-confidence*” segments.

For example:
- no speaker overlap detected by Pyannote
- strong agreement with Silero VAD (speech clearly exists)
- acceptable Whisper confidence proxies (`avg_logprob`, `no_speech_prob`)
- no obvious repetition / hallucination pattern
- no suspicious boundary behavior

These segments could be considered close to “**safe truth**” and used as the most reliable foundation for the final transcript.

Lower-confidence segments should not necessarily be discarded, but flagged for review, rescue passes, or more merge heuristics.

The goal is not to force certainty everywhere - which I doubt I could achieve, but to clearly separate:
high-confidence truth
from
probabilistic reconstruction.