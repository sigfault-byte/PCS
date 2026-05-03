The goal was to try to analyze the last run [[Journal_04_new_run]] and hopefully find high-confidence truth from probabilistic reconstruction.

Only one part, confidence, was analyzed correctly.

# Plan

Verify that Silero VAD and Pyannote are returning similar segments on the whole audio.
Select a mask of regions where they both agree, in order to create three states per transcript segment:
- Silero & Pyannote agree
- diarization agrees
- outside of a speech segment strongly indicating a hallucination for the transcript segments

Compute how much Silero and Pyannote agree.
Filter the different transcriptions from Whisper with `VAD=OFF` `VAD=1000ms` `VAD=2000ms`

Using `avg_log`, `compression` and `nospeech` variables returned by Whisper for each segment, select the best and worst segments for each.
Manually inspect the segment values to look for false positive or negative.

---
# Overlapping Silero and Pyannote

```text
VAD total speech:        14864.20s
Diarization total:       15319.50s
Agreement total:         14778.94s
Agreement / VAD:         99.43%
Agreement / diarization: 96.47%
```
Silero's segments appear to be sub-segments of Pyannote.
Silero may be better at separating speech from random noise, or it may detect silence within a sentence more precisely, while Pyannote may be more lenient around breaks inside speech.
> The non-overlapping part will be used to hopefully see if they fall on different speaker turns

----
# Filtering *bad* segments

```text
VAD_OFF transcript filtered by VAD+diarization agreement
  original rows:           6,697
  filtered rows:           5,893
  kept ratio:              87.99%
  marked overlap rows:     40
  marked overlap duration: 22.67s

VAD_1000 transcript filtered by VAD+diarization agreement
  original rows:           5,745
  filtered rows:           5,127
  kept ratio:              89.24%
  marked overlap rows:     41
  marked overlap duration: 24.02s

VAD_2000 transcript filtered by VAD+diarization agreement
  original rows:           7,295
  filtered rows:           6,444
  kept ratio:              88.33%
  marked overlap rows:     38
  marked overlap duration: 21.79s
```
>This is both confusing and interesting.
>Most of the VAD_OFF errors are in big gaps

VAD_1000 appears to be the most relevant here: fewer rows and the highest precision on speech segments.

---
# Electing the *"best"* VAD settings 

In order for the confidence proxies to be comparable across runs and segments, they were normalized like this:
```python
def zscore(series: pd.Series) -> pd.Series:
	return (series - series.mean()) / (series.std() + 1e-9)
```

Then the segments were sorted like this to create a *quality score*
```python
df["quality_score"] = (
        df["z_avg_logprob"] 
        - df["z_no_speech_prob"] 
        - df["z_compression_penalty"]
    )
```

### Worst distribution top 50:
```text 
VAD_2000    0.6
VAD_OFF     0.4
```
> VAD 1000 is not in the top 50.
### Best distribution top 50:
```text
VAD_1000    0.48
VAD_2000    0.28
VAD_OFF     0.24
```
> VAD 1000 dominates the ranking with almost 50% of the top 50.

### Worst 10%:
```text
VAD_2000    0.352234
VAD_1000    0.327033
VAD_OFF     0.320733
```

### Middle 10%:
```text
VAD_2000    0.344591
VAD_OFF     0.338867
VAD_1000    0.316543
```

### Best 10%:
```text
VAD_2000    0.457355
VAD_1000    0.277619
VAD_OFF     0.265026
```
>This looks counterintuitive, but
>VAD_2000 = high recall of good content
>VAD_1000 = higher peak precision

---
# Explanation of the result

### VAD 2000
It dominates the top 10%, but also the worst 10%.
It is paradoxical, it has a lot of the *best* segments, but also a lot of the *worst*.
Moreover its segment count is higher than the other, making it more distributed everywhere, for the worst and the best.
VAD_2000 increases variance
- better best segments
- worse worst segments
- recall issues (missing spans)
> The params finds a lot of good segments, but when it is failing it fails a lot.
> Also worth noting that it misses some words [[Journal_04_new_run#Precision and Recall]]
#### VAD 1000
It has the best balance across the three, but it is worth mentioning that this value also produced an error that only happened on this setting, where a number, 3000, was replaced by a word.
### VAD OFF
When it is not hallucinating in silence zones, the results are not bad.
But it also has some missing words and sentence for some reason.
> I was hoping it would be higher

---

Summary: 
1. Speech mask (VAD ∩ diarization)
2. Base transcript (VAD_1000)
3. Cross-run audit:
   - VAD_OFF → recall (did we miss something?)
   - VAD_2000 → precision (is this segment very clean?)
1. Confidence score (normalized metrics)
2. Manual inspection (top/bottom)

>Though, is it is neither viable nor smart to make multiple Whisper run each time, ideally, running VAD 1000, then splitting the audio where the proxies indicate an issue, and running that audio section with VAD OFF could be interesting. 

# Idea ( not to do now)

Pass 1:
VAD_1000 full transcript

Pass 2:
detect suspicious segments:
   - low logprob
   - high compression
   - disagreement across runs
   - outside speech mask

Pass 3:
re-run ONLY those segments with:
   - VAD_OFF (recover recall)
   - maybe different decoding params

Pass 4:
replace / flag segments

---
# The overlapping speaker segments

The idea here, is manually checking if the transcript segments ∩ overlapping speaker region are parts where the transcription across the 3 settings drifts. This could indicate that the stochastic nature of the Whisper decoder is also important to think about, as it might just be very confident that the end of a sentence is something, but it did not actually hear it.
This would make sense since it is the setting with the longest text / context.

---

# Overlapping speaker regions and segments

There are `46` raw segments with multi speaker region.

### The first one is `wseg_000419`
```bash
jq '.transcript.raw_segments[] | select(.segment_id == "wseg_000419")' 14-avril-2026_02_transcription_VAD-OFF.json
{
  "segment_id": "wseg_000419",
  "start_token_id": 3108,
  "end_token_id": 3115,
  "time": {
    "start_seconds": 2014.6,
    "end_seconds": 2017.22,
    "duration_seconds": 2.6200000000001182,
    "start_ts": "00:33:34.60",
    "end_ts": "00:33:37.22"
  },
  "raw_text": "Madame la présidente, mesdames et messieurs les députés,",
  "avg_logprob": -0.1223655521956294,
  "no_speech_prob": 0.0006260871887207031,
  "compression_ratio": 1.5981012658227849,
  "flags": 0
}
```

And looking around it:
```bash
jq -r '
  .transcript.raw_segments[]
  | select(.time.end_seconds > 2005.0 and .time.start_seconds < 2015.0)
  | {
      segment_id,
      raw_text,
      start: .time.start_seconds,
      end: .time.end_seconds,
      avg_logprob,
      no_speech_prob,
      compression_ratio
    }
' 14-avril-2026_02_transcription_VAD-OFF.json
{
  "segment_id": "wseg_000414",
  "raw_text": "La parole est à monsieur",
  "start": 2004.86,
  "end": 2006.02,
  "avg_logprob": -0.1223655521956294,
  "no_speech_prob": 0.0006260871887207031,
  "compression_ratio": 1.5981012658227849
}
{
  "segment_id": "wseg_000415",
  "raw_text": "Jean-Pierre Farandou, ministre du Travail",
  "start": 2006.02,
  "end": 2008.56,
  "avg_logprob": -0.1223655521956294,
  "no_speech_prob": 0.0006260871887207031,
  "compression_ratio": 1.5981012658227849
}
{
  "segment_id": "wseg_000416",
  "raw_text": "et de Solidarité.",
  "start": 2008.56,
  "end": 2009.46,
  "avg_logprob": -0.1223655521956294,
  "no_speech_prob": 0.0006260871887207031,
  "compression_ratio": 1.5981012658227849
}
{
  "segment_id": "wseg_000417",
  "raw_text": "Allez un peu de silence s'il vous plaît.",
  "start": 2010.62,
  "end": 2012.52,
  "avg_logprob": -0.1223655521956294,
  "no_speech_prob": 0.0006260871887207031,
  "compression_ratio": 1.5981012658227849
}
{
  "segment_id": "wseg_000418",
  "raw_text": "Chut.",
  "start": 2013.22,
  "end": 2013.7,
  "avg_logprob": -0.1223655521956294,
  "no_speech_prob": 0.0006260871887207031,
  "compression_ratio": 1.5981012658227849                                                                                        }
{
  "segment_id": "wseg_000419",
  "raw_text": "Madame la présidente, mesdames et messieurs les députés,",
  "start": 2014.6,
  "end": 2017.22,
  "avg_logprob": -0.1223655521956294,
  "no_speech_prob": 0.0006260871887207031,
  "compression_ratio": 1.5981012658227849
}
```

The *chut* raw text here made me curious, this is not really a word from dictionary.
So i wonder if VAD-1000 got it:

```bash
{
  "segment_id": "wseg_000295",                                                     
  "raw_text": "La parole est à monsieur Jean-Pierre Farandou, ministre du Travail et des Solidarités.",
  "start": 2004.87,
  "end": 2009.43,
  "avg_logprob": -0.10099511173184357,
  "no_speech_prob": 0.0029811859130859375,
  "compression_ratio": 1.6765578635014837
}
{
  "segment_id": "wseg_000296",
  "raw_text": "Allez, un peu de silence, s'il vous plaît.",
  "start": 2010.77,
  "end": 2012.51,
  "avg_logprob": -0.10099511173184357,
  "no_speech_prob": 0.0029811859130859375,
  "compression_ratio": 1.6765578635014837
}
{
  "segment_id": "wseg_000297",
  "raw_text": "Madame la Présidente, mesdames et messieurs les députés,",
  "start": 2014.54,
  "end": 2017.22,
  "avg_logprob": -0.10099511173184357,
  "no_speech_prob": 0.0029811859130859375,
  "compression_ratio": 1.6765578635014837
}
```

The *chut* is absent from the VAD-1000
And also present in the VAD-2000
```bash
{
  "segment_id": "wseg_000293",
  "raw_text": "Chut !",
  "start": 2013.66,
  "end": 2014.58,
  "avg_logprob": -0.12220292887937875,
  "no_speech_prob": 0.00018668174743652344,
  "compression_ratio": 1.604026845637584
}
```

### The next segments are `wseg_000555` `wseg_000556`  `wseg_000557` from VAD-OFF
timeframe is:
`"start_ts": "00:38:08.22"` - `"2286.2"` seconds for wseg_000555
`"end_ts": "00:38:11.24"` - `"2291.24"` seconds for wseg_000557

VAD-OFF
```text
[00:38:02.34 -> 00:38:04.16] Soit Séville ne valore rien,
[00:38:04.34 -> 00:38:05.84] et donc vous êtes d'accord.
[00:38:06.20 -> 00:38:08.22] Soit vous dites enfin stop
[00:38:08.22 -> 00:38:10.08] et vous vous unissez à l'Espagne
[00:38:10.08 -> 00:38:11.24] pour faire suspendre cet accord.

[00:38:14.22 -> 00:38:14.30] C'est ce qu'a fait
[00:38:14.30 -> 00:38:16.42] Jean-Noël Barraud, ministre de l'Europe
[00:38:16.42 -> 00:38:17.66] et des Affaires étrangères.
[00:38:22.88 -> 00:38:24.48] Merci Madame la Présidente,
[00:38:24.54 -> 00:38:26.38] Mesdames et Messieurs les députés,
[00:38:26.38 -> 00:38:27.74] Monsieur le député Jean-Paul Lecoq.
```
`"avg_logprob"`: -0.14515398356361667

VAD-1000
```text
[00:38:02.39 -> 00:38:05.85] Soit Séville ne valore rien, et donc vous êtes d'accord.
[00:38:06.51 -> 00:38:11.13] Soit vous dites enfin stop et vous vous unissez à l'Espagne pour faire suspendre cet accord.
[00:38:11.13 -> 00:38:12.97] Merci beaucoup monsieur le député.

[00:38:12.97 -> 00:38:17.67] La parole est à monsieur Jean-Noël Barraud, ministre de l'Europe et des Affaires étrangères.
[00:38:22.97 -> 00:38:25.61] Merci madame la présidente, mesdames et messieurs.
[00:38:25.61 -> 00:38:27.75] Monsieur les députés, monsieur le député Jean-Paul Lecoq.
```
`"avg_logprob"`: for the exact time frame is -0.11775362329638522

VAD-2000 
```text
[00:38:02.52 -> 00:38:04.16] Soit Séville ne valore rien
[00:38:04.16 -> 00:38:05.86] et donc vous êtes d'accord.
[00:38:06.78 -> 00:38:08.22] Soit vous dites enfin stop
[00:38:08.22 -> 00:38:10.08] et vous vous unissez à l'Espagne
[00:38:10.98 -> 00:38:11.16] pour répondre.
[00:38:11.16 -> 00:38:13.02] Merci beaucoup Monsieur le député.

[00:38:13.02 -> 00:38:15.00] La parole est à Monsieur Jean-Noël Barraud,
[00:38:15.14 -> 00:38:17.66] ministre de l'Europe et des Affaires étrangères.
[00:38:23.05 -> 00:38:24.49] Merci Madame la Présidente,
[00:38:24.55 -> 00:38:26.37] Mesdames et Messieurs les députés,
[00:38:26.39 -> 00:38:27.75] Monsieur le député Jean-Paul Lecoq.
```
`"avg_logprob"`:  for the exact timeframe segments of the overlap is 
 -0.06456801393891082 to -0.10768581126388665

There are issues with missing transcript, and diverging values, ground truth, after manually listening is:
```text
Soit ces vies ne valent rien et donc vous etes d'accord.
Soit vous dites enfin stop et vous vous unissez a l'espagne <overlapping different speaker "Merci le deputé">
Pour faire suspendre cet Ac... <Speaker microphone is cut, and "Merci beaucoup M. le depute" from other speaker>
```

So, objectively, none of the three transcripts is fully correct.
VAD OFF and VAD 1000 have "*hallucinated*" the correct word the speaker was going to say.

### Next segments are  `wseg_000628` `wseg_000629` 
timeframe is:
`start_ts": "00:40:31.98"` / `"start_seconds": 2431.98` wseg_000628
`"end_ts": "00:40:34.62"` / `"end_seconds": 2434.62` wseg_000629

VAD-OFF
```text
[00:40:30.16 -> 00:40:31.98] le droit international et abandonner
[00:40:31.98 -> 00:40:32.62] ces guerres sans fin.
[00:40:32.62 -> 00:40:34.62] Merci beaucoup Monsieur le Ministre.
```

VAD-1000
```text
[00:40:29.69 -> 00:40:32.59] respecter le droit international et abandonner ces guerres sans fin.
[00:40:33.03 -> 00:40:35.15] Merci beaucoup Monsieur le Ministre.
```

VAD-2000
```
[00:40:29.69 -> 00:40:30.85] respecter le droit international
[00:40:33.27 -> 00:40:33.59] et le droit de l'homme.
[00:40:33.59 -> 00:40:35.21] Merci beaucoup Monsieur le Ministre.
```

Manually listening, the speaker microphone was not cut, but the last sentence is overlapping the one that diverge.

>In overlapping speaker regions, the acoustic signal becomes ambiguous due to the presence of multiple voices.
>Whisper must therefore compress multiple audio signals into a single textual sequence.

>When the signal is degraded, the model relies more heavily on its internal language model to produce a plausible continuation.
>This effect is amplified in longer segments (e.g., VAD_2000), where additional context encourages the model to complete sentences confidently, even when the acoustic evidence is incomplete.

>In contrast, shorter segments (e.g., VAD_1000 or VAD_OFF) reduce the influence of long-range context, forcing the model to rely more on local acoustic information.

### The next two `wseg_000760` and `wseg_000761`
There is not much divergence between the transcripts.
The sentences `les Français` `pour la sécurité` are in two different segments for VAD-OFF and VAD-2000, but a single segment for VAD-1000.
Hence VAD-1000 added a `et` between the two. 
Listening to the recording, the speaker is almost screaming, speaks fast, and *eats* the words. 
And the speech before and after theirs is very *close* to it, certainly overlapping a tiny bit.

### The next is `wseg_000803`
All three transcripts agree on the exact transcript, though VAD-1000 has more complete sentences and less segments. The issue, after listening, is people shouting, clapping, making noise and the president asking them to be quiet.

### The next are `wseg_000968` `wseg_000969` `wseg_000970` `wseg_000971`
Nothing relevant in this one, the transcripts are identical, the overlapping speaker did not alter the transcription

### The next are `wseg_001246` and `wseg_001247`
timeframe:
`"start_ts": "01:01:06.16"` |  `"start_seconds": 3666.16"`
`"end_ts": "01:01:11.02"` | `"end_seconds": 3671.02`"

VAD-OFF
```text
[01:01:06.16 -> 01:01:08.88] Mme la présidente, Mme et M. les députés
[01:01:08.88 -> 01:01:11.02] Mme la députée
```

VAD-1000
```text
[01:01:05.90 -> 01:01:08.90] Merci Mme la présidente, Mme et M. les députés.
[01:01:10.10 -> 01:01:19.62] La France, depuis le début, c'est-à-dire depuis le 7 octobre 2023, a tenu une ligne claire et sans aucune ambiguïté.
```

VAD-2000
```text
[01:01:05.90 -> 01:01:11.04] Merci Mme la présidente, Mme et M. les députés, Mme la députée.
```

Reflection on the current segment:

After listening to the segment, the sequence is characteristic of the French Assembly *“cacophony”*.
The president is calling other deputies to be quiet, ( `01:00:54.86 -> 01:00:55.46] Chut` caught by VAD-OFF again ), people in the back are shouting. 

However, the overlap is not sustained or clear enough for Pyannote to label the entire region as multi-speaker.

> I was wondering if some classifier exists to detect this kind of noisy pattern. 
> Instead, a lighter idea would be to try if a FFT ( or some cheap proxies like short-time energy / RMS or spectral flux ) of the audio, where some *dumb* thresholds would correspond to region with people screaming, could help filter noisy regions.

In this segment, VAD_OFF provides a more accurate speaker boundary because it allows the decoder to reset at acoustic transitions.

In contrast, VAD_1000 and VAD_2000 merge adjacent speech into a single segment, allowing the language model to propagate context across speakers. This results in boundary contamination, where tokens from the previous speaker are incorrectly attributed to the next.

This type of error is not immediately problematic at the transcription level, since the text remains coherent and plausible.

However, it becomes significant when combined with speaker attribution, as boundary contamination leads to incorrect assignment of words to speakers. 
The earliest run encountered various similar cases.

Moreover, the segments must be flagged precisely so those with no flags can be used to compute centroid embedding of specific speaker whose names are extracted with NER analysis later.

### Next is `wseg_001501`
Quick overlapping, VAD_2000 hallucinates a *"bonjour"* instead of a *"merci"*. 

### Next are from `wseg_002001` to `wseg_002004`
timeframe:
`"start_ts": "01:26:11.60"` | `"start_seconds": 5171.6`
`"end_ts": "01:26:20.64"` | `"start_seconds": 5179.22`

In this one VAD-1000 is making mistakes.
The president of the assembly says an overlapping *"merci monsieur le deputé"* before he finishes.
VAD_1000 caught it and added it in between two segments as *"oui monsieur le deputé"*.

VAD-1000
```text
[01:26:11.81 -> 01:26:13.55] Parce que derrière ces chiffres, il y a des jeunes.
[01:26:13.55 -> 01:26:14.67] Oui, monsieur le député.
[01:26:14.69 -> 01:26:19.23] Des parcours de vie qui peuvent basculer ou être sauvés. Et aujourd'hui, ils attendent des réponses.
```

This is somehow similar to the previous error. It is not inherently problematic, but it will be in the rest of the pipeline, when needed to merge segments are speakers.

### Next is `wseg_002070`
VAD-OFF is - again - the closest to the truth and in the segmentation of speakers.
There are numerous repetitions of the word *"merci"* overlapping with the speech, and sudden change of speaker.

Both VAD-1000 and VAD-2000 missed some, while VAD-OFF caught the correct transcript.

Next is `wseg_002905`
It is a false positive. There is no overlapping speech, but the speaker is clearly reading a document for the first time, eats many words, softens his voice while trying to breathe and continue reading, then suddenly says *"BREF"*. 

A few seconds before this bref, VAD-OFF missed a part of the speech, where the speaker talks with a very soft voice compared to the rest of his speech.
~ 2 second of speech is missing.

VAD 1000 and VAD 2000 did not miss it.

### Next is `wseg_002992`
All transcript agree.

### Next is `wseg_003370`
VAD-OFF and VAD-1000 agree, VAD-2000 missed the overlapping part, which is the beginning of the sentence of the next speaker.

### Next is `wseg_003992`
The environment is noisy, we can hear people clapping their hands, shouting...
VAD-1000 makes a mistake in the transcript:
```text
[02:58:57.20 -> 02:58:58.12] pour le groupe blanc
[02:58:58.14 -> 02:58:59.08] semble pour la République
```
The correct transcript, caught by the other is `"ensemble"`.

### Next is `wseg_004607`
...
The president of the assembly is calling out ki- deputies that are not behaving. Stating *"On entend vraiment un brouhaha constant"*
Then the next speaker very loudly says *"madame la presidente"*.

All the transcript agree, they are all correct.

### Next is `wseg_004718`
Same speaker as the previous, the assembly is shouting. The president has to call them out again during the speech.
All transcripts agree

### Next is `wseg_004794`
(This is a bit tedious) A new speaker is, instead of debating, attacking the speaker. Resulting in people shouting in the assembly. 
All the transcripts agree.

### Next is `wseg_004896`
Very noisy environment. Multiple people shouting and booing.
VAD-2000 is the most accurate.
VAD-1000 hallucinates some words, and VAD-2000 changed the meaning.

### Next are `wseg_004902` `wseg_004903`
Same as before, except VAD-1000 and VAD-2000 both missed some words.

### Next is `wseg_005007`
All transcript agree.

This segment precedes a 5 min break. During this break, VAD-OFF is in a hallucination loop of the last word pronounced *"merci"*
```text
[04:13:04.92 -> 04:13:05.02] Merci.
[04:13:51.88 -> 04:13:53.28] Merci.
[04:14:01.20 -> 04:14:04.58] Madame la Présidente,
```
The hallucination loop is eliminated by the filtering of silence gap.
But the issue is that it is missing some text when the session begins:
VAD-1000
```text
[04:06:41.85 -> 04:13:14.16] chers collègues
[04:13:14.16 -> 04:13:15.26] la séance est reprise
```
VAD-2000
```text
[04:06:40.94 -> 04:06:41.84] de 5 minutes.
[04:13:13.10 -> 04:13:14.18] Chers collègues,
```
So this is really disorienting.

VAD-OFF misses ~ 1 min of speech.
VAD-1000 considers the whole 5-minute break as a continuous segment with only 2 words.
VAD-2000 correctly shut itself during the break, and starts correctly.

> This was already noted during the very early runs, this was also the reason why I started to get intrigued in the VAD parameter of Whisper.
> But the more I know the less it makes sense...

### Next is `wseg_005218`
No transcripts agree. All got it wrong.
It is very noisy, the person is probably walking away from the microphone while talking.

### Next are `wseg_006076` to `wseg_006078`
Noisy, multiple speaker, plus clapping. Mic is cut off from the speaker.
VAD-OFF has *"latched"* on the current speaker, where VAD-1000 latched on the exact overlapping speaker.

VAD-OFF:
```text
<SPEAKER A>[04:58:42.74 -> 04:58:44.50] intégralement par le Conseil constitutionnel
[04:58:44.50 -> 04:58:45.36] à la fin. Je veux dire...
<SPEAKER B>[04:58:45.36 -> 04:58:47.42] Ce n'est pas un rappel au règlement.
```
VAD-1000:
```text
[04:58:41.02 -> 04:58:44.80] et qui a été censurée intégralement par le Conseil constitutionnel à la fin
[04:58:44.80 -> 04:58:47.44] Vous êtes sur le fond, ce n'est pas un rappel au règlement
```
VAD-2000:
```text
[04:58:42.66 -> 04:58:44.48] intégralement par le Conseil constitutionnel
[04:58:44.48 -> 04:58:44.80] à la fin
[04:58:44.80 -> 04:58:47.44] ce n'est pas un rappel au règlement
```

### Next are `wseg_006142` to `wseg_006144`
The speaker is mumbling and is not able to correctly express itself, the president cuts him, and he answers back to her with a *je reviendrai*. 
All three transcripts correctly split and recover most of the signal. 

### Last is `wseg_006413`
All three agree.


---

# Observations
### VAD-2000
It is not precise enough in boundary handling and recall,
despite performing well on long, continuous speech segments. It also often naturally correctly splits segments across different speaker.
But the imprecision and multiple context-driven reconstruction errors prevent its usage for such a specific environment.

Though, it is the most precise regarding segment timing in the current run.
This is either the stochastic nature of the model, that sometimes collapse silence and speech and sometimes does not.

### VAD-1000
it is the most balanced, as expected, but after looking more precisely at multiple (noisy) segments it suffers from the same issue - though less often and with a lot fewer hallucinations- as VAD-2000. 

### VAD-OFF
It is, in the end, the best one for noisy parts.
It misses some words, in the middle of a sentence, when the speaker is speaking softly (`wseg_002070` ).
The model likely considers the audio signal too weak, and because it has less context, discards the frame. 

### Additional obs

The irregularities in the punctuation are impressive.
At first, I thought this was only VAD-OFF that would most often not add punctuation, no periods, no commas, no capitalization of a new sentence.

But, in fact, each transcript often exhibits different punctuation patterns, even when they agree to the exact *word token* across the whole text. 
Sometimes VAD-OFF does better than VAD-1000 or VAD-2000, or the opposite is true.

It is not *really* important for the current goal, but it is interesting to notice that the model is not consistent.
