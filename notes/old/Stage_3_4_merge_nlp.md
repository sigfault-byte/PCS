
# Overview

- Finding overlaps timestamps between transcription and diarization
- Electing the most plausible speaker
- Merging Diarization and Transcription per segment -> who said what
- Parsing segment with spaCy to extract semantics

---

### Observations

Whipser and Pyannote do not return the same timestamp.

Whisper:
```json
{
	"segment_id": "whisper_000001",
	"time": {
	  "start_seconds": 9.81,
	  "end_seconds": 26.85,
	  "duration_seconds": 17.04,
	  "start_ts": "00:00:09.81",
	  "end_ts": "00:00:26.85"
},
...
"segments_count": 5945,
```

Pyannote:
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
	"speaker_id": "SPEAKER_37",
	"confidence": null
  },
  {
	"segment_id": "dia_000002",
	"time": {
	  "start_seconds": 17.074718750000002,
	  "end_seconds": 18.762218750000002,
	  "duration_seconds": 1.6875,
	  "start_ts": "00:00:17.07",
	  "end_ts": "00:00:18.76"
	},
	"speaker_id": "SPEAKER_37",
	"confidence": null
},
...
"segments_count": 1033
```

By looking at the first segment logic, we could infer that Pyannote will eventually produce much more segments than Whisper.

But not at all, in the end Pyannote has ~ 1/6 segments compared to whisper. 

I guess that initially the Pyannote model is segmenting a lot, to prevent the embedding of multiple voices.
But, as the analysis move forward, the speakers might have a better flow? 

Transcript segments are the reference segments, and I attribute speaker to it, not the other way around by trying to attribute transcript to speaker segments. 

---
### Overlaps transcription diarization

```python
def overlap_seconds(
    a_start: float, a_end: float, b_start: float, b_end: float
) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))
    
...

ov = overlap_seconds(t_start, t_end, d_start, d_end)
```

We compare the overlaps, for instance for the second diarization segement we d have:
```python
ov = overlap_seconds(9.81, 26.85, 10.15, 14.6)

min(26.85, 14.6) = 14.6
max(9.81, 10.15) = 10.15
max(14.6 - 10.15) = 4.45

```

The speaker of the first segment of the Diarization covers ~ 40% of the first transcription. 

---
### Merging

The pipeline gets a copy of the current JSON, and creates a new one with added keys:

```json
{
  "schema_version": "0.1.0",
  "source": {...},
  "pipeline": {...},
  "transcript": {...},
  "diarization": {...},
  "segments": [
	  {
      "segment_id": "seg_000001",
      "time": {
        "start_seconds": 9.81,
        "end_seconds": 26.85,
        "duration_seconds": 17.04,
        "start_ts": "00:00:09.81",
        "end_ts": "00:00:26.85"
      },
      "speaker": {
        "speaker_id": "SPEAKER_37",
        "speaker_label": null,
        "speaker_label_source": null,
        "confidence": 0.7875751907276997
      },
      "text": {
        "raw": "Bonjour à tous, je vous ai entendu monsieur Cordier, la séance est ouverte et pas de photo effectivement. L'ordre du jour appelle les questions au gouvernement. La première va être posée par monsieur Paul Christophe, président du groupe Horizon.",
        "normalized": null,
        "language": "fr"
      },
      "flags": 0,
      "other_speakers": [],
      "entities": [],
      "keywords": [],
      "provenance": {
        "transcript_segment_ids": [
          "whisper_000001"
        ],
        "diarization_segment_ids": [
          "dia_000001",
          "dia_000002",
          "dia_000003"
        ],
        "stage_created_by": "merge"
      }
    },
    ...
  ]
}

```

The first segments of the transcript *needs* 3 diarization segments to cover the whole duration. 

But `whisper_000011` to `whisper_000017` are covered by a singled pyannote segment `dia_000033`.

---

### NLP enrichment

A new key per segment is added in the new JSON.
```json
"nlp": {
        "spacy": {
          "entities": [
            {
              "text": "Cordier",
              "label": "PER",
              "start": 9,
              "end": 10,
              "ignored": false
            },
            {
              "text": "Paul Christophe",
              "label": "PER",
              "start": 38,
              "end": 40,
              "ignored": false
            }
          ],
          "tokens": [
            {
              "i": 0,
              "text": "Bonjour",
              "lemma": "Bonjour",
              "pos": "PROPN",
              "dep": "obl:mod",
              "head_i": 7,
              "head_text": "entendu",
              "ent_type": "MISC",
              "morph": "Gender=Masc|Number=Sing"
            },
            {
              "i": 2,
              "text": "tous",
              "lemma": "tout",
              "pos": "ADJ",
              "dep": "nmod",
              "head_i": 0,
              "head_text": "Bonjour",
              "ent_type": "",
              "morph": "Gender=Masc|Number=Plur"
            },
            {
              "i": 4,
              "text": "je",
              "lemma": "je",
              "pos": "PRON",
              "dep": "nsubj",
              "head_i": 7,
              "head_text": "entendu",
              "ent_type": "",
              "morph": "Number=Sing|Person=1"
            },
            {
              "i": 5,
              "text": "vous",
              "lemma": "vous",
              "pos": "PRON",
              "dep": "iobj",
              "head_i": 7,
              "head_text": "entendu",
              "ent_type": "",
              "morph": "Number=Plur|Person=2"
            },
```

Two different runs with SPACY were made, one was *lossless* where every NER is kept, even *wrong* one like:
```json
"nlp": {
        "spacy": {
          "entities": [
            {
              "text": "Bonjour",
              "label": "MISC",
              "start": 0,
              "end": 1,
              "ignored": true
            },
```

*monster  21MB JSON*

So in an attempt to reduce it I cretaed another that uses some blacklist to prevent spaCy from extracting what I'd qualified as low signal or directly useless. 
```python
IGNORE_PERS = {"merci", "voici"}
IMPORTANT_LEMMAS = {"entendre", "dire", "répondre", "être"}
IMPORTANT_WORDS = {"monsieur", "madame", "m", "mme"}
KEEP_DEPS = {"nsubj", "csubj", "obj", "iobj", "obl:agent", "appos", "flat:name","nmod"}
```
*13MB JSON*

Even if this was making sense while coding, when writing those lines it hit me that this is wrong : as any logic shift in the following pipeline would require a new spaCy run. 

>The lossless extracting will be kept, because once spaCy is ran, all information are inside the data to then filter and apply logic, creating yet another *stage* in the pipeline, but prevent having to re run everything each time a new idea of filtering pops.

>spaCy ( `small model`) runs on the 6k segments in ~ 15 seconds, might as well extract everything, and run the logic after, instead of doing it all at once.

---

Next step is creating the logic to infer speaker from the extracted PER, finding words and lemma and their temporal indice to map them to their SpeakerID.

Before this, I ll create a `.npz` to save: pyannote's clustered centroid of the end run per speaker, all embedding of all segments, filter out segments that are *noisy* ( multiple speakers, very short duration, hallucination / gibberish transcript ... ), recompute the centroid from *clean* segments, and compare the PCA to confirm this was a good idea to have *cleaner* centroid. 