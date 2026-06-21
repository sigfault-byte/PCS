The idea is to extract all PER found by CamemBERT. 
But keeping only strong scored ones.

A first script, `experiments/camenBERT/testing.py`, was written to understand how to use CamemBERT for NER extraction.

# Usage

CamenBERT is called by importing `from transformer import pipeline`

```python
ner = pipeline(
    "token-classification",
    model="Jean-Baptiste/camembert-ner",
    aggregation_strategy="simple",
    token=token,
)
```
*`token` is the HF_TOKEN stored in the environment. It is used to authenticate against HuggingFace and avoid anonymous download rate limits.*

Then we run the model against the turns newly created by the previous merging stage:

```python
entities = []
for turn in turns[:25]:
    for ent in ner(turn["text"]):
        if ent["entity_group"] == "PER" and ent["score"] > 0.90:
            entities.append(
                {"entity": ent, "text": turn["text"], "id": turn["turn_id"]}
            )
```

### Example

Turn 42 is 
```json
...
"text": "Merci beaucoup Monsieur le Ministre. La parole est à présent à Madame Gabrielle Catala pour la France Insoumise."
...
```

CamenBERT ner outputs:
```bash
[
	{'entity_group': 'PER',
	'score': np.float32(0.7847673),
	'word': 'Monsieur le Ministre',
	'start': 14,
	'end': 35
	},
	{'entity_group': 'PER',
	 'score': np.float32(0.99891824),
	 'word': 'Gabrielle Catala', 
	 'start': 69,
	 'end': 86
	 },
	 {'entity_group': 'ORG',
	 'score': np.float32(0.99445677),
	 'word': 'France Insoumise',
	 'start': 94,
	 'end': 111
	 }
]
```

CamemBERT exposes a confidence score for each detected entity. 
This score is used to filter detections and prioritize precision over recall.

# Who speak when

```python
NEXT_SPEAKER_PATTERNS = [
    "la parole est à",
    "je donne la parole à",
    "vous avez la parole",
    "est à vous",
]
...
PREVIOUS_SPEAKER_PATTERNS = [
    "merci",
    "merci madame",
    "merci monsieur",
]
...
def context_window(text, start, end, radius=80):
    return text[max(0, start - radius) : min(len(text), end + radius)].lower()
...
	if any(p in ctx for p in NEXT_SPEAKER_PATTERNS):
		role = "probable_next_speaker"
	elif any(p in ctx for p in PREVIOUS_SPEAKER_PATTERNS):
		role = "probable_previous_speaker"
```

By exploiting the procedural structure of the Assembly session we can infer who **was** talking or is **about** to talk.

The naive logic is :

```text
Turn n     : "Merci Madame la Présidente"      -> infer previous speaker
Turn n + 1 : regular intervention
Turn n + 2 : "La parole est à Monsieur X"      -> infer next speaker
```

# Trying against the whole 213 turns

In order to fix the multiple issue that can occur in the PER extraction, because the transcription may fail to recover the exact orthography of a deputy name, the official list of the deputies was used to compare to the extracted PER.

Using `rapidfuzz`
First run using `fuzz.token_sort_ratio`
that identifies : `50 speaker references were identified.` and `21 number of deputees' exact name`

An example of a missed name was **HADRIEN*** vs **ADRIEN**. 
token_sort_ratio is too sensitive to the length.

And once changed to `fuzz.WRatio` the number of deputies was `25`

So out of 213 turn, 50 PER where identified.
From those 50 only 25 were successfully matched to an official deputy record.

# Association matrix between CamemBERT-inferred persons and Pyannote speaker IDs.

```python
for speaker_id, counts in matrix.items():
    person, count = counts.most_common(1)[0]
    total = sum(counts.values())
    purity = count / total

    speaker_id_to_person[speaker_id] = {
        "person": person,
        "count": count,
        "total": total,
        "purity": purity,
    }
```

Example:
```text
SPEAKER_21
  raw_per:laurent nunez: 2
  raw_per:monsieur le ministre: 1
  raw_per:madame la presidente: 1
================================================================================
SPEAKER_40
  raw_per:madame la presidente: 8
  raw_per:ministre: 1
  deputy:643192:chantal jourdan: 1
  raw_per:bergman: 1
  deputy:794154:jean-claude raux: 1
  raw_per:monsieur le ministre du travail: 1
  raw_per:aurore berger: 1
  raw_per:monsieur le premier ministre: 1
  raw_per:marc angeli: 1
  raw_per:lopez-ligori: 1
  raw_per:feignet: 1
```

Using a simple purity metric, the following high-confidence candidates were obtained:
```text
35 of High purity (90+) out fo 50
19 of High purity (90+) ID'd deputy out of 25
```

# Errors

The propagation logic assumes that the next detected speaker will appear in the immediately following turn. 

This assumption is frequently violated by the President of the Assembly, who often inserts procedural remarks between the nomination of a speaker and the actual intervention.

##### Fix 
After fixing the associative loop:

```text
  raw_per:aurore berger: 1				      <	
							      >	SPEAKER_33
							      >	  raw_per:aurore berger: 2
							      >	=============================================================
SPEAKER_33						      <	
  raw_per:aurore berger: 1				      <	
============================================================= <	
  raw_per:emmanuel morel: 1				      <	
  raw_per:laurent mazory: 1				      <	
							      >	  raw_per:emmanuel morel: 1
  raw_per:laurent mazory: 1				      |	  raw_per:laurent mazory: 2
total:      18						      |	total:      17
purity:     44.44%					      |	purity:     47.06%
							      >	speaker_id: SPEAKER_33
							      >	person:     raw_per:aurore berger
							      >	count:      2
							      >	total:      2
							      >	purity:     100.00%
							      >	=============================================================
speaker_id: SPEAKER_57					      |	speaker_id: SPEAKER_57                                       
person:     raw_per:aurelien lopez ligaurie		      <	
purity:     100.00%					      |	purity:     100.00%                                          
============================================================= <	
count:      1						      |	count:      1                                                
total:      1						      <	
speaker_id: SPEAKER_11					      |	speaker_id: SPEAKER_11                                       
person:     deputy:841665:denis fegne			      <	
purity:     100.00%					      |	purity:     100.00%                                          
============================================================= <	
speaker_id: SPEAKER_33					      <	
person:     raw_per:aurore berger			      <	
count:      1						      <	
total:      1						      <	
purity:     100.00%					      <	
============================================================= <	
purity:     50.00%					      |	purity:     50.00%                                           
============================================================= <	
count:      1						      |	count:      1                                                
total:      2						      <	
speaker_id: SPEAKER_43					      |	speaker_id: SPEAKER_43                                       
person:     raw_per:serge papin				      <	
purity:     100.00%					      |	purity:     100.00%                                          
============================================================= <	
count:      14						      |	count:      14                                               
total:      28						      |	purity:     53.85%
purity:     50.00%					      <	
total:      2						      |	total:      3
purity:     100.00%					      |	purity:     66.67%
count:      1						      |	count:      2
total:      1						      |	total:      2
35 of High purity (90+)					      |	34 of High purity (90+)
19 of High purity (90+) ID'd deputy			      |	18 of High purity (90+) ID'd deputy
\ No newline at end of file

```

It actually fixed some mistakes, not as much as hoped. 
No more test will be run to compare, as this version is more logical.

### quick tweaks nonetheless

The purity threshold strongly affects the number of identified deputies.

At a 90% threshold:
- 25 deputies identified
- 19 high-confidence deputy associations

At an 80% threshold:
- 32 deputies identified
- 22 high-confidence deputy associations

This suggests a significant number of speaker clusters lie close to the decision boundary and may benefit from future speaker-cluster refinement.

---

The next step will be to filter out of theses turns, the one that are *high purity* and that were not previously flag by the pipeline, so the centroid could be computed again on clean segments, hence producing a much better voiceprint associated to a real depute ID.

---

#  PCA Exploration 

While experimenting with CamemBERT, I generated embeddings for all turns and projected them using PCA to investigate whether any structure would emerge.

![[Camenbert-turn-embeddings-PCA.png]]
*Every red dot is speaker 40 or 36 -> the president of the assembly*

Interestingly, I thought that the clustered points were mainly short sentences, and the one in the scattered space where longer speech.

But this was wrong:

![[CamemBERT_PCA-turn-time.png]]*Here the dots are colored depending on the duration of the speech*

The scattered points are the shorts one. Even if the cluster does have short duration, all of the long speech are in the cluster. 
# Observation

CamenBERT embeds in 700+ dimensions. 

The PCA axes do not directly correspond to human-interpretable concepts. PCA selects the directions that explain the largest amount of variance in the embedding space, but these directions may represent any combination of linguistic, stylistic, procedural, semantic, or structural features.

The PCA nevertheless reveals visible structure.
In particular, the President's interventions appear to occupy a characteristic region of the embedding space. Further investigation is required to determine whether this signal can be exploited as an additional feature for speaker identification and cluster refinement.

