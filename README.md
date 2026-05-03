
## Current Status

- Modular pipeline stages implemented
- Multiple full session experiments conducted
- Merge logic under active redesign (*prioritizing precision to eliminate hallucination*) 
- No unified end-to-end runner yet

## Goal

Build a pipeline that ingests raw parliamentary sessions (video/audio), and transforms them into structured, searchable data:

* transcription
* speaker diarization
* segment merging
* speaker attribution
* storage and retrieval  
  
The goal is to produce a **reliable approximation of the truth**

Parliamentary audio is inherently noisy:

- multiple speakers overlap
- interruptions are frequent
- speakers cut each other mid-sentence
- the session president often interjects
- background noise and applause introduce ambiguity  

This results in conflicting and incomplete signals that must be reconciled carefully.  

The pipeline prioritizes:

precision over recall, using multiple validation layers to reduce uncertainty.

---

## Approach

The pipeline was redesigned after earlier iterations produced unreliable results during:

- segment merging
- speaker identification
- NER extraction

The current design follows a multi-stage, validation-driven approach:

### 00 VAD (Silero)

Initial pass to extract candidate speech regions.

Future work includes additional signal-based filtering (e.g. RMS, spectral features via librosa) to better identify “clean” audio segments.


### 01 Diarization (pyannote)

- segment speakers
- compute per-segment embeddings
- compute initial speaker centroids


### 02 Transcription (Whisper)

- VAD-enabled transcription (1000 ms)
- extract:
    - segments
    - tokens
    - confidence proxies (avg_log_prob, no_speech_prob, compression ratio)


### 03 Heuristic filtering

Identify “clean” segments based on:

- no speaker overlap
- stable confidence metrics
- low noise indicators


### 04 Alignment

Align transcription with diarization (Silero + pyannote):

- detect silence gaps
- identify mismatches
- filter likely hallucinated tokens


### 05 NLP (CamemBERT)

- extract named entities (speakers)
- infer speaker attribution from context
- validate against diarization turns
- extract keywords / topics


### 06 Embedding refinement

- recompute speaker centroids using only high-confidence segments
- compute semantic embeddings for clean segments


### 07 Storage

Store structured segments in SQLite:

- timestamps
- speaker attribution
- text
- embeddings
- keywords


---

The main challenge is that truth is not directly observable

Each component provides an imperfect signal:

- Whisper : probabilistic text (can hallucinate)
- Diarization: approximate speaker boundaries
- VAD : imperfect segmentation
- NLP : contextual but fallible inference

The problem becomes : reconciling **multiple weak signals** into a **consistent**, **high confidence representation**

This is less about *“running models”* and more about:

- signal validation
- conflict resolution
- uncertainty management

---

