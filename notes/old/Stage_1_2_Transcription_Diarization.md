# Overview

- Structuring the project correctly (modular pipeline)
- Running Whisper transcription on GPU ( float16 as oppose to int8 previously on CPU runs)
- Running pyannote diarization
- Designing a stable canonical JSON schema
- Fighting through a full CUDA / cuDNN / dependency hell (and winning)

---
## Architecture Progress

### Pipeline Direction

1.	Input: .wav ( Also updated the video -> audio converter to enforce 16kHz sample rate_ )
2.	Stage 1: Transcription (Whisper)
3.	Stage 2: Diarization (pyannote)
4.	Future: Merge -> NLP -> enrichment -> search

### Observations: 

Previous diarization runs on CPU (int8) showed noticeable errors, which could lead to **noisy speaker embeddings and unstable centroids**.

To mitigate this, I introduced a conceptual intermediate step:

Stage 1.5: Transcript parsing / filtering
Goal:
- detect very short 
- detect gibberish / non-French content
- compute ratio of valid words vs noise

These segments can later be:
- flagged ( bit flags )
- removed
- or attenuated before downstream processing (e.g. embedding extraction + cross embedding comparison across multiple session)

---

### Canonical JSON Strategy

Canonical JSON is enriched step-by-step, never mutated destructively

Each stage:
- reads JSON
- enriches it
- writes a new version

---

## Transcription (Whisper)

Result:
-	Ran faster-whisper large-v3 on RTX 4080
-	Performance: ~17x realtime
-	4h audio processed in ~15 minutes
-	~6000 segments generated

Outputs
-	.txt transcript (human readable)
-	.json transcript (canonical structured)

```json 
"transcript": {
  "engine": {
    "name": "faster-whisper",
    "model": "large-v3",
    "device": "cuda",
    "compute_type": "float16"
  },
  "language_detected": "fr",
  "language_probability": 1,
  "segments_count": 5945,
  "raw_segments": [...]
}
```

### Observations

-	CUDA auto-detection
-	Compute type auto (float16 on GPU)
-	Progress bar (real-time + speed)
-	Clean timestamp handling (HH:MM:SS.xx + raw seconds)

The jump in quality from CPU int8 to GPU float16 is **very noticeable**, especially around the ~4h10 mark.

---

### Diarization (pyannote)

  Result
  - Successfully ran pyannote/speaker-diarization-3.1
  - ~45 speakers detected on 4h session 
  - Raw segments correctly extracted

```json
"diarization": {
  "engine": {
    "name": "pyannote",
    "model": "pyannote/speaker-diarization-3.1",
    "device": "cuda"
  },
  "speakers_count": 45,
  "segments_count": 1033,
  "raw_segments": [...],
  "speaker_embeddings": [],
}
```

Key issues encoutered:

 **1.** `itertracks` API C
- New API returns DiarizeOutput
- Correct usage:
```python 
annotation = diarization.speaker_diarization
annotation.itertracks(...)
```

**2. Audio decoding crash (torchcodec)**
``` bash
Could not load libtorchcodec
libnppicc.so.13 missing
```
 - TorchCodec + CUDA + FFmpeg mismatch. 
 
Bypassed torchcodec entirely by loading audio manually:
 ```python
 waveform, sample_rate = torchaudio.load(...)
pipeline({
    "waveform": waveform,
    "sample_rate": sample_rate,
})
 ```

---

**3. CUDA / cuDNN Battle**
This was painful.

**Problems encountered**
- `libcublas.so.12` missing
- `libcudnn_ops_infer.so.8` missing
- `cuDNN 9` installed but incompatible
- `torchcodec` expecting older CUDA libs
- missing `libnppicc.so.13`

Config:
- PyTorch: CUDA 13
- System: Pop!_OS (Ubuntu 24.04)
- NVIDIA stack mismatch

Solved with:
- Installing correct CUDA libraries
- Installing cuDNN manually
- Updating ctranslate2
- Avoiding fragile components (torchcodec)

---

## Thoughts

CPU int8 transcription produced occasional severe anomalies:

```text
[4:09:51.63 -> 4:09:53.63] Voilà ma réponse à vos trois questions.
[4:09:53.63 -> 4:09:57.26] Merci.
[4:10:03.23 -> 4:10:04.23] Merci.
...
[4:12:04.78 -> 4:12:07.44] outre ce Feu de France, il y a deshehe D'accord !
[4:12:07.44 -> 4:12:08.44] réягé ?
[4:12:08.52 -> 4:12:09.21] Ah oui oui
[4:12:09.21 -> 4:12:09.97] Bleghth
[4:12:09.97 -> 4:12:10.70] Là, c'était effectivement un peu nigré, mais c'était vraiment une sorte…
```

Whereas GPU float16 output is clean and coherent:
```text
[04:09:51.96 -> 04:09:53.96] Voilà ma réponse à vos trois questions.
[04:09:53.96 -> 04:10:01.30] Merci Madame la Présidente.
[04:10:01.30 -> 04:10:05.30] Monsieur le ministre, la gouvernance des agences et opérateurs publics
[04:10:05.30 -> 04:10:08.30] et la répartition des compétences sont au cœur d'une action publique
[04:10:08.30 -> 04:10:10.30] claire et efficace, vous l'avez rappelé vous-même.
[04:10:10.30 -> 04:10:13.30] Et en matière de santé, ce qui sera mon focus,
```

This is very intriguing. I wonder if the CPU was overloaded and prone to errors, or if CPU floats are less precise than GPU floats?
Anyhow, CPU inference is tricky. 

Possible explanations
- Quantization artifacts (int8)
	 -> loss of precision in logits → unstable token selection
- Softmax instability
	-> small logit differences collapse → wrong token distribution
- Hallucination loops
	-> repeated tokens (“merci”)
. CPU inference instability
	(but NOT due to float precision — CPUs use float32 just fine !)

### Planned experiment
- Compare GPU int8 vs GPU float16
- Use Levenshtein distance per segment
- Treat float16 as reference
Goal:
- identify noisy segments
- derive a signal of transcription instability
- use that to filter segments for embeddings
This would be:
- model-agnostic
- audio-agnostic
- reusable signal for pipeline quality

---
### Next stage : Merge
- Align transcript segments with diarization

This will enable:
- NLP analysis ( spaCy )
- search
- fact checking
- speaker attribution 
  
>Example:
> “La parole est à …” → map to correct speaker ID

---

### Embedding Strategy (Exploration)
Pyannote exposes embeddings.

Planned approach:
1.	Let pyannote cluster normally
2.	Store embeddings (likely per speaker or segment hopefully)
3.	After filtering noisy segments:
	- recompute centroids
	- compare before/after

Use PCA for visualization:
- check cluster compactness
- detect outliers





