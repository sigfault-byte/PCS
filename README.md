# PCS

PCS is a data engineering and AI pipeline that transforms long French parliamentary audio sessions into a relational database enriched with speaker identities, semantic search capabilities, and audit metadata.

The pipeline combines voice activity detection, speaker diarization, speech recognition, entity extraction, semantic embeddings, audio-quality audits, and SQLite persistence.

Rather than producing only a transcript, PCS tracks uncertainty throughout the processing chain, highlighting where the audio, transcript, speaker attribution, and official parliamentary record agree, disagree, or require review.

A typical 5-hour parliamentary session (~580 MB WAV) produces a self-contained SQLite database of roughly 5 MB containing transcripts, speaker identities, named entities, semantic chunks, embeddings, and audit metadata.

Across multiple test runs, roughly 13 hours of parliamentary audio were distilled into about 13 MB of SQLite databases.

The resulting database can be explored with SQL, searched semantically, or inspected through generated audit artifacts.

### Example Output

```text
5h 20m parliamentary audio
          │
          ▼
+-----------------------+
|  assemblybot.sqlite   |
|        5.4 MB         |
+-----------------------+

13 relational tables
214 speaking turns
45 identified speakers
720 semantic chunks
934 embeddings
```

A standalone, dependency-free search demo is included in demo/, allowing the generated SQLite database to be explored immediately with nothing more than Python.
[demo](demo/)

## Why This Exists

Long-form audio processing involves multiple imperfect sources of information.

Speech recognition can hallucinate or drift. Speaker diarization estimates who spoke when but may split or merge speakers incorrectly. Voice activity detection identifies probable speech regions without understanding their content. Official parliamentary records provide valuable context but are edited documents rather than direct acoustic observations.

Instead of hiding uncertainty, the pipeline preserves it. Each stage contributes evidence, confidence signals, provenance, audit metrics, and intermediate artifacts. Disagreements between sources are treated as useful information rather than failures.

The goal is not to produce a perfectly clean transcript. The goal is to produce a structured dataset where uncertain regions can be inspected, filtered, reviewed, or reprocessed with full traceability.

## Pipeline Overview

```text
.wav
  │
  ▼
Voice Activity Detection
  │
  ▼
Speaker Diarization
  │
  ▼
Transcription
  │
  ▼
Audio & Transcript Audits
  │
  ▼
Speaker / Transcript Alignment
  │
  ▼
Turn Consolidation
  │
  ▼
Identity Resolution
  │
  ▼
Semantic Chunking & Embeddings
  │
  ▼
SQLite Export
```

## Current technologies

| Stage | Technology |
|---------|---------|
| Voice Activity Detection | Silero VAD |
| Speaker Diarization | pyannote |
| Transcription | Faster-Whisper (large-v3) |
| Audio Audit | librosa |
| Named Entity Recognition | CamemBERT NER |
| Semantic Embeddings | STS CamemBERT |
| Persistence | SQLite + SQLAlchemy |


## What It Does

Given one or more `.wav` files into `data/audio/unprocessed`, `main.py` discovers candidate audio files, creates a reproducible run folder, executes the pipeline stage by stage, writes a manifest after every stage, and moves the source file to `processed` or `failed` depending on the result.

The current pipeline runs:

1. **Silero VAD** to detect speech regions.
2. **pyannote diarization** to estimate who spoke when and generate speaker embeddings for later identity resolution.
3. **Faster Whisper transcription** to generate timestamped French transcript segments.
4. **librosa audio audit** to compute acoustic metrics such as energy, dB, spectral flatness, and related quality signals.
5. **Transcript audit** to flag suspicious transcript regions using VAD, diarization, audio evidence, whisper confidence proxy and heuristics.
6. **Transcript/diarization alignment** to connect text segments with speaker activity.
7. **Turn consolidation** to produce speaker turns.
8. **Named-entity extraction and speaker identity resolution** using French NER models and ground-truth lists of deputies and ministers.
9. **Semantic chunking and embeddings** for semantic search and retrieval.
10. **SQLite export** for queryable downstream analysis.

Each run is stored in its own reproducible run directory containing configuration snapshots, logs, intermediate artifacts, audit outputs, embeddings, and the final SQLite database.

## Repository Map

* `main.py` — pipeline entrypoint.
* `src/assemblybot/stages/` — processing stages (VAD, diarization, transcription, audits, enrichment, export).
* `src/assemblybot/orchestration/` — run management, manifests, provenance, and pipeline execution.
* `src/assemblybot/models/` — shared domain models and artifacts.
* `src/assemblybot/db/` — SQLite schema and export logic.
* `tests/` — unit and integration tests.
* `docs/` — technical documentation and diagrams.
* `notes/` — experiment journals and design decisions.
* `ground_truth_PER/` — deputy and minister reference datasets.

## Requirements

- Python >= 3.12
- uv
- NVIDIA GPU recommended
- Hugging Face access token required for model downloads

Tested on:

- RTX 4080 Laptop GPU (12 GB VRAM)
- CUDA 13
- Linux

## Running The Pipeline

Place one or more `.wav` files in:

```text
data/audio/unprocessed/
```

Run the default batch:

```bash
uv run python main.py
```

### Performance

Tested on:

- RTX 4080 Laptop GPU (12 GB VRAM)
- CUDA 13

Typical processing time:

```text
Input:   5h 20m parliamentary session
Runtime: 26 minutes

≈ 12× faster than real time
```


## Reproducibility and Traceability

This project is deliberately designed around reproducibility, provenance, and inspection.

* Run-level provenance, including source file hashes, ingestion metadata, and processing timestamps.
* Stage-by-stage manifests recording status, runtime, configuration, and generated artifacts.
* Persistent intermediate artifacts to allow inspection and reprocessing without rerunning the full pipeline.
* Audit signals and quality flags for suspicious transcript, diarization, or audio regions.
* Relational persistence through SQLite for downstream analysis and retrieval.
* Automated tests covering alignment, enrichment, chunking, and database export workflows.

The goal is to make every output traceable to the evidence that produced it. Model outputs are not treated as truth by default; they are treated as observations that can be validated, audited, and revisited.

## Further Reading

The journals document the evolution of the pipeline, including experiments, failed approaches, design decisions, and architectural changes.

- [Journal 05: Whisper VAD value](notes/current/Journal_05_whisper_vad_value.md) - comparison of Whisper VAD settings and failure modes.
- [Journal 08: flags and audits](notes/current/Journal_8_flags_audits.md) - how segment-level quality flags evolved.
- [Journal 11: Merging diarization and transcription](notes/current/Journal_11_merging_pyannote_and_whisper_experiment.md) - finding a strategy to align speaker activity with transcript segments.
- [Journal 13: PER extraction exploration](notes/current/Journal_13_PER_extraction_and_exploration.md) - early speaker/person extraction experiments.
- [Journal 14: Semantic Search / RAG](notes/current/Journal_14_experimenting_with_rag.md) - chunking strategy and early search results
- [Journal 16: SQLite v1](notes/current/Journal_16-sqlite-v1.md) - database schema direction.

Other journals are in `notes/current/`.

## Roadmap

PCS runs end to end on French parliamentary audio and produces auditable intermediate artifacts, speaker and person enrichment, semantic embeddings, and a searchable SQLite database.

Current areas of work include:

* improving speaker identity recall
* refining speaker attribution and confidence scoring
* expanding evaluation against manually reviewed samples
* improving retrieval quality through chunk and segment filtering
* simplifying the search and query experience for downstream users
