# Pipeline Chronology

This is the current end-to-end chronology when running `main.py`.

## 0. Entry Point And Queue Discovery

`main.py` parses CLI arguments, then asks `discover_candidates()` for audio files to process.

By default it scans:

```text
data/audio/unprocessed/
```

You can also pass one or more explicit files with `--file`.

The runner currently expects `.wav` files. Non-`.wav` inputs are skipped and moved to:

```text
data/audio/failed/
```

## 1. Run Folder Setup

For every candidate audio file, `PipelineRunner.run_candidate()` builds provenance:

- original filename
- detected session date from the filename, if present
- title/session slug
- file size
- SHA-256 hash
- ingestion timestamp

Then it creates a unique run directory under:

```text
data/runs/{ingestion-date}_{detected-date}_{session-slug}_{sha8}/
```

The source audio is moved from:

```text
data/audio/unprocessed/
```

to:

```text
data/audio/processing/
```

Then a copy is placed inside the run directory:

```text
input/{original_filename}.wav
```

The run also writes:

```text
config/pipeline_config.json
manifest.json
logs/pipeline.log
```

`manifest.json` is updated before and after every stage. `pipeline.log` captures the stage output.

## 2. Stage 1: VAD

File:

```text
src/assemblybot/stages/vad.py
```

Purpose: run Silero VAD to detect speech regions in the audio.

Creates:

```text
interim/{stem}_0_vad.json
```

## 3. Stage 2: Diarization

File:

```text
src/assemblybot/stages/diarization.py
```

Purpose: run pyannote diarization to identify speaker segments.

Creates:

```text
interim/{stem}_01_diarization.json
embeddings/pyannote/{stem}_01_segment_embeddings.npz
embeddings/pyannote/{stem}_01_speaker_centroids.npz
```

The embedding outputs are recorded in the manifest as a diarization sub-stage.

## 4. Stage 3: Transcription

File:

```text
src/assemblybot/stages/transcription.py
```

Purpose: run Faster Whisper transcription.

Creates:

```text
interim/{stem}_02_transcription.json
interim/{stem}_02_transcript.txt
```

## 5. Stage 4: Audio Audit

File:

```text
src/assemblybot/stages/audio_audit.py
```

Purpose: compute low-level audio metrics used later for quality checks.

Creates:

```text
audit/meta_json/{stem}_audio_audit.json
```

## 6. Stage 5: Whisper Segment Audit

File:

```text
src/assemblybot/stages/whisper_segment_audit.py
```

Purpose: compare Whisper segments with diarization and audio-audit evidence, then flag suspicious transcript segments.

Reads:

```text
interim/{stem}_02_transcription.json
interim/{stem}_01_diarization.json
audit/meta_json/{stem}_audio_audit.json
```

Creates:

```text
audit/turn_json/{stem}_02_transcription_flagged.json
audit/turn_json/whisper_segment_audio_audit.json
```

## 7. Stage 6: Alignment

File:

```text
src/assemblybot/stages/alignment.py
```

Purpose: align flagged transcript segments with diarization segments and propagate nearby anomaly flags.

Reads:

```text
audit/turn_json/{stem}_02_transcription_flagged.json
```

Creates:

```text
interim/{stem}_03_alignment.json
```

## 8. Stage 7: Turns

File:

```text
src/assemblybot/stages/turns.py
```

Purpose: consolidate aligned transcript and diarization segments into speaker turns.

Reads:

```text
interim/{stem}_03_alignment.json
```

Creates:

```text
audit/turn_json/{stem}_01_turns.json
```

## 9. Stage 8: PER Extraction

File:

```text
src/assemblybot/stages/per_extraction.py
```

Purpose: enrich turns with person/entity evidence using the deputies and ministers ground-truth CSV files.

Reads:

```text
audit/turn_json/{stem}_01_turns.json
docs/liste_deputes_libre_office_2026-06.csv
docs/liste_ministre_2026.csv
```

Creates:

```text
audit/turn_json/{stem}_02_per_extraction.json
```

## 10. Stage 9: Semantic Chunking

File:

```text
src/assemblybot/stages/semantic_chunk.py
```

Purpose: build embeddings and semantic chunks for retrieval/RAG.

Reads:

```text
audit/turn_json/{stem}_02_per_extraction.json
```

Creates:

```text
embeddings/rag/turn_embeddings.npz
embeddings/rag/semantic_chunks.npz
embeddings/rag/semantic_chunk_metadata.json
embeddings/rag/semantic_chunk_metadata.txt
```

## 11. Stage 10: SQLite Build

File:

```text
src/assemblybot/stages/build_sqlite.py
```

Purpose: load the final structured pipeline outputs into a per-run SQLite database.

Reads:

```text
interim/{stem}_03_alignment.json
audit/turn_json/{stem}_02_per_extraction.json
input/{original_filename}.wav
embeddings/rag/turn_embeddings.npz
embeddings/rag/semantic_chunks.npz
embeddings/rag/semantic_chunk_metadata.json
```

Creates:

```text
sqlite/assemblybot.sqlite
```

## 12. Run Completion

If all stages succeed, the original audio file is moved from:

```text
data/audio/processing/
```

to:

```text
data/audio/processed/
```

The manifest is updated with:

- `status: success`
- final audio destination
- stage durations
- artifacts created by each stage

If a stage fails, the manifest is updated with:

- `status: failed`
- error message
- traceback
- all successful stage artifacts up to that point

The original audio file is then moved to:

```text
data/audio/failed/
```

## Summary Shape

```text
main.py
  -> discover audio candidates
  -> create per-run folder
  -> move audio into processing
  -> copy audio into run/input
  -> write config + initial manifest
  -> Stage 1: VAD
  -> Stage 2: diarization + pyannote embeddings
  -> Stage 3: transcription
  -> Stage 4: audio audit
  -> Stage 5: Whisper segment audit
  -> Stage 6: alignment
  -> Stage 7: turns
  -> Stage 8: PER extraction
  -> Stage 9: semantic chunking + RAG embeddings
  -> Stage 10: SQLite build
  -> move original audio to processed or failed
  -> write final manifest
```
