The goal is to have a portable single unified sqlite .db file, not meant to be edited, alongside a python script to query it with a CLI. 
User should be able to download a python file + a .toml + .db
Do `uv sync`
Get a menu explaining the options
And then could do queries similar to
```bash
one session.sqlite
-> raw provenance
-> aligned segments
-> turns
-> identities
-> RAG chunks
-> embeddings
-> validation stats


assembly-db inspect session.sqlite
assembly-db speakers session.sqlite
assembly-db turn session.sqlite 42
assembly-db search session.sqlite “protoxyde d’azote”
assembly-db export session.sqlite report.md »
```

# Pipeline Run

```text
pipeline_run
------------
id
schema_ver                 
session_id
stage              -- vad / transcription / diarization / analysis / embedding
engine_name        -- silero-vad / faster-whisper / pyannote
model              -- large-v3 / speaker-diarization-3.1
device             -- cuda / cpu
config_json        -- all options
```
One row per stage of the pipeline

# Session

```text
session
-------
id
slug             --Optional
title
date
source_url       --Optional
duration_seconds
vad_duration     [fetch on the vad json object]
audio_file_hash
```

---
# Speaker

```text
speaker_cluster
---------------
id
session_id
label                    -- SPEAKER_36
total_detected_speech
purity
majority_person_id
evidence_purity FLOAT,
absolute_purity FLOAT,
```

```text
person
------
id
name
kind              -- minister/deputy/chair/etc
external_id       -- assembly id, government id, etc.
party
role
voice_centroid    [recomputed from clean segments]
```

---

# Transcription

```text
transcript_segment
------------------
id
pipeline_run_id
text
start_seconds
end_seconds
flags
avg_log_prob
no_speech_prob
compression_ratio
```

---

# Diarization

```text
diarization_segment
-------------------
id
pipeline_run_id
speaker_cluster_id
start_seconds
end_seconds
flags
overlap_speaker_ids        --list at most len <= 3
```

---
# Turn

```text
turn
----
id
session_id
speaker_cluster_id
speaker_confidence
text
start_seconds
end_seconds
flags
```

# Turn Analysis

```text
turn_analysis
-------------
id
current_person_id          -- FK to person, nullable
current_person_source      -- PER/fuzzy/confusion matrix/manual
current_person_purity
embedding             - embedding.astype(np.float32).tobytes()
keywords_json
organizations_json
mentioned_person_ids_json 
```
The json is to store arrays, and to be sure they stay arrays.
Turn analysis and Turn share the same ID. Turn_id 1 = Turn_Analysis_id 1

### Link tables

```text
turn_transcript_segment
-----------------------
turn_id
transcript_segment_id
```

```text
turn_diarization_segment
------------------------
turn_id
diarization_segment_id
```

# Rag like

```text
embedding
---------
id
model_name
dimension
vector
dtype
normalized
```

```text
semantic_chunk
--------------
id
turn_id
chunk_index
text
embedding_id
```

```text
turn_embedding
--------------
turn_id
embedding_id
```
---
# Flags

Currently there are not many flags, no SQL needed just decode with Python `IntFlag`
```python
# General quality: bits 0-9
# Text is too dense for a very short segment: word or character rate exceeds
# the configured plausible speech threshold.
IMPOSSIBLE_SPEECH_RATE = 1 << 0
# UTF-8 byte rate is too high for the segment duration, often indicating
# repeated text, symbols, or another dense transcript artifact.
INFORMATION_RATE_TOO_HIGH = 1 << 1
# Manual or downstream catch-all marker for a segment that should be
# inspected even if no more specific flag applies.
NEEDS_REVIEW = 1 << 2
# Intended for text that is nonsensical after transcription or normalization.
GIBBERISH = 1 << 3
# Intended for text detected as not French in a French-focused pipeline.
NON_FRENCH = 1 << 4
# Audio is mostly quiet, but has a short transient event that may have caused
# Whisper to hallucinate speech.
MOSTLY_SILENCE_WITH_SHORT_EVENT = 1 << 5
# propagate flag to segments neighboor to a noisy one
ADJACENT_INFORMATION_RATE_ANOMALY = 1 << 6

# VAD alignment: bits 10-19
# Whisper segment has no overlap with any VAD speech interval.
OUTSIDE_VAD = 1 << 10
# Whisper segment overlaps VAD, but coverage is below the partial-coverage
# threshold.
PARTIAL_VAD_OVERLAP = 1 << 11
# VAD coverage inside the Whisper segment has an internal gap longer than
# the configured threshold.
DISCONTIGUOUS_VAD_COVERAGE = 1 << 12
# Long Whisper segment has too little total VAD coverage.
LONG_WHISPER_SEGMENT_LOW_VAD_COVERAGE = 1 << 13

# Whisper quality: bits 20-29
# Whisper average log probability is below the configured confidence floor.
LOW_WHISPER_CONFIDENCE = 1 << 20
# Whisper no-speech probability is above the configured threshold.
HIGH_NO_SPEECH_PROB = 1 << 21
# Whisper compression ratio is above the configured threshold, a common sign
# of repetitive or degenerate decoder output.
HIGH_COMPRESSION_RATIO = 1 << 22
# Segment duration is long, but the text has too few words or characters.
LONG_DURATION_SHORT_TEXT = 1 << 23

# Diarization quality: bits 30-39
# Diarization segment intersects a diarization overlap region.
DIARIZATION_OVERLAP = 1 << 30
# Segment may contain more than one speaker. In the diarization stage this
# means an overlap region has at least two speakers; in the Whisper audit it
# means the Whisper segment intersects any diarization overlap region.
MULTI_SPEAKER_CANDIDATE = 1 << 31
# Intended for segments close to a speaker boundary where attribution may be
# unstable.
SPEAKER_CHANGE_NEARBY = 1 << 32
# Speaker assignment was decided by deterministic tie break rather than
# stronger diarization evidence.
TIE_BREAK_SPEAKER = 1 << 33

# Merge integrity: bits 40-49
# Intended for transcript content that could not be matched to diarization.
ORPHAN_TRANSCRIPT = 1 << 40
# Intended for diarization speech that could not be matched to transcript
# content.
ORPHAN_DIARIZATION = 1 << 41
# Diarization segments that are linked to a whisper hallucination / noise
UNSAFE_FOR_SPEAKER_CENTROID = 1 << 42
```


# Frozen schema fro v1:

```sql
sqlite> .schema
CREATE TABLE person (
        id INTEGER NOT NULL,
        name VARCHAR NOT NULL,
        normalized_name VARCHAR NOT NULL,
        kind VARCHAR NOT NULL,
        external_id VARCHAR,
        party VARCHAR,
        role VARCHAR,
        canonical_voice_centroid BLOB,
        PRIMARY KEY (id),
        CONSTRAINT uq_person_name_kind UNIQUE (normalized_name, kind)
);
CREATE TABLE session (
        id INTEGER NOT NULL,
        slug VARCHAR,
        title VARCHAR NOT NULL,
        date DATE NOT NULL,
        source_url VARCHAR,
        duration_seconds FLOAT,
        vad_duration FLOAT,
        audio_file_hash VARCHAR(64) NOT NULL,
        PRIMARY KEY (id),
        UNIQUE (slug)
);
CREATE TABLE pipeline_run (
        id INTEGER NOT NULL,
        schema_ver VARCHAR NOT NULL,
        session_id INTEGER NOT NULL,
        stage VARCHAR NOT NULL,
        engine_name VARCHAR NOT NULL,
        model VARCHAR,
        device VARCHAR,
        config_json JSON NOT NULL,
        PRIMARY KEY (id),
        CONSTRAINT uq_pipeline_run_session_stage UNIQUE (session_id, stage),
        FOREIGN KEY(session_id) REFERENCES session (id)
);
CREATE TABLE speaker_cluster (
        id INTEGER NOT NULL,
        session_id INTEGER NOT NULL,
        label VARCHAR NOT NULL,
        total_detected_speech FLOAT NOT NULL,
        majority_person_id INTEGER,
        evidence_purity FLOAT,
        absolute_purity FLOAT,
        PRIMARY KEY (id),
        CONSTRAINT uq_speaker_cluster_session_label UNIQUE (session_id, label),
        FOREIGN KEY(session_id) REFERENCES session (id),
        FOREIGN KEY(majority_person_id) REFERENCES person (id)
);
CREATE TABLE diarization_segment (
        id INTEGER NOT NULL,
        pipeline_run_id INTEGER NOT NULL,
        speaker_cluster_id INTEGER NOT NULL,
        start_seconds FLOAT NOT NULL,
        end_seconds FLOAT NOT NULL,
        flags BIGINT NOT NULL,
        overlap_speaker_ids JSON NOT NULL,
        PRIMARY KEY (id),
        FOREIGN KEY(pipeline_run_id) REFERENCES pipeline_run (id),
        FOREIGN KEY(speaker_cluster_id) REFERENCES speaker_cluster (id)
);
CREATE TABLE transcript_segment (
        id INTEGER NOT NULL,
        pipeline_run_id INTEGER NOT NULL,
        text TEXT NOT NULL,
        start_seconds FLOAT NOT NULL,
        end_seconds FLOAT NOT NULL,
        flags BIGINT NOT NULL,
        avg_log_prob FLOAT,
        no_speech_prob FLOAT,
        compression_ratio FLOAT,
        PRIMARY KEY (id),
        FOREIGN KEY(pipeline_run_id) REFERENCES pipeline_run (id)
);
CREATE TABLE turn (
        id INTEGER NOT NULL,
        session_id INTEGER NOT NULL,
        speaker_cluster_id INTEGER NOT NULL,
        speaker_confidence FLOAT NOT NULL,
        speaker_evidence_ratio FLOAT NOT NULL,
        text TEXT NOT NULL,
        start_seconds FLOAT NOT NULL,
        end_seconds FLOAT NOT NULL,
        flags BIGINT NOT NULL,
        PRIMARY KEY (id),
        FOREIGN KEY(session_id) REFERENCES session (id),
        FOREIGN KEY(speaker_cluster_id) REFERENCES speaker_cluster (id)
);
CREATE TABLE embedding (
        id INTEGER NOT NULL,
        session_id INTEGER NOT NULL,
        pipeline_run_id INTEGER NOT NULL,
        model_name VARCHAR NOT NULL,
        dimension INTEGER NOT NULL,
        vector BLOB NOT NULL,
        dtype VARCHAR NOT NULL,
        normalized BOOLEAN NOT NULL,
        PRIMARY KEY (id),
        FOREIGN KEY(session_id) REFERENCES session (id),
        FOREIGN KEY(pipeline_run_id) REFERENCES pipeline_run (id)
);
CREATE TABLE turn_analysis (
        id INTEGER NOT NULL,
        turn_id INTEGER NOT NULL,
        current_person_id INTEGER,
        current_person_source VARCHAR,
        current_person_purity FLOAT,
        embedding BLOB,
        keywords_json JSON NOT NULL,
        organizations_json JSON NOT NULL,
        mentioned_persons_json JSON NOT NULL,
        speaker_identity_evidence_json JSON NOT NULL,
        PRIMARY KEY (id),
        FOREIGN KEY(turn_id) REFERENCES turn (id),
        FOREIGN KEY(current_person_id) REFERENCES person (id)
);
CREATE TABLE turn_transcript_segment (
        turn_id INTEGER NOT NULL,
        transcript_segment_id INTEGER NOT NULL,
        PRIMARY KEY (turn_id, transcript_segment_id),
        FOREIGN KEY(turn_id) REFERENCES turn (id),
        FOREIGN KEY(transcript_segment_id) REFERENCES transcript_segment (id)
);
CREATE TABLE turn_diarization_segment (
        turn_id INTEGER NOT NULL,
        diarization_segment_id INTEGER NOT NULL,
        PRIMARY KEY (turn_id, diarization_segment_id),
        FOREIGN KEY(turn_id) REFERENCES turn (id),
        FOREIGN KEY(diarization_segment_id) REFERENCES diarization_segment (id)
);
CREATE TABLE semantic_chunk (
        id INTEGER NOT NULL,
        pipeline_run_id INTEGER NOT NULL,
        turn_id INTEGER NOT NULL,
        chunk_index INTEGER NOT NULL,
        start_sentence_index INTEGER NOT NULL,
        end_sentence_index INTEGER NOT NULL,
        text TEXT NOT NULL,
        embedding_id INTEGER NOT NULL,
        PRIMARY KEY (id),
        CONSTRAINT uq_semantic_chunk_turn_index UNIQUE (turn_id, chunk_index),
        FOREIGN KEY(pipeline_run_id) REFERENCES pipeline_run (id),
        FOREIGN KEY(turn_id) REFERENCES turn (id),
        FOREIGN KEY(embedding_id) REFERENCES embedding (id)
);
CREATE TABLE turn_embedding (
        turn_id INTEGER NOT NULL,
        embedding_id INTEGER NOT NULL,
        PRIMARY KEY (turn_id, embedding_id),
        FOREIGN KEY(turn_id) REFERENCES turn (id),
        FOREIGN KEY(embedding_id) REFERENCES embedding (id)
);
```