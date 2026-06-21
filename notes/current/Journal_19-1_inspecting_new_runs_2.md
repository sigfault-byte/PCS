# Second audio

`1ere-seance--renforcer-la-solidarite-envers-les-retraites-pauvres--nationalisation-d-arcelormittal-11-juin-2026`

| Duration (s) | Vad coverage (s) | Ratio |
| ------------ | ---------------- | ----- |
| 15302.12     | 12248.6          | 0.800 |

## Pipeline
|     stage     |              model               | device |
|---------------|----------------------------------|--------|
| vad           | silero-vad                       |        |
| transcription | large-v3                         | cuda   |
| diarization   | pyannote/speaker-diarization-3.1 | cuda   |
| embedding     | h4c5/sts-camembert-base          |        |

## Session
| id |                             slug                             |                            title                             |    date    | source_url | duration_seconds | vad_duration |                       audio_file_hash                        |
|----|--------------------------------------------------------------|--------------------------------------------------------------|------------|------------|------------------|--------------|--------------------------------------------------------------|
| 1  | 1ere-seance--renforcer-la-solidarite-envers-les-retraites-pa | 1ere seance renforcer la solidarite envers les retraites pau | 2026-06-11 |            | 15302.12         | 12248.6      | b9b64036029d12e0244ed9a89d9fb53a697e8fcbba94a3d43c9d9a6eed7e |
|    | uvres--nationalisation-d-arcelormittal-11-juin-2026          | vres nationalisation d arcelormittal                         |            |            |                  |              | 29f1                                                         |

---

## Table counts
|     table_name      | rows |
|---------------------|------|
| person              | 110  |
| speaker_cluster     | 37   |
| transcript_segment  | 2488 |
| diarization_segment | 1231 |
| turn                | 146  |
| turn_analysis       | 146  |
| semantic_chunk      | 563  |
| embedding           | 709  |

---

## Mentioned persons extraction coverage
| total_turns | empty_mentions | empty_mentions_pct |
| ----------- | -------------- | ------------------ |
| 146         | 41             | 28.1%              |

## speaker identified per turn
|      kind      | count(*) |  pct  |
|----------------|----------|-------|
| assembly_chair | 47       | 32.2% |
| deputy         | 27       | 18.5% |
| minister       | 5        | 3.4%  |
| raw_per        | 47       | 32.2% |

---

## identified person distribution
|      kind      | count |  pct  |
|----------------|-------|-------|
| assembly_chair | 1     | 0.9%  |
| minister       | 2     | 1.8%  |
| deputy         | 29    | 26.4% |
| raw_per        | 78    | 70.9% |

---

## Flag distribution
| turns |   flags    | flag_pct |
|-------|------------|----------|
| 51    | 0          | 34.9%    |
| 29    | 2048       | 19.9%    |
| 8     | 4096       | 5.5%     |
| 32    | 6144       | 21.9%    |
| 10    | 14336      | 6.8%     |
| 8     | 3221225472 | 5.5%     |
| 1     | 3221227520 | 0.7%     |
| 7     | 3221231616 | 4.8%     |

---

### Clean turns
| clean_turns |
|-------------|
| 51          |

### Ratio clean / total
| clean_pct |
|-----------|
| 34.9%     |

---
## Transcription health check
### compression_ratio distribution
| bucket | segments | segment_pct |
|--------|----------|-------------|
| 1      | 2481     | 99.7%       |
| 2      | 7        | 0.3%        |

### avg_log_prob distribution
| bucket | count(*) | segment_pct |
|--------|----------|-------------|
| -0.2   | 3        | 0.1%        |
| -0.15  | 82       | 3.3%        |
| -0.1   | 872      | 35.0%       |
| -0.05  | 1496     | 60.1%       |
| 0.0    | 35       | 1.4%        |

