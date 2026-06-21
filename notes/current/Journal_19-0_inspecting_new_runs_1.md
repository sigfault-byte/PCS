# First audio :

`1ere-seance--questions-au-gouvernement--conventions-france-finlande-et-france-suede-en-matiere-d-i-9-juin-2026.wav`

| Duration(s) | Vad detected(s) | Ratio actual VAD |
| ----------- | --------------- | ---------------- |
| 14495.24    | 11339.4         | 0.782            |

# Overall health

A bash script was created in order to simplify a basic health check of every run.

Part of the output is:
 
## # SQLite Health Check

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
| 1  | 1ere-seance--questions-au-gouvernement--conventions-france-f | 1ere seance questions au gouvernement conventions france fin | 2026-06-09 |            | 14495.24         | 11339.4      | 342c2d1388317ddb094a47c498578abd921131b5b481f39968e908031bc8 |
|    | inlande-et-france-suede-en-matiere-d-i-9-juin-2026           | lande et france suede en matiere d i                         |            |            |                  |              | 3eef                                                         |

---

## Table counts
|     table_name      | rows |
|---------------------|------|
| person              | 122  |
| speaker_cluster     | 55   |
| transcript_segment  | 2384 |
| diarization_segment | 1126 |
| turn                | 159  |
| turn_analysis       | 159  |
| semantic_chunk      | 565  |
| embedding           | 724  |

---

## Mentioned persons extraction coverage
| total_turns | empty_mentions | empty_mentions_pct |
|-------------|----------------|--------------------|
| 159         | 41             | 25.8%              |

## speaker identified per turn
|      kind      | count(*) |  pct  |
|----------------|----------|-------|
| assembly_chair | 51       | 32.1% |
| deputy         | 25       | 15.7% |
| minister       | 19       | 11.9% |
| raw_per        | 9        | 5.7%  |

---

## identified person distribution
|      kind      | count |  pct  |
|----------------|-------|-------|
| assembly_chair | 1     | 0.8%  |
| minister       | 9     | 7.4%  |
| deputy         | 40    | 32.8% |
| raw_per        | 72    | 59.0% |

---

## Flag distribution
| turns |   flags    | flag_pct |
|-------|------------|----------|
| 90    | 0          | 56.6%    |
| 13    | 2048       | 8.2%     |
| 8     | 4096       | 5.0%     |
| 22    | 6144       | 13.8%    |
| 2     | 14336      | 1.3%     |
| 11    | 3221225472 | 6.9%     |
| 2     | 3221227520 | 1.3%     |
| 1     | 3221229568 | 0.6%     |
| 10    | 3221231616 | 6.3%     |

---

### Clean turns
| clean_turns |
|-------------|
| 90          |

### Ratio clean / total
| clean_pct |
|-----------|
| 56.6%     |

---
## Transcription health check
### compression_ratio distribution
| bucket | segments | segment_pct |
|--------|----------|-------------|
| 1      | 2379     | 99.8%       |
| 2      | 5        | 0.2%        |

### avg_log_prob distribution
| bucket | count(*) | segment_pct |
|--------|----------|-------------|
| -0.15  | 87       | 3.6%        |
| -0.1   | 877      | 36.8%       |
| -0.05  | 1392     | 58.4%       |
| 0.0    | 28       | 1.2%        |
