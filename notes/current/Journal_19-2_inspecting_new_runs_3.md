# Third audio

From the original 5 h audio, but this uses the `.wav` format

`1ere-seance--questions-au-gouvernement--simplification-de-la-vie-economique-cmp--renforcer-la-s-14-avril-2026`

| Duration(s) | Vad detected(s) | Ratio actual VAD |
| ----------- | --------------- | ---------------- |
| 19194.94    | 14872.4         | 0.7748           |

## Pipeline
|     stage     |              model               | device |
|---------------|----------------------------------|--------|
| vad           | silero-vad                       |        |
| transcription | large-v3                         | cuda   |
| diarization   | pyannote/speaker-diarization-3.1 | cuda   |
| embedding     | h4c5/sts-camembert-base          |        |

## Session
| id |                             slug                             |                            title                             |    date    | source_url | duration_seconds |   vad_duration   |                       audio_file_hash                        |
|----|--------------------------------------------------------------|--------------------------------------------------------------|------------|------------|------------------|------------------|--------------------------------------------------------------|
| 1  | 1ere-seance--questions-au-gouvernement--simplification-de-la | 1ere seance questions au gouvernement simplification de la v | 2026-04-14 |            | 19194.94         | 14872.4000000001 | 7af45d5bc1899e30b66a994bff740efb31a1a340aa8b4fa1255c0dd7ba55 |
|    | -vie-economique-cmp--renforcer-la-s-14-avril-2026            | ie economique cmp renforcer la s                             |            |            |                  |                  | 3a3d                                                         |

---

## Table counts
|     table_name      | rows |
|---------------------|------|
| person              | 165  |
| speaker_cluster     | 63   |
| transcript_segment  | 3300 |
| diarization_segment | 1369 |
| turn                | 214  |
| turn_analysis       | 214  |
| semantic_chunk      | 720  |
| embedding           | 934  |

---

## Mentioned persons extraction coverage
| total_turns | empty_mentions | empty_mentions_pct |
|-------------|----------------|--------------------|
| 214         | 67             | 31.3%              |

## speaker identified per turn
|      kind      | count(*) |  pct  |
|----------------|----------|-------|
| assembly_chair | 68       | 31.8% |
| deputy         | 42       | 19.6% |
| minister       | 21       | 9.8%  |
| raw_per        | 13       | 6.1%  |

---

## identified person distribution
|      kind      | count |  pct  |
|----------------|-------|-------|
| assembly_chair | 1     | 0.6%  |
| minister       | 11    | 6.7%  |
| deputy         | 43    | 26.1% |
| raw_per        | 110   | 66.7% |

---

## Flag distribution
| turns |   flags    | flag_pct |
|-------|------------|----------|
| 99    | 0          | 46.3%    |
| 1     | 64         | 0.5%     |
| 1     | 65         | 0.5%     |
| 1     | 66         | 0.5%     |
| 35    | 2048       | 16.4%    |
| 1     | 2112       | 0.5%     |
| 1     | 2115       | 0.5%     |
| 5     | 4096       | 2.3%     |
| 1     | 4161       | 0.5%     |
| 25    | 6144       | 11.7%    |
| 2     | 10240      | 0.9%     |
| 3     | 14336      | 1.4%     |
| 23    | 3221225472 | 10.7%    |
| 1     | 3221225475 | 0.5%     |
| 1     | 3221225536 | 0.5%     |
| 5     | 3221227520 | 2.3%     |
| 1     | 3221227586 | 0.5%     |
| 7     | 3221231616 | 3.3%     |
| 1     | 3221239808 | 0.5%     |

---

### Clean turns
| clean_turns |
|-------------|
| 99          |

### Ratio clean / total
| clean_pct |
|-----------|
| 46.3%     |

---
## Transcription health check
### compression_ratio distribution
| bucket | segments | segment_pct |
|--------|----------|-------------|
| 1      | 3281     | 99.4%       |
| 2      | 19       | 0.6%        |

### avg_log_prob distribution
| bucket | count(*) | segment_pct |
|--------|----------|-------------|
| -0.3   | 5        | 0.2%        |
| -0.25  | 5        | 0.2%        |
| -0.2   | 9        | 0.3%        |
| -0.15  | 81       | 2.5%        |
| -0.1   | 712      | 21.6%       |
| -0.05  | 2337     | 70.8%       |
| 0.0    | 151      | 4.6%        |