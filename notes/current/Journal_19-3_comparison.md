
In order to get a better idea of how the pipeline is behaving, comparing the three runs should help.

The VAD coverage is similar
(0.78 + 0.80 + 0,77) / 3 ~ = 0.78


| Audio                                     | VAD coverage |
| ----------------------------------------- | ------------ |
| [[Journal_19-0_inspecting_new_runs_1\|1]] | 0.78         |
| [[Journal_19-1_inspecting_new_runs_2\|2]] | 0.8          |
| [[Journal_19-2_inspecting_new_runs_3\|3]] | 0.77         |


# Health

The table count of the sql script expose the number of element per table. 
Focusing on theses:
- person ( extracted names from the text)
- speaker_cluster ( number of speaker identified by pyannote)
- transcript_segments
- diarization_segments
- turns ( transcript and diarization merge)
- semantic_chunk

## Distribution and health

| Audio                                     | ws/t | ds/t | sw/ds | ct    | ws/c |
| ----------------------------------------- | ---- | ---- | ----- | ----- | ---- |
| [[Journal_19-0_inspecting_new_runs_1\|1]] | 14.9 | 7.0  | 2.1   | 56.6% | 4.2  |
| [[Journal_19-1_inspecting_new_runs_2\|2]] | 17.0 | 8.4  | 2     | 34.9% | 4.4  |
| [[Journal_19-2_inspecting_new_runs_3\|3]] | 15.4 | 6.3  | 2.4   | 46.3% | 4.5  |

ws : whisper_segment
ds : diarization_segment
ct : clean turn flag free
p: persons
t : turn
c: chunk

## PER And turns, chunks

| Audio                                     | p   | p/t  |
| ----------------------------------------- | --- | ---- |
| [[Journal_19-0_inspecting_new_runs_1\|1]] | 122 | 0.7  |
| [[Journal_19-1_inspecting_new_runs_2\|2]] | 110 | 0.7  |
| [[Journal_19-2_inspecting_new_runs_3\|3]] | 165 | 0.77 |
|                                           |     |      |
