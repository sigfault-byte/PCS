The code was sync on the *inference machine* that is a 4080 13g VRAM laptop.

Multiple errors were made:
- upgrading the machine
- using uv sync, without the additional flag for the *inference*
- not disabling the *sleep* on inactivity, relying on `systemd-inihbit` 
- not thinking to mirror every different parameters as the previous runs.

The usual audio file was split using `ffmpg` into a 30 min segments.
So the smoke test would be fact and easy to verify.

Then two new audio of ~ 4 hours each were downloaded. But for some reason the official website of the french assembly was not displaying the usual *download* button. 
Therefore some usual tricks were abused to get the video, and convert it to `.wav`

# First smoke run

### First error

The audio was *chomped* randomly from `00:15:00` to `00:45:00`.
The audio ends during a current speech. 
Pyannote was, as usual, *surrounding* the detected frame to embed it, but was trying to embed after the final time by 0.064 seconds.

> Now pyannote receives a clamp that is *end-time-stamp - 1 seconds*

Leaving it enough room in case this situation happens again.
### Second error

During the merging turns, the pipeline was failing, due to *empty* diarization segments.

The log file was exposing:

```text
...
  File "/PCS/src/assemblybot/db/loaders/chunks.py", line 245, in load_embedding_records
    raise MissingEmbeddingDependencyError(
assemblybot.db.loaders.chunks.MissingEmbeddingDependencyError: No turns exist for session_id=1

Finished candidate: test30min-assemblee-nationale-14-avril-2026.wav -> failed
Pipeline batch complete: 0 ok/skipped, 1 failed.

```

Initially, i thought that for some reasons, the last segment of diarization was empty because of the clamping rule, which was completly a wrong assumption.

Turns out the diarization orchestration was reading the json, saving the objects, but never writing them back on the new json.

Once this got fixed, and after some shenanigans with CUDA / CUNN / Libtorch ...

The first smoke test worked, and gave somewhat correct results on the small dataset.

# First real run 

The first 4 hour file was added to the *unprocessed* directory, and the script was launched.

> All previous runs were on `.mp3` files. This was the first on `.wav`. This little detail created a lot of confusion

During the inference of Whisper, the laptop went idle, and was woke back up.
The whisper inference was unusually longer than ever. Instead of the usual  ~ 13 min, it took 40.

> In the meantime after diving a little bit more into whisper inference. Turns out whisper is not reading the raw audio, and acts more like a *CNN / VIT* model, by interpreting 30 second frames of log scaled mel spectrogram. 

The resulted run was immediately queried with the rag quick script, where usual health check were completely ignore...

# Second real run

The second run was immediately launch.
Launching it with `systemd-inhibit`. But it did not work, so it was cancel, and re run with a good old *infinite video* on the foreground.

Just like before, whisper was taking much more time than usual.
This whole run took 61 min, for a 4 hour audio, where the usual test audio of 5 hour took ~ 30 min.

This time, instead of testing the RAG system, the number of identified PER was the first command, and only 2 were identified.

# Investigation

#### Initial hypothesis: pyannote regression

The first suspicion was that the new clamping logic introduced during the smoke test had side effects on diarization.

This hypothesis was quickly discarded:

* diarization outputs were present
* diarization segment counts looked reasonable
* the failure pattern appeared much later in the pipeline

The issue was therefore unlikely to originate from pyannote.

#### Initial hypothesis: .wav vs .mp3

The previous successful runs had all been executed on .mp3 files.

The first real runs used raw `.wav` files extracted from downloaded video sources.

This raised concerns about:

* sample rate mismatches
* channel layout differences
* ffmpeg conversion artifacts
* unexpected metadata

However, inspection of the generated audio and intermediate artifacts showed no obvious corruption.

No direct evidence was found linking the file format change to the observed failures....
### **Suspicion: interrupted GPU inference**

Both real runs shared a common characteristic:

- Whisper inference duration increased dramatically
- the laptop entered an idle/sleep state during the first run
- the machine resumed afterwards
- the second run exhibited similar symptoms despite attempts to prevent sleep

Historical observations suggested that CUDA workloads may become unstable after suspend/resume cycles on this laptop.

While not formally proven, this became a primary suspect.

# **Observation**

|**Run**|**Audio length**|**Whisper time**|
|---|---|---|
|Baseline|~5 h|~13 min|
|First run|~4 h|~40 min|
|Second run|~4 h|~60 min|

The degradation was significant enough to indicate abnormal execution conditions.

### **Evidence of Whisper degradation**

The strongest signal came from downstream quality metrics.

The second run produced:

- only 2 identified PER entities
- 0 flag-free turns
- large sections of repetitive text

Manual inspection revealed that Whisper entered a hallucination loop approximately 20 minutes into the audio and continued generating repetitive content until near the end of the recording.

This explained several downstream symptoms:

- speaker attribution quality collapsed
- turn segmentation quality collapsed
- NER extraction nearly disappeared
- RAG quality became meaningless

The pipeline itself was operating on invalid transcription data.

# **Root cause**

The immediate root cause of the failed runs was not the RAG system, turn merging, diarization, or speaker identification.

The root cause was a corrupted Whisper transcription.

The exact trigger, however, remained uncertain.

Several hypotheses were investigated:

- suspend/resume during GPU execution
- CUDA state corruption after wake-up
- environment drift after dependency upgrades
- Whisper decoding instability on very long uninterrupted speech segments

Subsequent experiments provided stronger evidence for the final hypothesis.

At the time of writing, interactions with the execution environment cannot be completely ruled out, but the data increasingly points toward a Whisper decoding failure mode rather than a pure infrastructure issue. 
Although a lot of energy was wasted thinking the configuration parameters passed to whisper was not working.

### **Additional observations and testing**

Multiple runs were performed again on one of the hallucination-prone audio files.
Hallucinations occurred consistently near the same timestamps.

Both problematic regions corresponded to extremely long uninterrupted speeches (>14k generated tokens).

This immediately distinguished them from the previously validated 5-hour recording, whose longest uninterrupted intervention was ~ half that length.

No direct proof could be established... 
but the observation was enough to trigger further investigation.

A run on the previous 5-hour audio was then executed.

The output was effectively identical to the earlier successful run, except that the original source was an `.mp3`.

To eliminate file format as a factor, the same audio was converted to `.wav` and processed again.

The results were nearly identical:

- 214 turns on the `.wav` versus 213 on the `.mp3`
- 67 turns with identified `mentioned_per` on the `.wav` versus 69 on the `.mp3`
- 165 PER records extracted on the `.wav` versus 166 on the `.mp3`

These differences were small enough to suggest that the audio format itself was not the primary cause.

Whisper parameters were then modified:

```json
{
  "condition_on_previous_text": false
}
```

This immediately eliminated the hallucination loops on the problematic recordings.

While not definitive proof, it strongly suggested that the issue was related to Whisper’s decoding strategy on long uninterrupted speeches.

### **Corrective actions beyond parameter changes**

Implemented:

- clamp pyannote embedding windows at audio boundaries
- fix diarization orchestration JSON persistence bug
- re-run inference from scratch after wake-up events
- verify PER counts before evaluating downstream stages
- verify turn quality before testing retrieval
- mirror inference parameters from known-good runs

Some files still appear to require `"condition_on_previous_text": false`.

> Further experimentation is required to fully understand the underlying mechanism...

### **Future improvements**

- record inference timing metrics automatically
- store pipeline configuration snapshots alongside each run
- add transcription sanity checks before downstream execution
- automatically propose a re-run with `"condition_on_previous_text": false` when excessive repetition is detected
- reject runs exhibiting excessive transcript repetition
- disable sleep entirely during long-running inference jobs

The most expensive mistake was not the bug itself, but trusting downstream metrics before validating the transcription layer. 
Once the transcript was corrupted, every subsequent component appeared broken despite functioning correctly.