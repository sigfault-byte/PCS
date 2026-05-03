import os

import soundfile as sf
import torch
from pyannote.audio import Pipeline

hf_token = os.environ["HF_TOKEN"]

audio, sample_rate = sf.read("data/input/test_10min.wav", dtype="float32")

if audio.ndim == 1:
    waveform = torch.from_numpy(audio).unsqueeze(0)
else:
    waveform = torch.from_numpy(audio.T)

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=hf_token,
)
assert pipeline is not None

if torch.cuda.is_available():
    pipeline.to(torch.device("cuda"))

diarization = pipeline(
    {
        "waveform": waveform,
        "sample_rate": sample_rate,
    }
)

annotation = diarization.speaker_diarization

for turn, _, speaker in annotation.itertracks(yield_label=True):
    print(f"[{turn.start:.2f} -> {turn.end:.2f}] {speaker}")
