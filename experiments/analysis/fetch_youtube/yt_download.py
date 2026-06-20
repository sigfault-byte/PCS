from typing import Any

import yt_dlp

# YOUTUBE_URL = "https://www.youtube.com/watch?v=Wfj2i-hKoG4"
# OUTPUT_WAV = "assemblee_nov26_2024.wav"
YOUTUBE_URL = "https://videos.assemblee-nationale.fr/video.19128777_6a2a596fc95d5.1ere-seance--renforcer-la-solidarite-envers-les-retraites-pauvres--nationalisation-d-arcelormittal-11-juin-2026"
OUTPUT_WAV = "1ere-seance--renforcer-la-solidarite-envers-les-retraites-pauvres--nationalisation-d-arcelormittal-11-juin-2026.wav"


ydl_opts: dict[str, Any] = {
    "format": "bestaudio/best",
    "outtmpl": OUTPUT_WAV.replace(".wav", ".%(ext)s"),
    "postprocessors": [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }
    ],
    "postprocessor_args": [
        "-ar",
        "16000",  # sample rate
        "-ac",
        "1",  # mono
    ],
}

try:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # type: ignore
        ydl.download([YOUTUBE_URL])
    print(f"✓ Downloaded and converted to {OUTPUT_WAV}\n")
except Exception as e:
    print(f"Error downloading: {e}")
    exit(1)
