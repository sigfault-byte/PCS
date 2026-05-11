
Because the foundation of transcription and diarization is, after all, extracting signals from the raw audio, exploring the audio using `librosa` was necessary.

---
# Quick audio crash-course

Sound is vibration. More specifically, sound is a pressure wave moving through a medium such as air.

When a sound is traveling, it pushes and pulls nearby air, creating a sound wave traveling outward.

Example of a -normalized- digital sound wave could be:
```text
0.00, 0.25, 0.70, 0.25, 0.00, -0.25, -0.70, -0.25, 0.00
```
0 -> no displacement from the center
\>1 -> pressure in one direction
<1 -> pressure in the opposite direction

### Sample rate
How many measurement is made per seconds.
### Amplitude
Amplitude is instantaneous -> size of a wave at time `t`
Loudness is how strong the soud *feels* to a listener

They are related but **NOT** identical.
Loudness depends on: amplitude, frequency, duration, surrounding sound... 

A single sample does not tell much about a soundwave, because audio is -bascially-  `f(time(x))`

### Frequency
How many cycle per second. 1Hz -> 1 cycle per sec.
Pitch is not an audio signal, it is the brain interpretation of a frequency.
The lower the frequency, the lower the pitch, the higher the frequency the higher the pitch.
> 20 Hz < Human hear < 20 kHz

# Metrics


1. Root Mean Square, a.k.a RMS

General formula is `RMS = sqrt(mean(x^2))`
The square is to prevent the negative amplitude from cancelling the positive one.
If a audio signal is `[0.5, -0.5, 0.5, -0.5]` -> square -> `[0.25, 0.25, 0.25, 0.25]`
Mean is 0.25, square root -> 0.5
RMS is 0.5

- estimates signal strength
- not loudness, measures signal energy. 

2. Decibel, a.k.a dB

It is *usually* not an absolute amount, it is a comparison.
It is log10 based, so decibel doubling do not mean double the strength.

3. Zero crossing Rate, a.k.a ZCR

Related to frequency, it measure how many time a wave crosses the horizontal axis.
Used in VAD to detect differences betwee *voice, no-voice, silence*

4. Spectral / spectrogram

A snapshot / plot of frequency over time.
Usually : 
- horizontal axis is time
- vertical is frequency
- color is intensity / energy

	4.1 spectral centroid
	Weighted mean of the combined frequency

	4.2 spectral bandwidth
	standard deviation. of the frequencies 

	4.3 spectral flatness
	geometric mean compared to arithmetic mean. Usually normalized from 0 to 1.
	Can be peaky or flat. Represent the spread.
	If spectrum A is `[0, 0, 0, 10, 0, 0, 0]`
	and spectrum B is `[4, 5, 4, 5, 4, 5, 4]`
	The most of the energy is concentrated in one frequency in A, this is *peaky* flatness is low
	B the energy is spread, flatness is high
	Can help measure noise like content, but it should be crossed with all other metrics.

# Example from the current session.

A first pass calculated every metrics in 100ms chunks.

A second pass aggregates theses data to generate 
- mean 
- median 
- std 
- min 
- max 
- p10 
- p90 

For each metrics, on a segment of detected pitch from Silero. 

#### Beginning of speech

```json
{
  "segment_id": "vad_000001",
  "time": {
    "start_seconds": 913.5,
    "end_seconds": 916.1,
    "duration_seconds": 2.6000000000000227,
    "start_ts": "00:15:13.50",
    "end_ts": "00:15:16.10"
  },
  "confidence": null,
  "librosa_window_count": 26,
  "librosa_stats": {
    "rms": {
      "mean": 0.06132906137307692,
      "median": 0.0649182163,
      "std": 0.014948008547417898,
      "min": 0.0204872247,
      "max": 0.0809066817,
      "p10": 0.03982640615,
      "p90": 0.0786142796
    },
    "db": {
      "mean": -5.682745933534615,
      "median": -4.82746982575,
      "std": 2.7544023729854303,
      "min": -14.8449707031,
      "max": -2.9149475098,
      "p10": -9.2495136261,
      "p90": -3.165143013
    },
    "zcr": {
      "mean": 0.11217322716153846,
      "median": 0.10180664065,
      "std": 0.03152100578921824,
      "min": 0.0546875,
      "max": 0.1801757812,
      "p10": 0.07849121095,
      "p90": 0.1566162109
    },
    "spectral_centroid": {
      "mean": 1669.4417016938692,
      "median": 1482.48703830875,
      "std": 549.263764100084,
      "min": 740.4506161986,
      "max": 2687.9595051892,
      "p10": 1055.43611859425,
      "p90": 2440.37037075125
    },
    "spectral_bandwidth": {
      "mean": 1796.239030430023,
      "median": 1701.6883037796001,
      "std": 441.46327299436615,
      "min": 1189.2809920968,
      "max": 2577.632984801,
      "p10": 1227.6967468516,
      "p90": 2427.6873378875
    },
    "spectral_flatness": {
      "mean": 0.013205645507692309,
      "median": 0.007760003,
      "std": 0.014038104127965387,
      "min": 0.0003465523,
      "max": 0.0525726192,
      "p10": 0.0007086625500000001,
      "p90": 0.03221013675
    }
```

RMS:
Mean and median are close, with a relatively low standard deviation.
This suggests the segment has fairly stable signal energy overall.
The lower minimum probably corresponds to the beginning or end of the detected VAD region, where speech energy ramps in or out.

dB:
The dB values are the logscaled version of RMS, so their distribution is expected to look different.
The median around -4.8 dB and p90 around -3.1 dB indicate that most of the segment is close to the loudest parts of the file.
The minimum around -14.8 dB shows that a small part of the segment is much softer, likely speech onset/offset or Silero padding.
This is useful as an energy-stability signal, but it should not be interpreted as absolute loudness.

ZCR:
The ZCR values are moderate and fairly stable.
This is consistent with speech containing both voiced parts, such as vowels, and unvoiced/fricative parts, such as “s”, “f”, or “ch”.
ZCR alone is not enough to classify speech, but sudden high ZCR regions may indicate fricatives, noise, or sharper transient sounds.

Spectral centroid:
The centroid is centered around the mid-frequency range, with a median around 1.5 kHz.
This is consistent with human speech.
The variation between min and max likely reflects the alternation between vowels, consonants, and short transitions inside the sentence.

Spectral bandwidth:
The bandwidth is moderate, with a median around 1.7 kHz.
This means the frequency energy is spread around the centroid, but not in a flat/noise-like way.
The p90 around 2.4 kHz indicates that some frames contain wider frequency content, likely consonants or transitions.
A larger bandwidth does not mean “more speech” : it means the frequency content is more spread out.

Spectral flatness:
The flatness values are very low overall.
This is characteristic of structured/tonal content rather than noise.
The higher max and p90 show that a few frames are more noise-like, likely consonants or short transients, but the segment as a whole remains clearly speech-like.

This segment looks like clean speech (from google research):
- strong energy
- low flatness
- moderate ZCR
- mid-range centroid
- moderate bandwidth

This matches the transcript: “Bonjour à tous, la séance est ouverte.” from the first segment of VAD-1000.

---
#### Compared to a silence segment:

```json
{
  "source": {
    "librosa_timeline_json": "librosa_audio_audit/audio_audit_timeline.json"
  },
  "time": {
    "start_seconds": 905.0,
    "end_seconds": 910.0,
    "duration_seconds": 5.0
  },
  "librosa_stats": {
    "rms": {
      "mean": 0.00065077159,
      "median": 0.0,
      "std": 0.0013377767144084635,
      "min": 0.0,
      "max": 0.0056146989,
      "p10": 0.0,
      "p90": 0.0018384830000000001
    },
    "db": {
      "mean": -67.48983421325401,
      "median": -80.0,
      "std": 20.22805398068612,
      "min": -80.0,
      "max": -26.088104248,
      "p10": -80.0,
      "p90": -35.78942680359
    },
    "zcr": {
      "mean": 0.022504882812,
      "median": 0.0,
      "std": 0.0358115831415699,
      "min": 0.0,
      "max": 0.1040039062,
      "p10": 0.0,
      "p90": 0.08107910156
    },
    "spectral_centroid": {
      "mean": 455.319821808844,
      "median": 0.0,
      "std": 782.5480963037254,
      "min": 0.0,
      "max": 3920.9983067775,
      "p10": 0.0,
      "p90": 1416.8542289011802
    },
    "spectral_bandwidth": {
      "mean": 488.033023134712,
      "median": 0.0,
      "std": 760.3655613508956,
      "min": 0.0,
      "max": 2510.9326937179,
      "p10": 0.0,
      "p90": 1632.23991932136
    },
    "spectral_flatness": {
      "mean": 0.724067775294,
      "median": 1.000000596,
      "std": 0.44248360856223334,
      "min": 0.0046127141,
      "max": 1.000000596,
      "p10": 0.01278762901,
      "p90": 1.000000596
    }
```
RMS / dB:
  mostly zero -> silence
  few higher values -> some activity

ZCR:
  mostly zero -> silence
  occasional spikes -> noise / speech fragments

Centroid / Bandwidth:
  mostly zero -> meaningless (no signal)
  occasional values -> background chatter

Flatness:
  mostly ~1 -> noise-like / unstructured

This segment is:
-  mostly silence
- with short noisy / background speech activity

> When dB_median is low, it is mostly safe to ignore spectral features. 

---

Compared to an "applause segment"

```json
{
  "source": {
    "librosa_timeline_json": "librosa_audio_audit/audio_audit_timeline.json"
  },
  "librosa_window_count": 30,
  "librosa_stats": {
    "rms": {
      "mean": 0.03335300118666667,
      "median": 0.03401697055,
      "std": 0.003641199346098424,
      "min": 0.0253222324,
      "max": 0.0384597555,
      "p10": 0.02758992907,
      "p90": 0.03693435785
    },
    "db": {
      "mean": -10.667318280539998,
      "median": -10.4408941269,
      "std": 0.9983116442480041,
      "min": -13.0045948029,
      "max": -9.3745040894,
      "p10": -12.26002941134,
      "p90": -9.726651000939999
    },
    "zcr": {
      "mean": 0.20649414062999996,
      "median": 0.2060546875,
      "std": 0.003926238741906466,
      "min": 0.1997070312,
      "max": 0.2202148438,
      "p10": 0.20229492185,
      "p90": 0.21032714841
    },
    "spectral_centroid": {
      "mean": 2014.7653183450632,
      "median": 2012.4387072506001,
      "std": 47.842148110796856,
      "min": 1872.9549636561,
      "max": 2089.9344072152,
      "p10": 1967.13129340274,
      "p90": 2072.7840161877302
    },
    "spectral_bandwidth": {
      "mean": 1423.1316586764765,
      "median": 1413.4371991988,
      "std": 49.447668957748725,
      "min": 1347.1366057449,
      "max": 1566.1616855177,
      "p10": 1375.8658738767901,
      "p90": 1474.0656611271502
    },
    "spectral_flatness": {
      "mean": 0.043146293860000004,
      "median": 0.04210500235,
      "std": 0.0065450816329186605,
      "min": 0.0340752192,
      "max": 0.0590373762,
      "p10": 0.03575932943,
      "p90": 0.05226679589000001
    }
```
RMS:
Energy is stable and concentrated around the mean/median.
This suggests a fairly constant signal, not a sharp or highly spiky event.

dB:
The signal is moderate-low compared to the loudest point of the file.
Values around -10 dB mean it is clearly audible, but not among the strongest sounds.
The low standard deviation suggests stable loudness.

ZCR:
ZCR is higher than the clean speech segment and very stable.
This suggests a more noise-like or rapidly varying signal than voiced speech.
It should not be interpreted as “constant rhythm”; ZCR is not rhythm, it is zero-crossing activity.

Spectral centroid:
The centroid is very stable around ~2 kHz.
This suggests the frequency “center of mass” is steady.
It is in a range compatible with speech, but that does not mean it is speech.

Spectral bandwidth:
Bandwidth is stable and moderate, around ~1.4 kHz.
This means the frequency energy is spread around the centroid, but not extremely broadly.

Spectral flatness:
Flatness around 0.04 is low, but higher than the clean speech example (~0.01).
This means the signal is still somewhat structured, not pure white noise.
It is more noise-like than clean speech, but not maximally flat/noisy.

---
# Summary

| Feature | **Clean Speech** (The Goal) | **Applause** (The Noise) | **Silence** (The Void) |
|---|---|---|---|
| **RMS / dB** | High & **Dynamic** (High Std) | High & **Static** (Low Std) | Very Low |
| **ZCR** | Moderate (~0.11) | **High & Consistent** (~0.20) | Near Zero |
| **Flatness** | **Very Low** (~0.01) | Low-Moderate (~0.04) | **Very High** (~0.72) |
| **Centroid** | Mid-Range (~1.5kHz) | High-Stable (~2kHz) | Meaningless/Zero |


---

### Last experiment on the longest diarization segment

```json
uv run python librosa-time-range-stats.py --start 11742.8 --end 11955
{
  "source": {
    "librosa_timeline_json": "librosa_audio_audit/audio_audit_timeline.json"
  },
  "time": {
    "start_seconds": 11742.8,
    "end_seconds": 11955.0,
    "duration_seconds": 212.20000000000073
  },
  "librosa_window_count": 2122,
  "librosa_stats": {
    "rms": {
      "mean": 0.041828344549528745,
      "median": 0.045055314900000004,
      "std": 0.016129256560012867,
      "min": 0.000740703,
      "max": 0.0811289623,
      "p10": 0.01728516185000001,
      "p90": 0.059513511500000005
    },
    "db": {
      "mean": -10.089133826210084,
      "median": -7.9997148514,
      "std": 6.555013276126546,
      "min": -43.6817512512,
      "max": -2.8911170959,
      "p10": -16.321419906609993,
      "p90": -5.58232269286
    },
    "zcr": {
      "mean": 0.13053136137040527,
      "median": 0.1042480469,
      "std": 0.06834591351166533,
      "min": 0.0383300781,
      "max": 0.48046875,
      "p10": 0.0646972656,
      "p90": 0.2337890625400001
    },
    "spectral_centroid": {
      "mean": 1625.1659926276438,
      "median": 1430.7012596813,
      "std": 632.1687940200259,
      "min": 643.5489311358,
      "max": 5483.9227076751,
      "p10": 1040.47770770188,
      "p90": 2489.38888354264
    },
    "spectral_bandwidth": {
      "mean": 1586.6241653575773,
      "median": 1505.70079865035,
      "std": 387.94634189604636,
      "min": 808.0718002345,
      "max": 2720.3605262778,
      "p10": 1143.5265203973602,
      "p90": 2164.36753556455
    },
    "spectral_flatness": {
      "mean": 0.023001653570311028,
      "median": 0.00713868345,
      "std": 0.0391045103482458,
      "min": 0.0002468547,
      "max": 0.26361534,
      "p10": 0.0021265466900000003,
      "p90": 0.07095411269000007
    },
    "db_rolling_median": {
      "mean": -8.242076829737654,
      "median": -8.0008387566,
      "std": 1.432644019939389,
      "min": -16.2148799896,
      "max": -5.5989208221,
      "p10": -9.848249149319999,
      "p90": -6.7216863632
    },
    "db_delta": {
      "mean": -1.8470569964746935,
      "median": 0.03128242495,
      "std": 6.1903252248935985,
      "min": -34.867770195,
      "max": 5.9635419846,
      "p10": -7.606515979749999,
      "p90": 2.30567274098
    }
  }
}
```