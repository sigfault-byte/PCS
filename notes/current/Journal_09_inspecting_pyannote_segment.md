
The pipeline saves all segments embeddings as long as they last more than 0.8s. 
To later recompute embedding centroids from clean segments of identified speakers once speaker identification / attribution has been performed

Therefore, it is important to have a clean diarization chunks.

# Diarization segments

## Duration

Diarization has 1 378 segments.
Covering 15 319.49 seconds ~ 255min ~ 4 hours and 15min. 

```text
count    1343.000000
mean       11.406193
std        20.495105
min         0.016875
25%         2.100938
50%         4.539375
75%        10.665000
max       212.861250
Name: duration, dtype: float64
```
Some needs attentions, the very long one, and some very short one. 
The std shows the distribution is wide, confirm by the mean far from the median.

The missing numbers are the very small segments durations:
```text
missingFrame describe: count    35.000000
mean      0.027964
std       0.016342
min       0.016875
25%       0.016875
50%       0.016875
75%       0.033750
max       0.084375
```

# Db

```text
db_mean
(-200, -100]      0
(-100, -60]       1
(-60, -50]        5
(-50, -40]        5
(-40, -30]      337
(-30, -20]      995
(-20, 0]          0
Name: count, dtype: int64

count    1343.000000
mean      -29.158664
std         2.932987
min       -61.847018
25%       -30.066490
50%       -28.761401
75%       -27.546990
max       -22.654895
Name: db_mean, dtype: float64
```
Only a few segments are below a certain threshold. 
The majority of the segments are within -26dB to -31dB.

# RMS

```text
count    1343.000000
mean        0.040218
std         0.009114
min         0.000959
25%         0.035252
50%         0.040906
75%         0.046165
max         0.075042
Name: rms_mean, dtype: float64
```

Some segments are extremely weak. Likely some soft speech, distance from the microphone or random *speech like* noise.
The shape is good overall, median and quartiles are much higher than the minimum. 

# Spectral Flatness

This is probably the most important data for embedding.
I thought that RCZ was also an important factor, but apparently, not so much.

```text
count    1343.000000
mean        0.027544
std         0.015274
min         0.001111
25%         0.018409
50%         0.024805
75%         0.034413
max         0.176305
Name: flatness_mean, dtype: float64
```
The max and min need some exploration and verification. 
The max might be some applause or crowd noise, typical of french assembly.

The low spectral flatness values suggest that energy is generally concentrated
in structured spectral regions rather than uniformly distributed across frequencies.

```text
count    1321.000000
mean        0.037116
std         0.015788
min         0.000232
25%         0.027616
50%         0.037520
75%         0.046536
max         0.129248
Name: flatness_std, dtype: float64
```
The std confirms the above statement. The acoustic structure is fairly stable across all the segments. 
The segments flatness std shape is concentrated around the median. 

# Overall interpretation

Most diarization segments appear to be:
- stable speech energy
- sufficiently long
- spectrally structured 
- clean of broadbamd noise

So most of them should be exploitables once the identification is made.
