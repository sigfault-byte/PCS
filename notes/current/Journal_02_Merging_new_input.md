> Following [[Journal_01_Merging|last journal]]

> A longer session of the french *assemblée nationale* of 5 hours and 20 minutes was downloaded.
> id: *1ere-seance--questions-au-gouvernement--simplification-de-la-vie-economique-cmp--renforcer-la-s-14-avril-2026*
> An XML version of the official *compte rendu* was also available and was initially considered as ground truth for comparison.  
> However, it quickly became clear that the *compte rendu* is not a verbatim transcript and therefore cannot be treated as strict ASR ground truth.
> It is better understood as institutional ground truth: a validated political record rather than an acoustic transcript.

> The current document tries to analyze the differences between the current pipeline (2026-04-27: Whisper + Pyannote + timestamp-and-heuristic merging) and the official *"compte rendu"* of the file.

> "*Compte rendu"* : [[human_official]]
> *"verbatim transcript from Whisper"*: [[human_merged]] 

### Observation about the *compte rendu*
The name itself expresses that it will not be a verbatim transcript.

### 1. Example of textual error: 
> Each example structure is `[Speaker][Timestamp in second] : Text` 

```text
[M. Laurent Lhardit (SOC)][9301.50] : ... qui le situe parfois aux confins des univers d’ et des , comme avec le fameux ...
```
*From "compte rendu"*

The *"univers d’ et"* error is particularly interesting because it appears more consistent with structured text loss than with deliberate human writing.
The sentence itself becomes semantically broken.

Whisper equivalent segment on the same part:
```text
[SPEAKER_55][9299.29 -> 9409.92]: ... aussi il le situe parfois aux confins des univers de huburois et des shadocs comme avec ce fameux ...
```

Whisper also failed to analyze what was said but the verbatim sentence is *"aux confins des univers de Ubu Roi et des Shadoks"*.

It is a bit disorienting that the official compte rendu has such an error. 
Because it is not a verbatim transcription but rather an editorially normalized institutional record, one would expect fewer textual corruption errors from it.
This makes such errors particularly surprising.

> off topic: the meaning is basically: *"this bill belongs somewhere between grotesque absurd tyranny and ridiculous nonsensical bureaucracy"*
> from : *Ubu Roi* (*“King Ubu”*) a famous absurd French play by Alfred Jarry (1896)
> and: *Les Shadoks* an old French absurd animated TV show.

### 2. Example of editorial rewrite and inconsistency

Whisper:
```text
[SPEAKER_25][1230.12 -> 1356.21] : Merci beaucoup madame la présidente, mesdames et messieurs les députés. Monsieur le député, vous avez raison de souligner le fléau que représente le protoxyde d'azote et sa consommation, notamment par les plus jeunes. Vous le savez, c'est un problème que nous avons pris à bras le corps. D'abord, nous avons introduit un certain nombre de mesures dans le projet de loi Riposte qui viendra bientôt en discussion parlementaire. ...
```

Compte rendu:
```text
[M. Laurent Nuñez][1234.56] : Vous avez raison de souligner que le protoxyde d’azote et sa consommation, notamment par les plus jeunes, représentent un vrai fléau. Nous avons pris le problème à bras-le-corps. Le projet de loi dit Ripost, qui sera débattu bientôt, ... 
```

#### 2.1 editorial rewrite 
For this part, the editorial rewrite is meaningless, but the deeper issue is not the presence of editorial rewrite itself, but its inconsistency.
Some expressions are normalized and protocol phrases are removed, while other oral figures of speech and rhetorical formulations are preserved.
The document oscillates between institutional normalization and near-verbatim transcription.
This inconsistency makes the *compte rendu* difficult to use as a stable evaluation target.

It is simplifying the speech of the first minister, while keeping the meaning, but keeps the figure of speech *"à bras-le-corps"*, which adds no value to the *"compte rendu"* as it is an oral expression used in the current speech.

#### 2.2 inconsistency
In the previous text, the *"merci"* addressed to the *president of the assembly* is stripped.
It would not be a problem if it was consistent, but some *merci* are stripped, some are not.
They are mainly stripped, but it is inconsistent.

It is both *verbatim* and not verbatim, both filtering noise, and not filtering noise...

---
# Interpretation

The most important consequence is that the XML cannot be treated as strict ASR ground truth.

There are at least three distinct notions of truth:
- acoustic truth: what was physically spoken in the room (Whisper target)
- speaker truth: who said what and when (diarization target)
- institutional truth: what the Assembly wants preserved in the official political record (*compte rendu* target)

These are not interchangeable and should not be evaluated using the same criteria.

#### Overall this gives more categories to investigate:

- Whisper errors 
	- Some hallucinations were filtered out, but some are still present. I am not sure yet how to extract the highest-signal cases; I may need to add Whisper confidence values as a filtering signal
- compte rendu editorial rewrite
	- I am unsure how to interpret this. Sometimes it is verbatim, or tries to be, sometimes it is a pure rewrite. 
- compte rendu omission 
	- this can be problematic, reviewing manually some segments on the same time frame, the omissions are mostly filling words, but some of them are kept.
- compte rendu textual error 
	- this is mostly benign, but the kind of error encountered indicates that the compte rendu could be part *machine generated*
- compte rendu factual error - problematic -
	- this could be problematic. So far, by randomly selecting segments with matching timestamps, I have not found significant factual drift.
- semantic equivalence
    - exact lexical match is rare, but semantic equivalence is often preserved.
    - the same political meaning may be retained even when the sentence is heavily rewritten.

> A stronger evaluation may be token or phrase-level alignment rather than segment-to-segment comparison.

> Segment boundaries are unstable because of editorial rewriting, but token and phrase-level comparison could better measure semantic preservation and identify where institutional rewriting modifies, preserves, or removes meaning.

