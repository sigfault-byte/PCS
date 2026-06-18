# PER Extraction Cycle

This note explains the PER extraction stage implemented around:

- `src/assemblybot/stages/PER_extraction.py`
- `src/assemblybot/stages/per_identity.py`
- `src/assemblybot/stages/per_analysis.py`
- `src/assemblybot/models/turn_document.py`

Current observed result on extracted names: 69% recall and 100% precision.

## Goal

The stage enriches an existing `TurnDocument` JSON.

It keeps the original `turns` unchanged and rebuilds `turns_analysis` with one
`TurnAnalysis` per turn. The important fields for this stage are:

- `current_speaker`: the inferred real person currently speaking in that turn.
- `current_speaker_source`: how the speaker identity was inferred.
- `current_speaker_purity`: how dominant that identity is for the pyannote speaker id.
- `mentioned_persons`: the PER names spoken inside that turn text.

The key distinction is:

- `mentioned_persons` means names that appear in the text.
- `current_speaker` means the person inferred to be speaking.

A dropped name does not become `current_speaker` by itself.

## Inputs

The CLI stage requires:

- `--input-json`: existing `TurnDocument` JSON.
- `--csv-ground-truth-PER`: deputy ground truth CSV.
- `--csv-ground-truth-ministers`: minister and Assembly chair ground truth CSV.

It optionally accepts:

- `--output-json`: explicit output path.

If no output path is provided, the stage writes a default `_02_per_extraction.json`
file beside the pipeline interim outputs.

## Ground Truth Loading

Ground truth is loaded in `per_identity.py`.

Both CSV files use the same exact headers:

- `identifiant`
- `Prénom`
- `Nom`
- `Région`
- `Département`
- `Numéro de circonscription`
- `Profession`
- `Groupe politique (complet)`
- `Groupe politique (abrégé)`

Rows are validated strictly. A malformed header or row with the wrong column
count raises `ValueError` with the file path and line information.

Each CSV row becomes a `KnownPerson`, wrapping a `PersonIdentity`:

```python
PersonIdentity(
    id=str | None,
    name=str,
    role=str | None,
    kind=str,
)
```

Kinds are assigned as follows:

- deputy CSV rows become `kind="deputy"` and `role="Député"`.
- `identifiant` starting with `minister:` becomes `kind="minister"`.
- `identifiant` starting with `assembly_chair:`, `assembly_chari:`, or
  `assembly_chaii:` becomes `kind="assembly_chair"`.
- unresolved NER names become `kind="raw_per"`, `id=None`, `role=None`.

Matched people keep the display name from the CSV, including accents. Raw PER
fallbacks store the normalized NER text.

## NER Filtering

The stage uses:

```python
NER_MODEL_NAME = "Jean-Baptiste/camembert-ner"
```

with:

```python
aggregation_strategy="simple"
```

`collect_person_mentions(...)` keeps only NER entities where:

- `entity_group == "PER"`
- `score > PER_CONFIDENCE_THRESHOLD`

The current threshold is:

```python
PER_CONFIDENCE_THRESHOLD = 0.8
```

So score `0.8` exactly is rejected, because the code skips entities with
`score <= threshold`.

Generic titles such as `monsieur le ministre` or `madame la presidente` are
filtered before mention handling. This prevents title-only PER spans from
becoming identities.

## Mention Resolution

Every kept PER mention is resolved against the merged deputy, minister, and
Assembly chair ground truth list.

The resolver:

1. Normalizes names by lowercasing, removing accents, trimming, and collapsing
   whitespace.
2. Refuses fuzzy matching for single-token PER mentions.
3. Allows a single-token match only if it exactly equals a full known normalized
   name.
4. Uses `rapidfuzz.WRatio` for multi-token mentions.
5. Requires the full-name fuzzy score to pass:

```python
FUZZY_MATCH_THRESHOLD = 80
```

6. Requires token overlap:

```python
TOKEN_MATCH_THRESHOLD = 65
```

For multi-token names, at least one token must match exactly, and every query
token must be close enough to a candidate token. This keeps cases like
`bernard chex` resolving to `Bernard Chaix`, while `fleuristes` cannot resolve
to `Sandrine Le Feur` because it is a single token.

## Mentioned Persons

`mentioned_persons` is built for every turn from all kept PER mentions in that
turn.

These mentions are:

- resolved to structured `PersonIdentity` objects;
- deduplicated per turn by `(kind, id, name)`;
- stored in first-seen order.

This field is descriptive only. It says whose name was spoken, not who is
speaking.

## Current Speaker Inference

`current_speaker` is inferred only from president or chair handoff patterns.

The active next-speaker patterns are:

```python
NEXT_SPEAKER_PATTERNS = [
    "la parole est à",
    "je donne la parole à",
    "vous avez la parole",
]
```

Previous-speaker `merci` attribution is intentionally disabled:

```python
PREVIOUS_SPEAKER_PATTERNS = []
```

This was changed because `merci` created too much ambiguity. For example, a
chair turn can thank someone and then mention another person in the next
sentence. That should not identify the previous diarized speaker.

## Sentence-Bounded Anchors

Handoff pattern detection is sentence bounded.

For each PER entity, the code finds the local sentence-like span by stopping at:

```python
SENTENCE_BOUNDARY_CHARS = ".?!"
```

The handoff pattern must appear before the PER mention in that same span.

This means:

- `La parole est à Monsieur X.` can infer that `X` is the next speaker.
- `Merci madame la presidente. Monsieur X...` does not infer `X`.

## Mixed PER Handling

If several PER entities appear after the same next-speaker anchor in the same
sentence, only the first PER after that anchor is allowed to predict the next
speaker.

This is a precision-first rule. The first name after `la parole est à` is the
best candidate for the speaker being introduced. Later names in the same turn
can still be stored in `mentioned_persons`, but they do not become
`current_speaker`.

## Pyannote Speaker Propagation

The stage uses pyannote diarization speaker ids from the `Turn.speaker_id`
field to propagate identity.

The cycle is:

1. A chair turn says `la parole est à X`.
2. The PER mention `X` is resolved to a `PersonIdentity`.
3. `find_predicted_turn_id(...)` searches forward for the next turn whose
   `speaker_id` differs from the chair turn's `speaker_id`.
4. That predicted turn's `speaker_id` is associated with the resolved identity.
5. Every turn with the same `speaker_id` receives that identity as
   `current_speaker`.

This is the core trick: the president identifies one turn, and pyannote carries
that identity across all turns with the same diarized speaker label.

## Purity

Speaker identity votes are collected in a speaker/person matrix.

The current active vote source is the next-speaker anchor:

```python
NEXT_SPEAKER_WEIGHT = 2
```

For each pyannote `speaker_id`, the stage chooses the most common resolved
person. It then computes:

```text
current_speaker_purity = best_weighted_count / total_weighted_count
```

High purity means the diarized speaker id consistently maps to the same person.
Lower purity means the same pyannote speaker id received conflicting identity
evidence, which can indicate diarization merging, bad NER, bad fuzzy resolution,
or a wrong turn prediction.

## Assembly Chair Hardcoding

Chair turns are assigned directly to:

```python
PersonIdentity(
    id="assembly_chair:yael-braun-pivet",
    name="Yaël Braun-Pivet",
    role="Présidente de l'Assemblée nationale",
    kind="assembly_chair",
)
```

This applies when the turn text looks like an Assembly chair turn, including
handoff phrases such as `la parole est à`.

So a chair handoff turn can do both things:

1. Its own `current_speaker` is the Assembly chair.
2. The introduced PER name identifies the next diarized speaker.

## Output Shape

The output `TurnDocument` preserves:

- `turns`

and replaces:

- `turns_analysis`

with one `TurnAnalysis` per turn.

For a resolved speaker turn:

```json
{
  "turn_id": 52,
  "current_speaker": {
    "id": "720614",
    "name": "Marine Le Pen",
    "role": "Député",
    "kind": "deputy"
  },
  "current_speaker_source": "inferred_from_next_speaker",
  "current_speaker_purity": 1.0,
  "mentioned_persons": []
}
```

For a turn with only dropped names:

```json
{
  "turn_id": 51,
  "current_speaker": {
    "id": "assembly_chair:yael-braun-pivet",
    "name": "Yaël Braun-Pivet",
    "role": "Présidente de l'Assemblée nationale",
    "kind": "assembly_chair"
  },
  "current_speaker_source": "hardcoded_assembly_chair",
  "current_speaker_purity": 1.0,
  "mentioned_persons": [
    {
      "id": "605782",
      "name": "Laurent Marcangeli",
      "role": "Député",
      "kind": "deputy"
    }
  ]
}
```

## Verification

The focused verification commands are:

```bash
env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 uv run python -m py_compile \
  src/assemblybot/stages/PER_extraction.py \
  src/assemblybot/stages/per_identity.py \
  src/assemblybot/stages/per_analysis.py \
  src/assemblybot/models/turn_document.py
```

```bash
env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 uv run python -m unittest \
  tests.test_per_identity \
  tests.test_per_analysis \
  tests.test_per_extraction
```

## Precision-First Design Choices

The current implementation intentionally favors precision over recall:

- only high-confidence PER entities are kept;
- generic titles are filtered;
- single-token fuzzy matching is blocked;
- `merci` attribution is disabled;
- anchors are sentence bounded;
- only the first PER after a handoff anchor can identify the next speaker;
- identities are propagated through pyannote `speaker_id`;
- purity is exposed to inspect conflicting identity evidence.

This explains the current behavior: fewer identities are inferred, but the
ones that are inferred are much more trustworthy.
