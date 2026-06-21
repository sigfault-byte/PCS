While analyzing the current _turns_, it became apparent that, within a single speaker intervention, sentence boundaries do not necessarily align with topic boundaries.
For instance, most of the time the speaker starts by thanking the president, then follows with their speech.

# More PCA

## Random sentences

Using the following sentences: 

```python
sentences = [
    "J'aime le pain",
    "J'aime la baguette",
    "J'adore le pain",
    "J'apprécie le pain",
    "Je déteste le pain",
    "Le chat dort sur le canapé",
]
```

Then encoding each of them with:
`model = SentenceTransformer("h4c5/sts-camembert-base")`

And combine them into a 2D PCA:
![[PCA-pain-chat-love-hate.png]]

The resulting graph is surprisingly interpretable, with a variance: `0.6444332 + 0.20011231 = 0.84`

- "Pain" related sentences are on the left side, and the cat sentence far right.
- "j'aime, j'adore, 'j'apprecie le pain" are clustered together in the lower center of the left side
- "j'aime la baguette" is at the top left
- "je deteste le pain" is at the bottom left.

The example is of course based on a very constrained semantic space, adding a sentence like *"The dog sleeps on the car"* drops the variance and flips the PCA.

## Using an actual exemple from the pipeline turns.

Using the following:

```python
raw_text = "Merci madame la présidente, messieurs les ministres, monsieur le rapporteur, monsieur le président de la commission, monsieur le président de la commission spéciale. La simplification de notre vie économique n'est pas un sujet que l'on peut balayer d'un revers de main. Dans le contexte qui est le nôtre, compétitivité sous pression, entrepreneurs qui peinent à se projeter, charges administratives qui pèsent sur nos entreprises, rejeter ce texte sans même le soumettre au débat est un choix politique lourd de sang, chers collègues de gauche. La simplification n'est pas un sujet technique parmi d'autres, elle conditionne directement la capacité de notre pays à créer des emplois, à attirer des investissements et à libérer l'énergie de celles et ceux qui y entreprennent. Voter cette motion, c'est refuser à nos acteurs économiques les réponses qu'ils attendent. Nos entrepreneurs, nos artisans, nos commerçants ont besoin de stabilité, de lisibilité, de prévisibilité. Ils ont besoin de règles claires et proportionnées qui leur permettent de se projeter, d'investir, de recruter, de transmettre. Ce projet de loi est le fruit d'un travail approfondi, conduit depuis plus de deux ans au Sénat, à l'Assemblée nationale et en commission mixte paritaire. Il mérite mieux qu'un rejet sans examen. C'est pourquoi nous voterons contre cette motion de rejet préalable."
```

The question was: if I embed everything at once, much of the detail may be compressed away during embedding because the model will try to generalize over details that are not interesting in semantics as was hypothesized and partially confirmedin the [[Journal_13_PER_extraction_and_exploration|previous journal]].

Splitting on sentence to get granularity when embedding was considered, but immediately discarded because of pronouns. 
If sentence 1 is *The hospital is on fire*, and sentence 2 is *It needs firefighters*, the pronoun _it_ carries very little information when embedded in isolation.

Hence, lets plot the semantic relationship across each sentences:

```python
for i in range(len(sentences) - 1):
    cos = cosine(embeddings[i], embeddings[i + 1])
    cosarr.append(cos)

plt.plot(cosarr)

```

![[rolling_cosine_ex.png]]

The cosine similarity between consecutive sentence embeddings fluctuates throughout the speech.

The plot suggests that the speech could potentially be divided into three semantically coherent subsets by the *breakpoints:*`4` and `8`


```text
1: Merci madame la présidente, messieurs les ministres, monsieur le rapporteur, monsieur le président de la commission, monsieur le président de la commission spéciale. 
2: La simplification de notre vie économique n'est pas un sujet que l'on peut balayer d'un revers de main. 
3: Dans le contexte qui est le nôtre, compétitivité sous pression, entrepreneurs qui peinent à se projeter, charges administratives qui pèsent sur nos entreprises, rejeter ce texte sans même le soumettre au débat est un choix politique lourd de sang, chers collègues de gauche. 

4: La simplification n'est pas un sujet technique parmi d'autres, elle conditionne directement la capacité de notre pays à créer des emplois, à attirer des investissements et à libérer l'énergie de celles et ceux qui y entreprennent.
5: Voter cette motion, c'est refuser à nos acteurs économiques les réponses qu'ils attendent.
6: Nos entrepreneurs, nos artisans, nos commerçants ont besoin de stabilité, de lisibilité, de prévisibilité.
7: Ils ont besoin de règles claires et proportionnées qui leur permettent de se projeter, d'investir, de recruter, de transmettre. 
 
8: Ce projet de loi est le fruit d'un travail approfondi, conduit depuis plus de deux ans au Sénat, à l'Assemblée nationale et en commission mixte paritaire.
9: Il mérite mieux qu'un rejet sans examen. 
10: C'est pourquoi nous voterons contre cette motion de rejet préalable."
```

This could have been either a *lucky* find, especially considering that the splitting strategy is a simple `sentences = [s.strip() for s in raw_text.split(".") if s.strip()]` which would immediately break with `!, ?` or simply `M. Whatever`

But it seems that a *dynamic chunking* of a given document could be made.

# Dirty Ra(g)

Updating the split logic a bit, to `return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]` which is still bad, but enougth for a POC.

8 different *long* speech where randomly chosen in the turns.
Each was split into subsets of sentences, while keeping their *turn_id* origin.

```text
Built 49 chunks from 8 speech segment for 17884 chars
```
The script dynamically splits the long speech into *semantically* related sentences, like in the previous example across the 8 turns segments hopefully allowing the system to retain semantic granularity.

The *drop* was for now randomly chosen to be `-0.1`, meaning that a sufficiently large negative change in cosine similarity triggers the creation of a new chunk.
> This is something that needs, of course more tweaking, but, just like the split is enought for the poc

All embeddings are normalized so that cosine similarity can be computed efficiently through a simple dot product.

> The chunks embedding are filtered again with a simple `if word_count >= 8` to discard every short sentence that would only add noise to the current experiment.

## Results

```text
Built 49 chunks from 8 speech segment for 17884 chars
Query is ''Quelle mesures de sécurité contre la délinquance?'' --- top-k retrived = ''10''
================================================================================
turn=2 chunk=2 score=0.3303
Vous l'avez souligné, il y a énormément d'accidents. Et puis une mesure de police administrative qui sera extrêmement importante.
================================================================================
turn=7 chunk=7 score=0.2831
En tout cas, soyez assurés, madame la députée, que tout est mis en œuvre pour garantir la sécurité à Menton comme partout sur le territoire national.
```

The retrieved segments appear relevant to the query and correspond to the expected topic !

Trying other queries:
```text
Query is ''Pourquoi rejeter la motion?'' --- top-k retrived = ''10''
================================================================================
turn=0 chunk=1 score=0.2933
Ce projet de loi est le fruit d'un travail approfondi, conduit depuis plus de deux ans au Sénat, à l'Assemblée nationale et en commission mixte paritaire. Il mérite mieux qu'un rejet sans examen. C'est pourquoi nous voterons contre cette motion de rejet préalable.
(next top-k are irrelevant)
```

```text
Query is ''Le prix des carburant a t'il été discuté?'' --- top-k retrived = ''10''
================================================================================
turn=3 chunk=4 score=0.4383
La question de ces filières essentielles est donc simple. Allez-vous enfin agir pour faire baisser durablement le prix de l'essence.
```

