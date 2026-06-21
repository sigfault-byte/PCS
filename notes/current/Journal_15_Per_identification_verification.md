The pipeline now identifies who is speaking, but also their roles.

The `turn document` gives:
```bash
jq '[.turns[]] | length' data/interim/1ere-seance--questions-au-gouvernement--simplification-de-la-vie-economique-cmp--renforcer-la-s-14-avril-2026_01_turns.json
213

jq '[.turns_analysis[]] | length' data/interim/1ere-seance--questions-au-gouvernement--simplification-de-la-vie-economique-cmp--renforcer-la-s-14-avril-2026_01_turns_02_per_extraction.json
213
```

Example with index `42`:
```json
 jq '[.turns[42]]' data/interim/1ere-seance--questions-au-gouvernement--simplification-de-la-vie-economique-cmp--renforcer-la-s-14-avril-2026_01_turns.json
[
  {
    "turn_id": 43,
    "audio_time": {
      "start_seconds": 3523.02471875,
      "end_seconds": 3643.1072187500004,
      "duration_seconds": 120.08250000000044,
      "start_ts": "00:58:43.02",
      "end_ts": "01:00:43.11"
    },
    "text": "Merci Présidente. Monsieur le Premier Ministre, 100 sites en 10 minutes. Le 8 avril au Liban, Israël a bombardé 100 sites en 10 minutes, quelques heures à peine après l'annonce d'un cessez-le-feu avec l'Iran. Plus de 300 personnes ont été assassinées en une seule journée. Au Liban, Israël a déjà ôté la vie à plus de 2000 personnes. 600 enfants ont été tués ou blessés. A Gaza, le génocide se poursuit. En Cisjordanie, le nettoyage ethnique se poursuit. 34 nouvelles colonies seront implantées et s'ajouteront aux 68 déjà établies. Par le gouvernement Netanyahou depuis 2022. L'action meurtrière d'Israël suit implacablement son cours. Face à elle, seules deux attitudes sont possibles pour un pays comme le nôtre. Le refus de l'impunité ou la complicité. Le refus, c'est réclamer la suspension de l'accord d'association entre l'UE et Israël comme l'a fait l'Espagne. Notre initiative citoyenne européenne a d'ailleurs atteint cette nuit le million de signatures pour contraindre la Commission européenne à suspendre cet accord. Un record de rapidité pour une telle initiative sans aucun soutien et relais médiatique. Mais votre gouvernement a choisi le camp de la complicité. Absence de sanctions, livraison d'armes à Israël ou encore citations racistes quand M. Barrault fait sien les propos génocidaires de Golda Meir. Même l'Italie de Mme Meloni a annoncé suspendre son accord de défense avec Israël. Pire encore, vous maintenez à l'ordre du jour la loi Yadan qui essentialise nos compatriotes juifs pour vous en prendre à ceux qui dénoncent un génocide plutôt qu'à ceux qui le commettent. Cinq rapporteurs des Nations Unies vous appellent à retirer ce texte. Comptez-vous donner suite à leur demande ? Car en réalité, le vote de cette loi ne serait qu'un cadeau de plus au gouvernement suprémaciste de M. Netanyahou. M. le Premier ministre, un État qui a violé 1201 résolutions des Nations Unies depuis 1947, qui extermine des civils dont des milliers d'enfants, tue des casques bleus, installe des colonies illégales, annexe ses voisins, instaure la peine de mort raciste. Peut-il être un allié de la France ?",
    "speaker_id": "SPEAKER_12",
    "speaker_confidence": 1.0,
    "transcript_segment_ids": [
      635,
      636,
      637,
      638,
      639,
      640,
      641,
      642,
      643,
      644,
      645,
      646,
      647,
      648,
      649,
      650,
      651,
      652,
      653,
      654,
      655,
      656,
      657,
      658,
      659,
      660,
      661
    ],
    "diarization_segment_ids": [
      161,
      162
    ],
    "flags": 2048
  }
]
```

Corresponding turn_analysis:
```json
jq '[.turns_analysis[42]] ' data/interim/1ere-seance--questions-au-gouvernement--simplification-de-la-vie-economique-cmp--renforcer-la-s-14-avril-2026_01_turns_02_per_extraction.json
[
  {
    "turn_id": 43,
    "keywords": [],
    "current_speaker": {
      "id": "842187",
      "name": "Gabrielle Cathala",
      "role": "Député",
      "kind": "deputy"
    },
    "current_speaker_source": "inferred_from_next_speaker",
    "current_speaker_purity": 1.0,
    "mentioned_persons": [
      {
        "id": null,
        "name": "merci presidente",
        "role": null,
        "kind": "raw_per"
      },
      {
        "id": null,
        "name": "monsieur le premier ministre",
        "role": null,
        "kind": "raw_per"
      },
      {
        "id": null,
        "name": "netanyahou",
        "role": null,
        "kind": "raw_per"
      },
      {
        "id": null,
        "name": "m. barrault",
        "role": null,
        "kind": "raw_per"
      },
      {
        "id": null,
        "name": "golda meir",
        "role": null,
        "kind": "raw_per"
      },
      {
        "id": null,
        "name": "mme meloni",
        "role": null,
        "kind": "raw_per"
      },
      {
        "id": null,
        "name": ". netanyahou",
        "role": null,
        "kind": "raw_per"
      }
    ],
    "organizations": [],
    "locations": [],
    "embedding_id": null,
    "audio_audit": null
  }
]
```

Because ground truth does not exist, a manual verification is needed.
Therefore, a manual audit of 50 turns of `current_speake != null` will be make to evaluate a precision proxy.

# Verification

### Recall

Overall recall is:
```bash
jq '[.turns_analysis[] | select(.current_speaker != null)] | length ' data/interim/1ere-seance--questions-au-gouvernement--simplification-de-la-vie-economique-cmp--renforcer-la-s-14-avril-2026_01_turns_02_per_extraction.json
144
```

144/213~0.67
67% recall

### Precision

#### Minister

```json
jq '
[.turns_analysis[] | select(.current_speaker.kind == "minister")
  | {
      turn_id,
      name: .current_speaker.name,
      id: .current_speaker.id,
      role: .current_speaker.role
    }
]
' data/interim/1ere-seance--questions-au-gouvernement--simplification-de-la-vie-economique-cmp--renforcer-la-s-14-avril-2026_01_turns_02_per_extraction.json
[
  {
    "turn_id": 6,
    "name": "Laurent Nunez",
    "id": "minister:laurent_nunez",
    "role": "Ministre de l'interieur"
  },
  {
    "turn_id": 14,
    "name": "Jean-Pierre Farandou",
    "id": "minister:jean-pierre-farandou",
    "role": "Ministre du Travail et des Solidarités"
  },
  {
    "turn_id": 18,
    "name": "Jean-Pierre Farandou",
    "id": "minister:jean-pierre-farandou",
    "role": "Ministre du Travail et des Solidarités"
  },
  {
    "turn_id": 28,
    "name": "Laurent Nunez",
    "id": "minister:laurent_nunez",
    "role": "Ministre de l'interieur"
  },
  {
    "turn_id": 33,
    "name": "Édouard Geffray",
    "id": "minister:edouard-geffray",
    "role": "Ministre de l’Éducation nationale"
  },
  {
    "turn_id": 37,
    "name": "Stéphanie Rist",
    "id": "minister:stephanie-rist",
    "role": "Ministre déléguée auprès du ministre de l’Europe et des Affaires étrangères, chargée de la Francophonie, des Partenariats internationaux et des Français de l’étranger"
  },
  {
    "turn_id": 41,
    "name": "Jean-Pierre Farandou",
    "id": "minister:jean-pierre-farandou",
    "role": "Ministre du Travail et des Solidarités"
  },
  {
    "turn_id": 46,
    "name": "Aurore Bergé",
    "id": "minister:aurore_berge",
    "role": "Ministre déléguée auprès du Premier ministre, chargée de l'Égalité entre les femmes et les hommes et de la Lutte contre les discriminations"
  },
  {
    "turn_id": 50,
    "name": "David Amiel",
    "id": "minister:david-amiel",
    "role": "Ministre de l’Action et des Comptes publics"
  },
  {
    "turn_id": 58,
    "name": "Édouard Geffray",
    "id": "minister:edouard-geffray",
    "role": "Ministre de l’Éducation nationale"
  },
  {
    "turn_id": 62,
    "name": "Édouard Geffray",
    "id": "minister:edouard-geffray",
    "role": "Ministre de l’Éducation nationale"
  },
  {
    "turn_id": 66,
    "name": "Stéphanie Rist",
    "id": "minister:stephanie-rist",
    "role": "Ministre déléguée auprès du ministre de l’Europe et des Affaires étrangères, chargée de la Francophonie, des Partenariats internationaux et des Français de l’étranger"
  },
  {
    "turn_id": 70,
    "name": "Édouard Geffray",
    "id": "minister:edouard-geffray",
    "role": "Ministre de l’Éducation nationale"
  },
  {
    "turn_id": 74,
    "name": "Aurore Bergé",
    "id": "minister:aurore_berge",
    "role": "Ministre déléguée auprès du Premier ministre, chargée de l'Égalité entre les femmes et les hommes et de la Lutte contre les discriminations"
  },
  {
    "turn_id": 82,
    "name": "Serge Papin",
    "id": "minister:serge-papin",
    "role": "Ministre des Petites et moyennes entreprises, du Commerce, de l’Artisanat, du Tourisme et du Pouvoir d’achat"
  },
  {
    "turn_id": 84,
    "name": "Mathieu Lefèvre",
    "id": "minister:mathieu-lefevre",
    "role": "Ministre délégué auprès de la ministre de la Transition écologique, de la Biodiversité et des Négociations internationales sur le climat et la nature, chargé de la Transition écologique"
  },
  {
    "turn_id": 141,
    "name": "Mathieu Lefèvre",
    "id": "minister:mathieu-lefevre",
    "role": "Ministre délégué auprès de la ministre de la Transition écologique, de la Biodiversité et des Négociations internationales sur le climat et la nature, chargé de la Transition écologique"
  },
  {
    "turn_id": 143,
    "name": "Mathieu Lefèvre",
    "id": "minister:mathieu-lefevre",
    "role": "Ministre délégué auprès de la ministre de la Transition écologique, de la Biodiversité et des Négociations internationales sur le climat et la nature, chargé de la Transition écologique"
  },
  {
    "turn_id": 145,
    "name": "Mathieu Lefèvre",
    "id": "minister:mathieu-lefevre",
    "role": "Ministre délégué auprès de la ministre de la Transition écologique, de la Biodiversité et des Négociations internationales sur le climat et la nature, chargé de la Transition écologique"
  },
  {
    "turn_id": 183,
    "name": "Laurent Nunez",
    "id": "minister:laurent_nunez",
    "role": "Ministre de l'interieur"
  },
  {
    "turn_id": 205,
    "name": "Laurent Nunez",
    "id": "minister:laurent_nunez",
    "role": "Ministre de l'interieur"
  }
]
```
6 , 14, 18 , 28, 33, 37, 41, 46, 50, 58, 62, 66, 70, 74, 82, 84, -> correct, from PER extractaion / logic
141 , 143, 145, 183, 205->  Propagated from the speakerID of pyannote when purity is > 80%

### Deputies

```json
[
  {
    "turn_id": 12,
    "name": "Nicole Dubré-Chirat",
    "id": "720154",
    "role": "Député"
  },
  {
    "turn_id": 16,
    "name": "Hadrien Clouet",
    "id": "793736",
    "role": "Député"
  },
  {
    "turn_id": 20,
    "name": "Hadrien Clouet",
    "id": "793736",
    "role": "Député"
  },
  {
    "turn_id": 22,
    "name": "Jean-Paul Lecoq",
    "id": "335612",
    "role": "Député"
  },
  {
    "turn_id": 39,
    "name": "Bernard Chaix",
    "id": "840701",
    "role": "Député"
  },
  {
    "turn_id": 43,
    "name": "Gabrielle Cathala",
    "id": "842187",
    "role": "Député"
  },
  {
    "turn_id": 48,
    "name": "Stéphane Viry",
    "id": "721474",
    "role": "Député"
  },
  {
    "turn_id": 52,
    "name": "Laurent Marcangeli",
    "id": "605782",
    "role": "Député"
  },
  {
    "turn_id": 60,
    "name": "Josiane Corneloup",
    "id": "722390",
    "role": "Député"
  },
  {
    "turn_id": 64,
    "name": "Denis Fégné",
    "id": "841665",
    "role": "Député"
  },
  {
    "turn_id": 68,
    "name": "Catherine Ibled",
    "id": "795954",
    "role": "Député"
  },
  {
    "turn_id": 72,
    "name": "Sandra Regol",
    "id": "794778",
    "role": "Député"
  },
  {
    "turn_id": 75,
    "name": "Sandra Regol",
    "id": "794778",
    "role": "Député"
  },
  {
    "turn_id": 78,
    "name": "Christophe Naegelen",
    "id": "721486",
    "role": "Député"
  },
  {
    "turn_id": 80,
    "name": "Stéphane Travert",
    "id": "607395",
    "role": "Député"
  },
  {
    "turn_id": 86,
    "name": "Gérard Leseul",
    "id": "774958",
    "role": "Député"
  },
  {
    "turn_id": 92,
    "name": "Olivier Fayssat",
    "id": "840773",
    "role": "Député"
  },
  {
    "turn_id": 94,
    "name": "Matthias Renault",
    "id": "841955",
    "role": "Député"
  },
  {
    "turn_id": 96,
    "name": "Marie Lebec",
    "id": "721852",
    "role": "Député"
  },
  {
    "turn_id": 98,
    "name": "Claire Lejeune",
    "id": "842029",
    "role": "Député"
  },
  {
    "turn_id": 100,
    "name": "Laurent Lhardit",
    "id": "840765",
    "role": "Député"
  },
  {
    "turn_id": 104,
    "name": "Geneviève Darrieussecq",
    "id": "719914",
    "role": "Député"
  },
  {
    "turn_id": 106,
    "name": "Thomas Lam",
    "id": "842125",
    "role": "Député"
  },
  {
    "turn_id": 114,
    "name": "Olivier Fayssat",
    "id": "840773",
    "role": "Député"
  },
  {
    "turn_id": 116,
    "name": "Pierre Meurin",
    "id": "793852",
    "role": "Député"
  },
  {
    "turn_id": 118,
    "name": "Marie Lebec",
    "id": "721852",
    "role": "Député"
  },
  {
    "turn_id": 122,
    "name": "Laurent Lhardit",
    "id": "840765",
    "role": "Député"
  },
  {
    "turn_id": 130,
    "name": "Thomas Lam",
    "id": "842125",
    "role": "Député"
  },
  {
    "turn_id": 133,
    "name": "Laurent Mazaury",
    "id": "841931",
    "role": "Député"
  },
  {
    "turn_id": 135,
    "name": "Laurent Mazaury",
    "id": "841931",
    "role": "Député"
  },
  {
    "turn_id": 137,
    "name": "Christophe Naegelen",
    "id": "721486",
    "role": "Député"
  },
  {
    "turn_id": 139,
    "name": "Stéphane Travert",
    "id": "607395",
    "role": "Député"
  },
  {
    "turn_id": 147,
    "name": "Stéphane Travert",
    "id": "607395",
    "role": "Député"
  },
  {
    "turn_id": 149,
    "name": "Pierre Meurin",
    "id": "793852",
    "role": "Député"
  },
  {
    "turn_id": 155,
    "name": "Gérard Leseul",
    "id": "774958",
    "role": "Député"
  },
  {
    "turn_id": 164,
    "name": "Stéphane Travert",
    "id": "607395",
    "role": "Député"
  },
  {
    "turn_id": 171,
    "name": "Laurent Mazaury",
    "id": "841931",
    "role": "Député"
  },
  {
    "turn_id": 173,
    "name": "Elsa Faucillon",
    "id": "721896",
    "role": "Député"
  },
  {
    "turn_id": 175,
    "name": "Olivier Fayssat",
    "id": "840773",
    "role": "Député"
  },
  {
    "turn_id": 177,
    "name": "Michaël Taverne",
    "id": "794502",
    "role": "Député"
  },
  {
    "turn_id": 179,
    "name": "Stella Dupont",
    "id": "643175",
    "role": "Député"
  },
  {
    "turn_id": 197,
    "name": "Elsa Faucillon",
    "id": "721896",
    "role": "Député"
  }
]
```


12, 16, 20, 22, 39, 43, 48, 52, 60, 64, 68, 72, 78, 80, 86, 92, 94, 96,  98, 100, 103, 106, 144, 116, 118, 122, 130, 133, 135, 149, 155, 171, 173, 175, 179, 197 -> correct from PER extraction
75, 137, 139, 147, 164 -> confusion matrix propagation 

### Assembly Chair:

68 turns:
1 , 3, 5, 7, 9, 11, 13, 15, 17, 21, 23, 25, 27, 30, 32, 34, 36, 38, 40, 42, 44, 47, 49, 51, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 77, 79, 81, 83, 85, 91, 93, 95, 97, 99, 101, 103, 105, 107, 113, 115, 117, 119, 121, 123, 129, 131, 142, 154, 160, 165, 168, 170, 172, 174, 176, 178, 212, 213
Are all correctly attributed to her.


### Identified person without ground truth

13 *raw person* are detected:
```json
[
  {
    "turn_id": 8,
    "name": "nicolas maisonneuve",
    "id": null,
    "role": null
  },
  {
    "turn_id": 10,
    "name": "maude brejon",
    "id": null,
    "role": null
  },
  {
    "turn_id": 24,
    "name": "jean-noel barraud",
    "id": null,
    "role": null
  },
  {
    "turn_id": 26,
    "name": "alexandra masson",
    "id": null,
    "role": null
  },
  {
    "turn_id": 29,
    "name": "alexandra masson",
    "id": null,
    "role": null
  },
  {
    "turn_id": 35,
    "name": "jean claudreau",
    "id": null,
    "role": null
  },
  {
    "turn_id": 56,
    "name": "aurelien lopez ligaurie",
    "id": null,
    "role": null
  },
  {
    "turn_id": 88,
    "name": "yann boucard",
    "id": null,
    "role": null
  },
  {
    "turn_id": 102,
    "name": "elisabeth lucot",
    "id": null,
    "role": null
  },
  {
    "turn_id": 120,
    "name": "anne stenbach",
    "id": null,
    "role": null
  },
  {
    "turn_id": 124,
    "name": "yann boucard",
    "id": null,
    "role": null
  },
  {
    "turn_id": 126,
    "name": "elisabeth lucot",
    "id": null,
    "role": null
  },
  {
    "turn_id": 151,
    "name": "elisabeth lucot",
    "id": null,
    "role": null
  }
]
``` 

8 -> The correct name from the list of deputee is 
`"719436","Nicolas","Meizonnet","Occitanie","Gard","2","Ingénieur en informatique","Rassemblement National","RN"`
maisonneuve / Meizonnet -> 6 levenstein distance

10 -> The correct minister from the list is : `"minister:maud_bregeon","Maud","Bregeon","","","","Ministre déléguée, porte-parole du Gouvernement auprès du Premier ministre et ministre déléguée, chargée de l'Énergie auprès du ministre de l'Économie, des Finances et de la Souveraineté industrielle, énergétique et numérique","Gouvernement","GOUV"`

brejon / Bregeon -> 3 

24 -> Correct name in the list is: `"minister:jean-noel-barrot","Jean-Noël","Barrot","","","","Ministre de l’Éducation nationale","Gouvernement","GOUV"`

barraud / barrot -> 4

26, 29 -> The name is correct, she is not in the deputy list.  As of today *june 2026* she is the mayor of Menton. 
The deputee list used for cross checking is also june 2026.

35 -> Correct name from the deputee is `"794154","Jean-Claude","Raux","Pays de la Loire","Loire-Atlantique","6","Professeur, profession scientifique","Écologiste et Social","EcoS"`

jean claudreau / Jean-Claude Raux -> 7 

56 -> not in the deputy list. He was a deputee at the time of the session

88, 124 -> Correct name is `"721816","Ian","Boucard","Bourgogne-Franche-Comté","Territoire de Belfort","1","Autre cadre (secteur privé)","Droite Républicaine","DR"`

102, 126, 151 -> `"795362","Lisa","Belluco","Nouvelle-Aquitaine","Vienne","1","Cadre de la fonction publique","Écologiste et Social","EcoS"`

elisabeth lucot / lisa belluco -> 6

119 -> whispere transcripted *Anne Stenbach, Terre Noire* and the correct name is : `"793744","Anne","Stambach-Terrenoir","Occitanie","Haute-Garonne","2","Professeure de piano","La France insoumise - Nouveau Front Populaire","LFI-NFP"`

# Conclusion

Speaker identification results

- 212 turns processed
- 144 turns assigned to a named speaker (68%)
- 131 manually verified assignments
- 0 incorrect assignments observed during manual review

Known causes of unassigned speakers:
- NER misses
- Historical office holders absent from current reference datasets
- Whisper transcription errors
- Ambiguous or incomplete person mentions

The system is intentionally tuned for precision over recall, and it worked.
