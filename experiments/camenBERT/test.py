import unicodedata

from rapidfuzz import distance, fuzz


def norm(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("-", " ").strip()


def candidate_aliases(first: str, last: str) -> list[str]:

    return [
        f"{first} {last}",
        f"{last}",
        f"{first[0]} {last}" if first else last,
        f"{first} {last.replace('-', ' ')}",
    ]


def best_name_score(raw_name: str, first: str, last: str):
    raw = norm(raw_name)

    aliases = [norm(x) for x in candidate_aliases(first, last)]

    scored = [
        {
            "alias": alias,
            "wratio": fuzz.WRatio(raw, alias),
            "partial": fuzz.partial_ratio(raw, alias),
            "token_sort": fuzz.token_sort_ratio(raw, alias),
            "lev": distance.Levenshtein.distance(raw, alias),
        }
        for alias in aliases
    ]

    return max(scored, key=lambda x: x["wratio"])


pairs = [
    ("maisonneuve", "Nicolas Meizonnet"),
    ("brejon", "Maud Bregeon"),
    ("barraud", "Jean-Noël Barrot"),
    ("jean claudreau", "Jean-Claude Raux"),
    ("elisabeth lucot", "Lisa Belluco"),
    ("Anne Stenbach Terre Noire", "Anne Stambach-Terrenoir"),
]

for raw, truth in pairs:
    a, b = norm(raw), norm(truth)

    print("\n", raw, "<->", truth)
    print("WRatio       :", fuzz.WRatio(a, b))
    print("ratio        :", fuzz.ratio(a, b))
    print("partial      :", fuzz.partial_ratio(a, b))
    print("token_sort   :", fuzz.token_sort_ratio(a, b))
    print("token_set    :", fuzz.token_set_ratio(a, b))
    print("levenshtein  :", distance.Levenshtein.distance(a, b))
