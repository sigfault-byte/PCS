import json

from bs4 import BeautifulSoup

with open("deputes.html", "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

rows = soup.select("table.deputes tbody tr")

data = []

for row in rows:
    cols = row.find_all("td")
    if len(cols) < 3:
        continue

    name_cell = cols[0]

    raw_name = name_cell.get_text(" ", strip=True)
    sort_name = name_cell.get("data-sort", "")

    # Parse normalized name
    if "_" in sort_name:  # type: ignore
        last, first = sort_name.split("_", 1)  # type: ignore
    else:
        last, first = "", ""

    departement = cols[1].get_text(" ", strip=True)
    circo = cols[2].get_text(" ", strip=True)

    data.append(
        {
            "raw_name": raw_name,
            "first_name": first,
            "last_name": last,
            "full_name": f"{first} {last}".strip(),
            "departement": departement,
            "circonscription": circo,
        }
    )

# Save JSON
with open("deputes.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"{len(data)} députés extracted")
