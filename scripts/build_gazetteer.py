"""
Build flarewatch/data/gazetteer.json from GeoNames dumps (CC BY 4.0, https://www.geonames.org).

Inputs (download into training_data/geonames/):
    https://download.geonames.org/export/dump/cities15000.zip   (unzipped to cities15000.txt)
    https://download.geonames.org/export/dump/countryInfo.txt
    https://download.geonames.org/export/dump/admin1CodesASCII.txt

Usage:
    venv/bin/python scripts/build_gazetteer.py

Countries and first-level regions (states, provinces) are placed at the population-weighted
centre of their cities. Single-word city and region names that are ordinary English words (e.g. "Mobile",
"Nice", "Reading") are dropped using the system word list, so they are not mistaken for places.
"""
import json
import os
import unicodedata
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE_DIR, "training_data", "geonames")
OUT = os.path.join(BASE_DIR, "flarewatch", "data", "gazetteer.json")
WORDS = "/usr/share/dict/words"


def normalize(name):
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = name.lower().replace("-", " ").replace(".", "").replace("'", "")
    return " ".join(name.split())


def main():
    common = set()
    if os.path.exists(WORDS):
        with open(WORDS, encoding="utf-8", errors="ignore") as f:
            common = {w.strip() for w in f if w[:1].islower()}

    countries = {}
    with open(os.path.join(SRC, "countryInfo.txt"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            countries[cols[0]] = {"name": cols[4], "population": int(cols[7] or 0)}

    admin1 = {}
    with open(os.path.join(SRC, "admin1CodesASCII.txt"), encoding="utf-8") as f:
        for line in f:
            code, name, ascii_name, _ = line.rstrip("\n").split("\t")
            admin1[code] = {"name": ascii_name or name}

    entries = defaultdict(list)  # normalized name -> candidates [display, lat, lon, country_code, kind, population]
    region_acc = defaultdict(lambda: [0.0, 0.0, 0])  # key -> [sum lat*pop, sum lon*pop, sum pop]

    def offer(key, record):
        if len(key) < 3:
            return
        entries[key].append(record)

    with open(os.path.join(SRC, "cities15000.txt"), encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            name, ascii_name = cols[1], cols[2]
            lat, lon = float(cols[4]), float(cols[5])
            cc, a1 = cols[8], cols[10]
            pop = int(cols[14] or 0)
            weight = max(pop, 1)
            for key in (f"C:{cc}", f"A:{cc}.{a1}"):
                acc = region_acc[key]
                acc[0] += lat * weight; acc[1] += lon * weight; acc[2] += weight
            for variant in {name, ascii_name}:
                key = normalize(variant)
                if " " not in key and key in common:
                    continue  # ordinary English word, too ambiguous
                offer(key, [ascii_name or name, round(lat, 3), round(lon, 3), cc, "city", pop])

    for code, info in admin1.items():
        acc = region_acc.get(f"A:{code}")
        if not acc or not acc[2]:
            continue
        cc = code.split(".")[0]
        key = normalize(info["name"])
        if " " not in key and key in common:
            continue  # regions literally named "West", "Centre", "Capital"...
        offer(key, [info["name"], round(acc[0] / acc[2], 3), round(acc[1] / acc[2], 3), cc, "region", acc[2]])

    for cc, info in countries.items():
        acc = region_acc.get(f"C:{cc}")
        if not acc or not acc[2]:
            continue
        # Countries always win over a same-named town; population makes regions vs countries comparable
        offer(normalize(info["name"]), [info["name"], round(acc[0] / acc[2], 3), round(acc[1] / acc[2], 3), cc, "country", max(info["population"], acc[2]) * 10])

    # Keep the three most prominent candidates per name (one per country) for context-based disambiguation
    places = {}
    for key, cands in entries.items():
        cands.sort(key=lambda r: -r[5])
        picked, seen_cc = [], set()
        for r in cands:
            if r[3] not in seen_cc:
                picked.append(r)
                seen_cc.add(r[3])
            if len(picked) == 3:
                break
        places[key] = picked

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    doc = {
        "attribution": "Place data from GeoNames (https://www.geonames.org), licensed CC BY 4.0.",
        "fields": ["name", "lat", "lon", "country_code", "kind", "population"],
        "places": places,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(doc, f, separators=(",", ":"))
    kinds = defaultdict(int)
    for cands in places.values():
        kinds[cands[0][4]] += 1
    print(f"{len(places)} names ({dict(kinds)}) -> {os.path.relpath(OUT, BASE_DIR)} ({os.path.getsize(OUT) // 1024} KB)")


if __name__ == "__main__":
    main()
