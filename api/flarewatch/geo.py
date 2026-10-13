"""
Place-name extraction with a GeoNames gazetteer (built by scripts/build_gazetteer.py).

Finds countries, states/provinces and cities (population >= 15k) mentioned in a message and
returns their coordinates. Rule-based on purpose: no model download, works in serverless.
"""
import json
import os
import re
import unicodedata

MAX_NGRAM = 4
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'.]*")
# Words that are capitalised at the start of sentences or in titles and happen to be place names
IGNORE = {"un", "us", "uk", "eu", "red cross", "god", "mercy", "hope", "faith", "charity", "university", "university of"}

_default = None


def _normalize(text):
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return " ".join(text.lower().replace("-", " ").replace(".", "").replace("'", "").split())


class Gazetteer:
    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        self.places = doc["places"]
        self.attribution = doc.get("attribution", "")

    def find(self, text, max_results=5):
        """Return [{name, lat, lon, country_code, kind}] for places mentioned in text (longest match first)."""
        text = str(text or "")
        if not text:
            return []
        spans = [(m.group(0), m.start()) for m in TOKEN_RE.finditer(text.replace("-", " "))]
        mixed_case = any(c.islower() for c in text) and any(c.isupper() for c in text)
        matches, i = [], 0
        while i < len(spans):
            match = None
            for n in range(min(MAX_NGRAM, len(spans) - i), 0, -1):
                words = [w for w, _ in spans[i:i + n]]
                # In normally-cased text a place name starts and ends with a capital letter
                if mixed_case and not (words[0][0].isupper() and words[-1][0].isupper()):
                    continue
                key = _normalize(" ".join(words))
                if key in IGNORE:
                    continue
                candidates = self.places.get(key)
                if candidates:
                    match = (n, candidates)
                    break
            if match:
                matches.append(match[1])
                i += match[0]
            else:
                i += 1

        # Disambiguate with context: "Alabama and Georgia" -> the US state, not the country
        context = {cands[0][3] for cands in matches if len(cands) == 1}
        found, seen = [], set()
        for cands in matches:
            pick = next((c for c in cands if c[3] in context), cands[0])
            name, lat, lon, cc, kind, _pop = pick
            if (name, cc) in seen:
                continue
            seen.add((name, cc))
            found.append({"name": name, "lat": lat, "lon": lon, "country_code": cc, "kind": kind})
            if len(found) >= max_results:
                break
        return found


def default_gazetteer():
    """Shared instance (loaded once per process); None if the data file is missing."""
    global _default
    if _default is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gazetteer.json")
        _default = Gazetteer(path) if os.path.exists(path) else False
    return _default or None


def find_locations(text, max_results=5):
    gaz = default_gazetteer()
    return gaz.find(text, max_results) if gaz else []
