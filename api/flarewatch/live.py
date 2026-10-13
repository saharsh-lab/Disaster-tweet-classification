"""
Live social signals for Flarewatch.

Sources
- Mastodon hashtag timelines (public, no key): https://<instance>/api/v1/timelines/tag/<tag>
- Bluesky search (optional): enabled when BLUESKY_HANDLE and BLUESKY_APP_PASSWORD are set
  (a free app password from Bluesky settings; anonymous search is no longer reliable).

Posts are classified with the deployed models and cached in memory for FLAREWATCH_LIVE_TTL seconds
(default 180), so page views do not hammer the upstream APIs. Only the standard library is used,
which keeps the serverless bundle dependency-free.

Configuration (environment variables)
    FLAREWATCH_MASTODON_INSTANCE   default "mastodon.social"
    FLAREWATCH_LIVE_TAGS           comma-separated hashtags (default: disaster-related tags below)
    FLAREWATCH_LIVE_TTL            cache lifetime in seconds (default 180)
    FLAREWATCH_LIVE_INCLUDE_BOTS   "1" to keep automated accounts (quake/alert bots are skipped by default)
    BLUESKY_HANDLE, BLUESKY_APP_PASSWORD   optional Bluesky credentials
"""
import json
import os
import ssl
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser

DEFAULT_TAGS = ["flood", "earthquake", "wildfire", "hurricane", "cyclone", "tornado", "disasterrelief", "evacuation"]
USER_AGENT = "Flarewatch/3.1 (crisis-signal research demo; +https://github.com/saharsh-lab/Disaster-tweet-classification)"
TIMEOUT = 6
MAX_POSTS = 60


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ("br", "p"):
            self.parts.append(" ")

    def handle_data(self, data):
        self.parts.append(data)


def html_to_text(html):
    parser = _TextExtractor()
    parser.feed(html or "")
    return " ".join(unescape("".join(parser.parts)).split())


def _ssl_context():
    # python.org builds on macOS ship without CA certificates; prefer certifi's bundle when present
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


_SSL = _ssl_context()


def _get_json(url, headers=None, data=None):
    req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT, "Accept": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=TIMEOUT, context=_SSL) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_mastodon(instance, tag, limit=20):
    url = f"https://{instance}/api/v1/timelines/tag/{urllib.parse.quote(tag)}?limit={limit}"
    posts = []
    include_bots = os.environ.get("FLAREWATCH_LIVE_INCLUDE_BOTS") == "1"
    for status in _get_json(url):
        if status.get("reblog") or status.get("language") not in (None, "en"):
            continue
        if (status.get("account") or {}).get("bot") and not include_bots:
            continue  # automated alert feeds are not people asking for help
        text = html_to_text(status.get("content", ""))
        if not text:
            continue
        posts.append({
            "id": f"mastodon:{status.get('id')}",
            "source": "mastodon",
            "url": status.get("url") or status.get("uri"),
            "author": (status.get("account") or {}).get("acct", ""),
            "created_at": status.get("created_at"),
            "text": text,
            "tag": tag,
        })
    return posts


def bluesky_session():
    handle, password = os.environ.get("BLUESKY_HANDLE"), os.environ.get("BLUESKY_APP_PASSWORD")
    if not handle or not password:
        return None
    body = json.dumps({"identifier": handle, "password": password}).encode("utf-8")
    return _get_json("https://bsky.social/xrpc/com.atproto.server.createSession", {"Content-Type": "application/json"}, body)["accessJwt"]


def fetch_bluesky(token, query, limit=25):
    url = "https://bsky.social/xrpc/app.bsky.feed.searchPosts?" + urllib.parse.urlencode({"q": query, "limit": limit, "lang": "en", "sort": "latest"})
    posts = []
    for post in _get_json(url, {"Authorization": f"Bearer {token}"}).get("posts", []):
        record = post.get("record") or {}
        handle = (post.get("author") or {}).get("handle", "")
        rkey = post.get("uri", "").rsplit("/", 1)[-1]
        text = " ".join(str(record.get("text", "")).split())
        if not text:
            continue
        posts.append({
            "id": f"bluesky:{post.get('uri')}",
            "source": "bluesky",
            "url": f"https://bsky.app/profile/{handle}/post/{rkey}" if handle and rkey else None,
            "author": handle,
            "created_at": record.get("createdAt") or post.get("indexedAt"),
            "text": text,
            "tag": query,
        })
    return posts


class LiveFeed:
    def __init__(self, classifier):
        self.classifier = classifier
        self.instance = os.environ.get("FLAREWATCH_MASTODON_INSTANCE", "mastodon.social")
        tags = os.environ.get("FLAREWATCH_LIVE_TAGS")
        self.tags = [t.strip().lstrip("#") for t in tags.split(",") if t.strip()] if tags else DEFAULT_TAGS
        self.ttl = int(os.environ.get("FLAREWATCH_LIVE_TTL", "180"))
        self._lock = threading.Lock()
        self._cache = None
        self._fetched = 0.0

    def _collect(self):
        errors, posts = [], []
        jobs = [("mastodon", tag) for tag in self.tags]
        token = None
        try:
            token = bluesky_session()
        except Exception as e:  # bad credentials or network: keep Mastodon working
            errors.append(f"bluesky: session failed ({type(e).__name__})")
        if token:
            jobs += [("bluesky", tag) for tag in self.tags[:4]]

        def run(job):
            source, tag = job
            try:
                return fetch_mastodon(self.instance, tag) if source == "mastodon" else fetch_bluesky(token, tag)
            except Exception as e:
                errors.append(f"{source} #{tag}: {type(e).__name__}")
                return []

        with ThreadPoolExecutor(max_workers=8) as pool:
            for batch in pool.map(run, jobs):
                posts.extend(batch)

        # De-duplicate (same post via several tags, or cross-posted text)
        unique, seen_ids, seen_text = [], set(), set()
        for p in posts:
            key = p["text"].lower()[:140]
            if p["id"] in seen_ids or key in seen_text:
                continue
            seen_ids.add(p["id"])
            seen_text.add(key)
            unique.append(p)
        unique.sort(key=lambda p: p.get("created_at") or "", reverse=True)
        unique = unique[:MAX_POSTS]

        for p in unique:
            r = self.classifier.classify(p["text"][:1000], explain=False)
            p.update({
                "is_request": r["is_request"],
                "probability": r["probability"],
                "confidence_percent": r["confidence_percent"],
                "cleaned": r["cleaned"],
                "resource": r["resource"],
                "resources": [n["name"] for n in r["resources"]],
                "locations": r["locations"],
            })
            p["text"] = p["text"][:600]

        sources = [f"mastodon:{self.instance}"] + (["bluesky"] if token else [])
        status = "ok" if unique and not errors else ("partial" if unique else "unavailable")
        return {
            "status": status,
            "sources": sources,
            "tags": self.tags,
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "counts": {"posts": len(unique), "requests": sum(1 for p in unique if p["is_request"])},
            "posts": unique,
            "errors": errors,
        }

    def get(self, force=False):
        with self._lock:
            fresh = self._cache is not None and time.time() - self._fetched < self.ttl
            if not fresh or force:
                result = self._collect()
                # Keep serving the last good snapshot if the upstream is temporarily down
                if result["posts"] or self._cache is None:
                    self._cache, self._fetched = result, time.time()
                else:
                    self._cache = dict(self._cache, status="stale", errors=result["errors"])
            age = int(time.time() - self._fetched)
            return dict(self._cache, cached=fresh and not force, age_seconds=age, ttl_seconds=self.ttl)
