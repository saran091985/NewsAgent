"""
Rule-based filtering (free, no AI):

1. keep only stories published inside the time window
2. keep only the chosen Hindu categories; drop round-ups
3. remove exact and near-duplicate stories (across all sources)
4. put each story in a topic bucket
5. flag stories that may not suit kids (flagged, not deleted)
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from difflib import SequenceMatcher

from . import config
from .models import NewsItem

_WATCH = re.compile(r"^\s*watch\s*[:|\-–]\s*", re.I)
_SKIP = [re.compile(p, re.I) for p in config.HINDU_SKIP_TITLE_PATTERNS]
_SKIP_URL = [re.compile(p, re.I) for p in config.HINDU_SKIP_URL_PATTERNS]
_SENSITIVE = re.compile(config.SENSITIVE_PATTERN, re.I)
_NOISE = re.compile(config.NOISE_PATTERN, re.I)
_BUCKETS = [(name, re.compile(rx, re.I)) for name, rx in config.BUCKET_KEYWORDS]
_UAE = dict(_BUCKETS)["UAE"]
_KEEP = {c.lower() for c in config.HINDU_KEEP_CATEGORIES}
_SPORT_CATS = {"sport", "cricket", "tennis", "football", "hockey", "athletics",
               "races", "other sports", "chess", "motorsport"}
_BUSINESS_CATS = {"business", "economy", "industry"}
_STOP = {"the", "a", "an", "of", "to", "in", "on", "for", "and", "with", "at", "by",
         "as", "is", "after", "from", "over", "its", "his", "her", "says", "said"}


def clean_headline(title: str) -> str:
    return _WATCH.sub("", title).strip()


def _norm(title: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", title.lower()).strip()


def _tokens(title: str) -> set[str]:
    return {w for w in _norm(title).split() if w not in _STOP and len(w) > 2}


def _similar(a: str, b: str) -> bool:
    if SequenceMatcher(None, _norm(a), _norm(b)).ratio() >= config.DEDUPE_SIMILARITY:
        return True
    ta, tb = _tokens(a), _tokens(b)
    if len(ta) >= 4 and len(tb) >= 4:
        return len(ta & tb) / min(len(ta), len(tb)) >= 0.8
    return False


_GENERIC = {"asian", "games", "2026", "2027", "india", "indian", "world", "says", "today", "live", "updates",
            "update", "news", "first", "after", "over", "their", "with", "from", "amid", "year", "years", "govt",
            "government", "minister", "president", "state", "states", "national", "global", "official", "report"}


def _stems(title: str) -> set[str]:
    return {w[:5] for w in _norm(title).split() if w not in _STOP and w not in _GENERIC and len(w) > 3}


def same_event(a: str, b: str) -> bool:
    """Looser check used when picking: do two headlines describe the same event?"""
    if _similar(a, b):
        return True
    sa, sb = _stems(a), _stems(b)
    shared = len(sa & sb)
    return shared >= 3 or (shared >= 2 and shared / max(1, min(len(sa), len(sb))) >= 0.5)


def assign_bucket(item: NewsItem) -> str:
    cat = (item.category or "").lower()
    headline = item.headline.lower()
    # A UAE-paper story is UAE news only if it is about the UAE (they also run
    # world, sport and lifestyle stories); otherwise classify it like any other
    feed_bucket = item.bucket
    if feed_bucket == "UAE":
        if _UAE.search(headline):
            return "UAE"
        feed_bucket = "World"
    # URL paths are only meaningful for The Hindu (RSS links are redirects)
    text = f"{item.headline} {item.url if item.origin == 'hindu' else ''}".lower()
    if cat in _SPORT_CATS:               # The Hindu's sport desks are reliable
        return "Sports"
    for name, rx in _BUCKETS:
        if rx.search(text):
            return name
    if feed_bucket:                      # RSS feed's own bucket
        return feed_bucket
    if cat in _SPORT_CATS:
        return "Sports"
    if cat in _BUSINESS_CATS:
        return "Business"
    if cat == "world":
        return "World"
    return config.DEFAULT_BUCKET


def run(items: list[NewsItem], since: datetime, until: datetime) -> tuple[list[NewsItem], dict]:
    stats = {"collected": len(items)}

    # 1. time window
    items = [i for i in items if i.published and since <= i.published <= until]
    stats["in_window"] = len(items)

    # 2. Hindu categories + round-ups
    kept = []
    for i in items:
        if i.origin == "hindu":
            if (i.category or "").lower() not in _KEEP:
                continue
            if any(p.search(i.headline) for p in _SKIP) or any(p.search(i.url) for p in _SKIP_URL):
                continue
        i.headline = clean_headline(i.headline)
        if len(i.headline.split()) < config.MIN_HEADLINE_WORDS:   # "Travel", "Latest Europe News"
            continue
        if _NOISE.search(i.headline):                               # stock chatter, adverts
            continue
        kept.append(i)
    items = kept
    stats["after_category_filter"] = len(items)

    # 3. dedupe — exact URL, then near-duplicate titles (keep the earliest copy)
    items.sort(key=lambda i: i.published)
    seen_urls: set[str] = set()
    unique: list[NewsItem] = []
    for i in items:
        if i.url in seen_urls:
            continue
        seen_urls.add(i.url)
        dup = next((u for u in unique if _similar(u.headline, i.headline)), None)
        if dup:
            if i.source != dup.source and i.source not in dup.also_in:
                dup.also_in.append(i.source)
            dup.summary = dup.summary or i.summary
            continue
        unique.append(i)
    items = unique
    stats["after_dedupe"] = len(items)

    # 4 + 5. bucket, safety flag, id
    for i in items:
        i.bucket = assign_bucket(i)
        i.sensitive = bool(_SENSITIVE.search(i.headline))
        i.id = hashlib.sha1(i.url.encode()).hexdigest()[:6]

    order = list(config.BUCKET_TARGETS)
    items.sort(key=lambda i: (order.index(i.bucket) if i.bucket in order else 99,
                              -i.published.timestamp()))
    stats["sensitive_flagged"] = sum(i.sensitive for i in items)
    stats["by_bucket"] = {b: sum(i.bucket == b for i in items) for b in order}
    return items, stats
