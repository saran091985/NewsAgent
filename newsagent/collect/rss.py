"""
RSS collector (free) for space, world, tech and UAE news.

Every RSS item carries its own publish time, so filtering to "today" is exact.
"""

from __future__ import annotations

import html
import re
from datetime import datetime, timezone

import feedparser
import requests

from .. import config
from ..models import NewsItem

_TAG = re.compile(r"<[^>]+>")


def _clean(text: str | None) -> str | None:
    if not text:
        return None
    text = html.unescape(_TAG.sub(" ", text))
    return re.sub(r"\s+", " ", text).strip() or None


def _published(entry) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        t = entry.get(key)
        if t:
            return datetime(*t[:6], tzinfo=timezone.utc)
    return None


def fetch_feed(feed: dict) -> list[NewsItem]:
    try:
        r = requests.get(feed["url"], headers=config.HTTP_HEADERS, timeout=config.HTTP_TIMEOUT)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"  ✗ {feed['name']}: {e}")
        return []
    parsed = feedparser.parse(r.content)
    items: list[NewsItem] = []
    for e in parsed.entries[: config.RSS_MAX_ITEMS_PER_FEED]:
        title = _clean(e.get("title"))
        if not title:
            continue
        source = feed["name"]
        # Google News titles look like "Headline - Gulf News"; keep the real publisher
        src = e.get("source", {}).get("title") if isinstance(e.get("source"), dict) else None
        if src and title.endswith(f" - {src}"):
            title = title[: -len(src) - 3]
            source = src
        items.append(NewsItem(
            headline=title,
            url=e.get("link", ""),
            source=source,
            published=_published(e),
            summary=_clean(e.get("summary")),
            bucket=feed.get("bucket"),
        ))
    return items


def collect() -> list[NewsItem]:
    out: list[NewsItem] = []
    for feed in config.RSS_FEEDS:
        items = fetch_feed(feed)
        print(f"  {feed['name']}: {len(items)} items")
        out.extend(items)
    return out
