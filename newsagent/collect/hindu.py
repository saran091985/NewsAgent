"""
The Hindu latest-news scraper (free).

Pages through https://www.thehindu.com/latest-news/ and stops as soon as a
page contains stories older than the time window, so it no longer blindly
downloads 10 pages.
"""

from __future__ import annotations

import time
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from .. import config
from ..models import NewsItem

BASE_URL = "https://www.thehindu.com/latest-news/"
SOURCE = "The Hindu"


def _fetch(page: int, retries: int = 3) -> str | None:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(BASE_URL, params={"page": page},
                             headers=config.HTTP_HEADERS, timeout=config.HTTP_TIMEOUT)
            r.raise_for_status()
            return r.text
        except requests.RequestException as e:
            if attempt == retries:
                print(f"  ✗ The Hindu page {page} failed: {e}")
                return None
            time.sleep(2 ** attempt)
    return None


def parse_page(html: str) -> list[NewsItem]:
    soup = BeautifulSoup(html, "html.parser")
    items: list[NewsItem] = []
    for li in soup.select("div.latest-news ul.timeline-with-img > li"):
        a = li.select_one("h3.title > a")
        if not a:
            continue
        cat = li.select_one("div.right-content div.label > a")
        t = li.select_one("div.news-time.time")
        published = None
        if t and t.has_attr("data-published"):
            try:
                published = datetime.fromisoformat(t["data-published"])
            except ValueError:
                pass
        items.append(NewsItem(
            headline=a.get_text(strip=True),
            url=a.get("href", ""),
            source=SOURCE,
            published=published,
            category=cat.get_text(strip=True) if cat else None,
            origin="hindu",
        ))
    return items


def collect(since: datetime) -> list[NewsItem]:
    """Scrape pages until stories are older than `since`."""
    out: list[NewsItem] = []
    for page in range(1, config.HINDU_MAX_PAGES + 1):
        html = _fetch(page)
        if html is None:
            break
        items = parse_page(html)
        out.extend(items)
        dated = [i.published for i in items if i.published]
        print(f"  The Hindu page {page}: {len(items)} items")
        if not items or (dated and min(dated) < since):
            break  # reached the edge of the time window
    return out
