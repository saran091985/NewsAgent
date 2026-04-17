"""
News scraper for The Hindu latest news pages.

Scrapes headlines, URLs, categories, and publication timestamps
from The Hindu's latest news section and saves to Excel/JSON.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.thehindu.com/latest-news/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )
}

MAX_RETRIES = 3
TIMEOUT = 30  # seconds


def scrape_the_hindu_page(page: int) -> List[Dict]:
    params = {"page": page}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(
                BASE_URL, params=params, headers=HEADERS, timeout=TIMEOUT
            )
            resp.raise_for_status()
            break
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt == MAX_RETRIES:
                print(f"  ✗ Page {page} failed after {MAX_RETRIES} attempts: {e}")
                return []
            wait = 2 ** attempt
            print(f"  ⚠ Page {page} attempt {attempt} failed, retrying in {wait}s…")
            time.sleep(wait)

    soup = BeautifulSoup(resp.text, "html.parser")

    items: List[Dict] = []

    # Page-level date (e.g. "Thursday, 8th January 2026")
    page_date_el = soup.select_one("div.latest-news div.latest-date")
    page_date: Optional[str] = (
        page_date_el.get_text(strip=True) if page_date_el else None
    )

    # Each news item on that page
    for li in soup.select("div.latest-news ul.timeline-with-img > li"):
        # headline and url
        title_a = li.select_one("h3.title > a")
        if not title_a:
            continue

        headline = title_a.get_text(strip=True)
        url = title_a.get("href")

        # category (e.g. "Kerala")
        category_a = li.select_one("div.right-content div.label > a")
        category: Optional[str] = (
            category_a.get_text(strip=True) if category_a else None
        )

        # published datetime (ISO) from data-published
        time_div = li.select_one("div.news-time.time")
        published_iso: Optional[str] = None
        if time_div and time_div.has_attr("data-published"):
            published_iso = time_div["data-published"]

        items.append(
            {
                "headline": headline,
                "url": url,
                "category": category,
                "page": page,
                "page_date": page_date,
                "published_datetime": published_iso,
            }
        )

    return items


def scrape_the_hindu_latest_n_pages(n_pages: int = 10) -> List[Dict]:
    all_items: List[Dict] = []
    for page in range(1, n_pages + 1):
        print(f"Scraping page {page}/{n_pages}…")
        page_items = scrape_the_hindu_page(page)
        all_items.extend(page_items)
        print(f"  ✓ Got {len(page_items)} items (total: {len(all_items)})")
    return all_items


def save_news_to_files(
    items: List[Dict],
    excel_path: str = "output/the_hindu_latest.xlsx",
    json_path: str = "output/the_hindu_latest.json",
) -> None:
    """
    Save scraped news items to an Excel file and a JSON file.

    items: list of dicts returned by scrape_the_hindu_latest_n_pages()
    excel_path: output .xlsx path
    json_path: output .json path
    """

    # Ensure parent dirs exist
    Path(excel_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)

    # Excel
    df = pd.DataFrame(items)
    df.to_csv(excel_path, index=False)

    # JSON (pretty-printed UTF-8)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    items = scrape_the_hindu_latest_n_pages(n_pages=10)
    print(f"Scraped {len(items)} items")
    save_news_to_files(items)
    print("Saved to the_hindu_latest.xlsx and the_hindu_latest.json")
