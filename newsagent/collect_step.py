"""
Step 1 of the pipeline: collect today's headlines from all free sources,
filter them, and save a small candidate list for the review screen.

Output folder: output/YYYY-MM-DD/
  candidates.json   — filtered stories (used by the review screen)
  candidates.csv    — same, for opening in Excel
  on_this_day.json  — "Today in history" events
  collect_log.json  — counts at each filter stage
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import config, filter as news_filter
from .collect import hindu, onthisday, rss
from .models import NewsItem

OUTPUT_ROOT = Path("output")


def run_dir(day: datetime) -> Path:
    d = OUTPUT_ROOT / day.strftime("%Y-%m-%d")
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_old_hindu_json(path: str) -> list[NewsItem]:
    """Load a scrape saved by the old src/news.py (for offline testing)."""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [NewsItem(headline=r["headline"], url=r["url"], source="The Hindu", origin="hindu",
                     category=r.get("category"),
                     published=datetime.fromisoformat(r["published_datetime"])
                     if r.get("published_datetime") else None)
            for r in rows]


def save(items: list[NewsItem], history: list[dict], stats: dict, folder: Path) -> None:
    rows = [i.to_row() for i in items]
    (folder / "candidates.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(folder / "candidates.csv", "w", newline="", encoding="utf-8-sig") as f:
        cols = ["id", "bucket", "sensitive", "published", "source", "category", "headline", "url", "also_in"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (folder / "on_this_day.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (folder / "collect_log.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (folder / "scores.json").unlink(missing_ok=True)   # scores belong to the previous collection


def run(now: datetime | None = None, hindu_json: str | None = None,
        skip_rss: bool = False, hours: int | None = None) -> tuple[list[NewsItem], Path]:
    now = now or datetime.now(timezone.utc)
    since = now - timedelta(hours=hours or config.WINDOW_HOURS)
    local_now = now.astimezone(config.LOCAL_TZ)
    print(f"Collecting news published {since.astimezone(config.LOCAL_TZ):%d %b %H:%M} → "
          f"{local_now:%d %b %H:%M} (UAE time)")

    items: list[NewsItem] = []
    items += load_old_hindu_json(hindu_json) if hindu_json else hindu.collect(since)
    if not skip_rss:
        items += rss.collect()
    history = [] if skip_rss else onthisday.collect(local_now)

    filtered, stats = news_filter.run(items, since, now)
    stats["window"] = {"from": since.isoformat(), "to": now.isoformat()}
    folder = run_dir(local_now)
    save(filtered, history, stats, folder)

    print("\nFilter funnel:")
    for k in ("collected", "in_window", "after_category_filter", "after_dedupe", "sensitive_flagged"):
        print(f"  {k:<24}{stats[k]}")
    print("By topic:", ", ".join(f"{b} {n}" for b, n in stats["by_bucket"].items()))
    print(f"\nSaved to {folder}/")
    return filtered, folder
