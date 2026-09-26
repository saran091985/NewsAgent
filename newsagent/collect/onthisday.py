"""'Today in history' from Wikipedia's free On-This-Day API."""

from __future__ import annotations

from datetime import datetime

import requests

from .. import config


def collect(day: datetime) -> list[dict]:
    url = config.ON_THIS_DAY_URL.format(mm=f"{day.month:02d}", dd=f"{day.day:02d}")
    try:
        r = requests.get(url, headers=config.HTTP_HEADERS, timeout=config.HTTP_TIMEOUT)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"  ✗ On this day: {e}")
        return []
    events = []
    for ev in r.json().get("selected", [])[: config.ON_THIS_DAY_MAX]:
        page = (ev.get("pages") or [{}])[0]
        events.append({
            "year": ev.get("year"),
            "text": ev.get("text"),
            "url": page.get("content_urls", {}).get("desktop", {}).get("page"),
        })
    # international / special days (e.g. International Tea Day)
    try:
        h = requests.get(config.HOLIDAYS_URL.format(mm=f"{day.month:02d}", dd=f"{day.day:02d}"),
                         headers=config.HTTP_HEADERS, timeout=config.HTTP_TIMEOUT)
        h.raise_for_status()
        for ev in h.json().get("holidays", [])[:15]:
            events.append({"year": None, "text": ev.get("text"), "url": None, "kind": "day"})
    except requests.RequestException as e:
        print(f"  ✗ Special days: {e}")
    print(f"  On this day: {len(events)} events and special days")
    return events
