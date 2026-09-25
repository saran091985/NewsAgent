from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime


@dataclass
class NewsItem:
    headline: str
    url: str
    source: str                      # "The Hindu", "BBC World", ...
    published: datetime | None       # timezone-aware
    category: str | None = None      # source's own label (The Hindu category)
    summary: str | None = None       # RSS description, if any
    bucket: str | None = None        # our topic bucket (World, UAE, Space & Science…)
    sensitive: bool = False          # flagged by the kid-safety word list
    also_in: list[str] = field(default_factory=list)  # other sources carrying the same story
    origin: str = "rss"              # "hindu" for the latest-news scraper
    id: str = ""                     # short stable id, set after filtering

    def to_row(self) -> dict:
        d = asdict(self)
        d["published"] = self.published.isoformat() if self.published else None
        d["also_in"] = ", ".join(self.also_in)
        return d
