"""
Command line:

  uv run python -m newsagent run                # collect + pick top 20 (one step)
  uv run python -m newsagent run --top 15       # pick a different number
  uv run python -m newsagent collect            # only collect today's candidates
  uv run python -m newsagent select --top 20    # only pick from the latest candidates
  uv run python -m newsagent select --no-ai     # free rule-based pick, no OpenAI call
  uv run python -m newsagent check-feeds        # test every RSS feed once
  uv run python -m newsagent ui                 # open the review screen in the browser
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

from . import collect_step, config, select_step
from .ai_errors import AIError

app = typer.Typer(add_completion=False)


@app.command()
def collect(
    hours: int = typer.Option(config.WINDOW_HOURS, help="Time window in hours"),
    hindu_json: Optional[str] = typer.Option(None, help="Use an old saved Hindu scrape instead of scraping (testing)"),
    now: Optional[str] = typer.Option(None, help="Pretend the run happens at this ISO time (testing)"),
    skip_rss: bool = typer.Option(False, help="Only use The Hindu"),
):
    """Collect + filter today's headlines (free, no AI)."""
    collect_step.run(now=datetime.fromisoformat(now) if now else None,
                     hindu_json=hindu_json, skip_rss=skip_rss, hours=hours)


@app.command()
def select(
    top: int = typer.Option(config.TOP_N, help="How many stories to pick"),
    no_ai: bool = typer.Option(False, "--no-ai", help="Free rule-based pick, no OpenAI call"),
    date: Optional[str] = typer.Option(None, help="Folder date YYYY-MM-DD (default: latest)"),
):
    """Pick the TOP most important stories from the collected candidates."""
    folder = config.OUTPUT_DIR / date if date else None
    _select(top, folder, no_ai)


@app.command()
def run(
    top: int = typer.Option(config.TOP_N, help="How many stories to pick"),
    hours: int = typer.Option(config.WINDOW_HOURS, help="Time window in hours"),
    no_ai: bool = typer.Option(False, "--no-ai", help="Free rule-based pick, no OpenAI call"),
):
    """Collect today's news, then pick the TOP stories — one step."""
    _, folder = collect_step.run(hours=hours)
    print()
    _select(top, folder, no_ai)


def _select(top, folder, no_ai):
    try:
        select_step.run(n=top, folder=folder, use_ai=not no_ai)
    except AIError as e:
        typer.secho(f"\n✗ AI pick failed: {e}\n  (Use --no-ai for the free rule-based pick.)", fg="red", err=True)
        raise typer.Exit(1)


@app.command()
def ui(
    port: Optional[int] = typer.Option(None, help="Port (default: $PORT or 7860)"),
    host: Optional[str] = typer.Option(None, help="Host (default: 127.0.0.1 locally, 0.0.0.0 on a server)"),
):
    """Open the review screen (collect → pick → review → write) in your browser."""
    from . import ui as ui_module
    ui_module.main(port=port, host=host)


@app.command("check-feeds")
def check_feeds():
    """Fetch every configured RSS feed once and report how many items it returned."""
    from .collect import rss
    for feed in config.RSS_FEEDS:
        items = rss.fetch_feed(feed)
        newest = max((i.published for i in items if i.published), default=None)
        status = "OK " if items else "FAIL"
        print(f"{status} {feed['name']:<30} {len(items):>3} items  newest: {newest}")


if __name__ == "__main__":
    app()
