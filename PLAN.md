# NewsAgent — one-step daily news for the kids' YouTube show

## Goal
One run produces, for **today only**:
- a **detailed version** of each chosen story (for you, ~120–150 words each, with source link)
- a **YouTube script** for a 10-year-old host, **10–15 minutes** long (~1,400–1,900 words at ~130 words/min)

AI is used only to pick and to write. Finding the news and checking dates is done by code, which is why old news can no longer slip in.

## Pipeline

| Step | What happens | Cost |
|---|---|---|
| 1. Collect | The Hindu latest-news (scraped until stories fall outside the window) + RSS: NASA, Space.com, Google News (space), BBC World / Science / Tech, Google News (Gulf News, Khaleej Times) + Wikipedia "On this day" | free |
| 2. Filter | last 24 h only · Hindu categories kept: Business, Economy, India, Industry, News, Shorts, premium, Sport (+ sport sub-desks), Videos, World, Science · drops round-ups and opinion essays · removes duplicates across sources · sorts into topic buckets · flags unsafe-for-kids stories | free |
| 3. Suggest (optional) | one gpt-4o-mini call pre-ticks ~17 stories following the bucket targets | ~$0.001 |
| 4. **Review screen** | Gradio page: stories grouped by topic with checkboxes, per-topic target counter (e.g. "UAE 1 / 2"), flagged stories hidden behind a toggle, box to add a story by URL that you found elsewhere | free |
| 5. Enrich | fetch first paragraphs of each ticked article (RSS summary as fallback) | free |
| 6. Write | 2 structured gpt-4o-mini calls → detailed.md + youtube_script.md, facts only from the article text | ~$0.01 |
| 7. Check | every story has a source + time in window; words within limits; numbers must appear in the article | free |

## Episode shape (10–15 min, ~16–18 stories)
Intro (30 s) → World 3 → India 3 → UAE 2 → Sports 2 → Space & Science 3 → Tech 2 → Weather & Nature 1 → Business 1 → Today in history (1 min) → Outro (30 s).
Each story segment: 70–100 words, a hook line, one "did you know?" fact, simple words, no dates spoken. Adjust targets in `newsagent/config.py → BUCKET_TARGETS`.

## Output — `output/YYYY-MM-DD/`
`candidates.json/.csv` (~110 rows instead of 454) · `on_this_day.json` · `selected.json` · `news_detailed.md` · `youtube_script.md` · `run_log.json`

## Build order
1. **Collect + filter** — ✅ built (`newsagent/`), tested offline on the 31 May scrape: 454 → 109 candidates, all 8 of your Hindu picks kept.
2. **Pick top N** — ✅ built (`newsagent/select_step.py`): one gpt-4o-mini call picks `TOP_N` (default 20, `config.py` or `--top`) using the topic mix; free rule-based fallback (`--no-ai`). Output `selected.csv`.
3. **Review screen** — ✅ built (`newsagent/ui.py`, `uv run python -m newsagent ui`): collect, pick top N, tick/untick per topic with a live counter, add a story by link, save `final.csv`.
4. **Write** — ✅ built (`newsagent/write_step.py`): reads each article, one gpt-4o-mini call per story (detailed + YouTube segment + did-you-know), one for intro / today-in-history / outro; flags numbers not found in the article. Length from `SCRIPT_MINUTES` in config.
5. Replace `app.py`, redeploy on Render; retire `src/news.py`, `src/news_for_kids.py`, `Final1.txt` step.
6. One trial week next to the manual process; tune buckets, targets and blocklist.

## Commands
```
uv sync
uv run python -m newsagent check-feeds     # verify every RSS source once
uv run python -m newsagent run             # collect + pick top 20 → output/<date>/selected.csv
uv run python -m newsagent run --top 15    # different number
uv run python -m newsagent collect         # today's candidates only
uv run python -m newsagent collect --hindu-json output/the_hindu_latest.json --now 2026-05-31T18:45:00+05:30 --skip-rss   # offline test
```
