"""
Step 3: write the two versions for the final story list.

For each chosen story:
  1. fetch the article's first paragraphs (free); fall back to the RSS summary
  2. one gpt-4o-mini call returns a detailed version and a YouTube segment,
     using ONLY facts from that text
Then one more call writes the intro, "Today in history" and the outro.

Output (same folder):
  news_detailed.md   — for you: detailed version + source link per story
  youtube_script.md  — for the host: read-aloud script with timings
  write_log.json     — tokens, cost, stories with thin source text, number checks
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from . import config
from .ai_errors import AIError, friendly

PRICE_IN, PRICE_OUT = 0.15, 0.60  # gpt-4o-mini, $ per 1M tokens

# ---------------------------------------------------------------------------
# 1. Article text (free)
# ---------------------------------------------------------------------------

_BODY_SELECTORS = ["div.articlebodycontent p", "div[itemprop=articleBody] p",
                   "article p", "div.article-body p", "main p"]


def fetch_article_text(url: str, max_chars: int = 2500) -> str:
    """First paragraphs of an article. Google News links are redirects, so skip them."""
    if not url or "news.google.com" in url:
        return ""
    try:
        r = requests.get(url, headers=config.HTTP_HEADERS, timeout=20)
        r.raise_for_status()
    except requests.RequestException:
        return ""
    soup = BeautifulSoup(r.text, "html.parser")
    for sel in _BODY_SELECTORS:
        paras = [p.get_text(" ", strip=True) for p in soup.select(sel)]
        paras = [p for p in paras if len(p) > 60]
        if len(paras) >= 2:
            return "\n".join(paras)[:max_chars]
    meta = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
    return meta.get("content", "") if meta else ""


def source_text(story: dict) -> str:
    text = fetch_article_text(story.get("url", ""))
    summary = story.get("summary") or ""
    if len(text) < 200 and summary and summary not in text:
        text = (text + "\n" + summary).strip()
    return text


# ---------------------------------------------------------------------------
# 2. Writing (AI)
# ---------------------------------------------------------------------------

class StoryOut(BaseModel):
    title: str = Field(description="short, catchy title for kids, max 8 words")
    detailed: str = Field(description="detailed version for the show's editor")
    script: str = Field(description="what the young host reads aloud for this story")
    did_you_know: str = Field(description="one fun, true fact linked to the story, one sentence")


class ShowOut(BaseModel):
    intro: str
    history: str = Field(description="'Today in history' segment")
    outro: str


def _story_prompt(story: dict, text: str, script_words: int) -> str:
    lo, hi = config.DETAILED_WORDS
    return f"""Headline: {story['headline']}
Source: {story['source']}
Topic: {story['bucket']}

Article text:
\"\"\"{text or '(no article text available — use only what the headline says)'}\"\"\"

Write:
1. detailed: {lo}-{hi} words, clear and neutral, for the show's editor. What happened, who, where, why it matters.
2. script: about {script_words} words for a 10-year-old host to read aloud to kids aged 8-14.
   Start with a hook line, use short sentences and simple words, explain any hard word,
   end with why it matters or a question to the viewers. Friendly, not babyish.
3. did_you_know: one fun related fact that is common knowledge (not about today's event).

Rules: use ONLY facts from the article text or headline — never invent names, numbers or quotes.
Do not mention dates or days of the week. At most one emoji in the script, none in detailed.
If the story involves deaths or violence, say it gently and briefly without details."""


SYSTEM = ("You write a daily news show for children, presented by a 10-year-old. "
          "You are accurate first, fun second.")


def _llm(schema):
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    load_dotenv(override=True)
    return ChatOpenAI(model=config.WRITE_MODEL, temperature=0.5).with_structured_output(
        schema, include_raw=True, method="function_calling")


def _usage(res) -> tuple[int, int]:
    u = getattr(res.get("raw"), "usage_metadata", None) or {}
    return u.get("input_tokens", 0), u.get("output_tokens", 0)


def words_per_story(n: int) -> int:
    total = config.SCRIPT_MINUTES * config.WORDS_PER_MINUTE
    return max(40, int((total - 280) / max(n, 1)))   # ~280 words for intro, history, outro


# ---------------------------------------------------------------------------
# 3. Checks (free)
# ---------------------------------------------------------------------------

_NUM = re.compile(r"\d[\d,.]*")


def unsupported_numbers(written: str, text: str, headline: str) -> list[str]:
    """Numbers in the script that are not in the article text or headline."""
    have = {n.replace(",", "").rstrip(".") for n in _NUM.findall(text + " " + headline)}
    return sorted({n for n in (m.replace(",", "").rstrip(".") for m in _NUM.findall(written))
                   if n and n not in have})


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(folder: Path, stories: list[dict], progress=None) -> dict:
    n = len(stories)
    target = words_per_story(n)
    log: dict = {"stories": n, "script_words_per_story": target, "input_tokens": 0,
                 "output_tokens": 0, "thin_source": [], "number_warnings": {}, "errors": {}}

    def say(frac, msg):
        print(msg)
        if progress:
            progress(frac, desc=msg)

    say(0.05, f"Reading {n} articles…")
    with ThreadPoolExecutor(max_workers=6) as ex:
        texts = list(ex.map(source_text, stories))
    for s, t in zip(stories, texts):
        if len(t) < 200:
            log["thin_source"].append(s["headline"])

    say(0.35, f"Writing {n} stories…")
    llm = _llm(StoryOut)
    from langchain_core.messages import HumanMessage, SystemMessage
    inputs = [[SystemMessage(content=SYSTEM), HumanMessage(content=_story_prompt(s, t, target))]
              for s, t in zip(stories, texts)]
    results = llm.batch(inputs, config={"max_concurrency": 5}, return_exceptions=True)
    failures = [r for r in results if isinstance(r, Exception)]
    if failures and len(failures) == len(results):
        # nothing was written (bad key, no credit, no internet) — stop with a clear message
        raise AIError(friendly(failures[0]))

    written: list[dict] = []
    for s, t, res in zip(stories, texts, results):
        if isinstance(res, Exception) or res.get("parsed") is None:
            err = str(res if isinstance(res, Exception) else res.get("parsing_error"))
            log["errors"][s["headline"]] = err
            written.append({"title": s["headline"], "detailed": "[could not write this story]",
                            "script": "", "did_you_know": "", "story": s})
            continue
        tin, tout = _usage(res)
        log["input_tokens"] += tin
        log["output_tokens"] += tout
        out = res["parsed"].model_dump()
        bad = unsupported_numbers(out["script"] + " " + out["detailed"], t, s["headline"])
        if bad:
            log["number_warnings"][s["headline"]] = bad
        written.append({**out, "story": s})

    say(0.8, "Writing intro, history and outro…")
    history = []
    hist_file = folder / "on_this_day.json"
    if hist_file.exists():
        history = json.loads(hist_file.read_text(encoding="utf-8"))
    host = f"The host's name is {config.HOST_NAME}. " if config.HOST_NAME else ""
    show_prompt = f"""Show name: {config.SHOW_NAME}. {host}
Today's stories: {'; '.join(w['title'] for w in written)}

On this day in history (pick the ONE event most interesting for kids, ignore wars and disasters):
{chr(10).join(f"- {h['year']}: {h['text']}" for h in history) or '(none — skip history, write one line)'}

Write for the 10-year-old host to read aloud:
- intro: ~60 words, greet viewers, tease 2-3 of today's stories. No date.
- history: ~80 words, "Today in history" about the chosen event, say the year.
- outro: ~50 words, thank viewers, invite them to comment their favourite story, like and subscribe."""
    try:
        res = _llm(ShowOut).invoke([SystemMessage(content=SYSTEM), HumanMessage(content=show_prompt)])
        tin, tout = _usage(res)
        log["input_tokens"] += tin
        log["output_tokens"] += tout
        show = res["parsed"].model_dump() if res.get("parsed") else {"intro": "", "history": "", "outro": ""}
    except Exception as e:
        log["errors"]["intro/outro"] = str(e)
        show = {"intro": "", "history": "", "outro": ""}

    say(0.95, "Saving…")
    detailed_md, script_md, words = _render(written, show)
    (folder / "news_detailed.md").write_text(detailed_md, encoding="utf-8")
    (folder / "youtube_script.md").write_text(script_md, encoding="utf-8")
    log["script_words"] = words
    log["script_minutes"] = round(words / config.WORDS_PER_MINUTE, 1)
    log["cost_usd"] = round((log["input_tokens"] * PRICE_IN + log["output_tokens"] * PRICE_OUT) / 1e6, 4)
    log["run_at"] = datetime.now().isoformat(timespec="seconds")
    (folder / "write_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    say(1.0, f"Done: {words} words ≈ {log['script_minutes']} min, cost ≈ ${log['cost_usd']}")
    return {"detailed": detailed_md, "script": script_md, "log": log}


def _render(written: list[dict], show: dict) -> tuple[str, str, int]:
    today = datetime.now(config.LOCAL_TZ).strftime("%d %B %Y")
    d = [f"# News — detailed version\n_{today} · {len(written)} stories_\n"]
    sc = [f"# {config.SHOW_NAME} — YouTube script\n_{today}_\n", "## Intro", show["intro"], ""]
    words = len(show["intro"].split())
    current = None
    for k, w in enumerate(written, 1):
        s = w["story"]
        pub = (s.get("published") or "")[:16].replace("T", " ")
        d += [f"## {k}. {w['title']}", f"**{s['bucket']}** · {s['source']} · {pub}  ",
              f"Original headline: _{s['headline']}_\n", w["detailed"], f"\nSource: {s['url']}\n"]
        if s["bucket"] != current:
            current = s["bucket"]
            sc.append(f"## {current}")
        sc += [f"### {k}. {w['title']}", w["script"]]
        words += len(w["script"].split())
        if w.get("did_you_know"):
            sc.append(f"\n**Did you know?** {w['did_you_know']}")
            words += len(w["did_you_know"].split())
        sc.append("")
    sc += ["## Today in history", show["history"], "", "## Outro", show["outro"]]
    words += len(show["history"].split()) + len(show["outro"].split())
    sc.insert(1, f"_About {words} words ≈ {words / config.WORDS_PER_MINUTE:.0f} minutes_\n")
    return "\n".join(d), "\n".join(sc), words
