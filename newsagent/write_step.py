"""
Step 3: write the two versions for the final story list.

For each chosen story:
  1. fetch the article's first paragraphs (free); fall back to the RSS summary,
     and for stories with little text, one news search (Serper) for more facts
  2. one gpt-4o-mini call writes the story in the show's style (style/style_guide.md):
     a playful title, bullet points with bold labels, and a "Wait, what's…?" explainer
     — a longer version (detailed) and a shorter one (YouTube script)
Then one more call writes the intro, the topic openers, "Special today" and the outro.

Output (same folder):
  news_detailed.md   — for you: detailed version + source link per story
  youtube_script.md  — for the host: read-aloud script with timings
  youtube_script.txt — the same script as plain text for a teleprompter app: no * # > symbols,
                       ●●● between paragraphs and ★ TITLE ★ around titles (the app drops line breaks)
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


def search_snippets(headline: str) -> str:
    """Today's news snippets about this headline. Used only when the article is thin.
    Free DuckDuckGo news first; Serper (~$0.001) only if that fails (config.SEARCH_PROVIDER)."""
    import os
    if not config.SEARCH_THIN_STORIES:
        return ""
    mode = config.SEARCH_PROVIDER
    if mode in ("free", "free_then_serper"):
        try:
            from ddgs import DDGS
            rows = DDGS().news(headline, timelimit="d", max_results=6) or []
            if rows or mode == "free":
                return "\n".join(f"- {n.get('title', '')}: {n.get('body', '')} ({n.get('source', '')})"
                                 for n in rows[:6])
        except Exception:
            if mode == "free":
                return ""
    if not os.getenv("SERPER_API_KEY"):
        return ""
    try:
        from langchain_community.utilities import GoogleSerperAPIWrapper
        res = GoogleSerperAPIWrapper(type="news", tbs="qdr:d", k=6).results(headline)
    except Exception:
        return ""
    lines = [f"- {n.get('title', '')}: {n.get('snippet', '')} ({n.get('source', '')})"
             for n in res.get("news", [])[:6]]
    return "\n".join(lines)


def source_text(story: dict) -> str:
    text = fetch_article_text(story.get("url", ""))
    summary = story.get("summary") or ""
    if len(text) < 200 and summary and summary not in text:
        text = (text + "\n" + summary).strip()
    if len(text) < 400:
        more = search_snippets(story["headline"])
        if more:
            text = (text + "\n\nOther reports today:\n" + more).strip()
    return text


# ---------------------------------------------------------------------------
# 2. Writing (AI)
# ---------------------------------------------------------------------------

class Point(BaseModel):
    label: str = Field(description="2-4 word bold label that fits the story, e.g. 'The Battle', 'Why it matters'")
    text: str = Field(description="1-2 short sentences")


class StoryOut(BaseModel):
    title: str = Field(description="playful, specific title in the show's style, max 9 words")
    emoji: str = Field(description="one emoji that fits the story")
    opener: str = Field(description="one lively lead-in sentence the host says before the points")
    script_points: list[Point] = Field(description="2-3 points for the YouTube script")
    detailed_points: list[Point] = Field(description="4-5 points for the detailed version: more facts, same style")
    explain_term: str = Field(default="", description="the hardest word that is USED in script_points, or empty")
    explain_text: str = Field(default="", description="1-2 sentence kid explanation with an everyday comparison")


class Opener(BaseModel):
    topic: str
    line: str


class ShowOut(BaseModel):
    intro: str = Field(description="fun greeting that teases 2-3 of today's biggest stories")
    topic_openers: list[Opener] = Field(description="one lively line in quotes for each topic")
    special_today: str = Field(default="", description="short 'special today' segment, or empty")
    outro: str


def _style_guide() -> str:
    try:
        return Path(config.STYLE_GUIDE).read_text(encoding="utf-8")
    except OSError:
        return ""


SYSTEM = ("You write \"" + config.SHOW_NAME + "\", a daily YouTube news show for kids aged 8-14, presented by "
          "a 10-year-old host. You are accurate first, fun second: every fact comes from the text you are given.")


def _story_prompt(story: dict, text: str, script_words: int) -> str:
    lo, hi = config.DETAILED_WORDS
    return f"""{_style_guide()}

---
Write today's story in exactly that style.

Topic: {story['bucket']}
Headline: {story['headline']}
Source: {story['source']}

Facts you may use (article text and other reports):
\"\"\"{text or '(no article text available — use only what the headline says)'}\"\"\"

- script_points: the YouTube version, about {script_words} words in total (opener + points + explainer).
- detailed_points: the longer version, {lo}-{hi} words in total, more facts and background, same kid style.
- explain_term / explain_text: explain the one hardest word that you actually USED in the opener or
  script_points (the YouTube version). Never explain a word that only appears in detailed_points.
  If the YouTube version has no hard word, leave both empty.
- Use ONLY facts from the text above. Never invent names, numbers, quotes or reasons.
  If the text is thin, keep the story short rather than guessing.
- Do not mention today's date or days of the week."""


def _llm(schema):
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    load_dotenv(override=True)
    return ChatOpenAI(model=config.WRITE_MODEL, temperature=0.7).with_structured_output(
        schema, include_raw=True, method="function_calling")


def _usage(res) -> tuple[int, int]:
    u = getattr(res.get("raw"), "usage_metadata", None) or {}
    return u.get("input_tokens", 0), u.get("output_tokens", 0)


def words_per_story(n: int) -> int:
    total = config.SCRIPT_MINUTES * config.WORDS_PER_MINUTE
    return max(40, min(110, int((total - 300) / max(n, 1))))   # ~300 words for intro, openers, special today, outro


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

    say(0.02, f"Reading {n} articles…")
    with ThreadPoolExecutor(max_workers=6) as ex:
        texts = list(ex.map(source_text, stories))
    for s, t in zip(stories, texts):
        if len(t) < 200:
            log["thin_source"].append(s["headline"])

    say(0.2, f"Writing story 0 of {n}…")
    llm = _llm(StoryOut)
    from concurrent.futures import as_completed
    from langchain_core.messages import HumanMessage, SystemMessage
    inputs = [[SystemMessage(content=SYSTEM), HumanMessage(content=_story_prompt(s, t, target))]
              for s, t in zip(stories, texts)]

    def call(msgs):
        try:
            return llm.invoke(msgs)
        except Exception as e:          # keep going; the story is marked as failed
            return e

    results: list = [None] * n
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(call, m): i for i, m in enumerate(inputs)}
        for done, fut in enumerate(as_completed(futures), 1):
            results[futures[fut]] = fut.result()
            say(0.2 + 0.6 * done / n, f"Writing story {done} of {n}…")
    failures = [r for r in results if isinstance(r, Exception)]
    if failures and len(failures) == len(results):
        # nothing was written (bad key, no credit, no internet) — stop with a clear message
        raise AIError(friendly(failures[0]))

    written: list[dict] = []
    for s, t, res in zip(stories, texts, results):
        if isinstance(res, Exception) or res.get("parsed") is None:
            err = str(res if isinstance(res, Exception) else res.get("parsing_error"))
            log["errors"][s["headline"]] = err
            written.append({"title": s["headline"], "emoji": "", "opener": "[could not write this story]",
                            "script_points": [], "detailed_points": [], "explain_term": "", "explain_text": "",
                            "story": s})
            continue
        tin, tout = _usage(res)
        log["input_tokens"] += tin
        log["output_tokens"] += tout
        out = res["parsed"].model_dump()
        body = " ".join(p["text"] for p in out["script_points"] + out["detailed_points"]) + " " + out["opener"]
        bad = unsupported_numbers(body, t, s["headline"])
        if bad:
            log["number_warnings"][s["headline"]] = bad
        written.append({**out, "story": s})

    say(0.8, "Writing intro, topic openers and outro…")
    history = []
    hist_file = folder / "on_this_day.json"
    if hist_file.exists():
        history = json.loads(hist_file.read_text(encoding="utf-8"))
    topics = list(dict.fromkeys(w["story"]["bucket"] for w in written))
    host = f"The host's name is {config.HOST_NAME}. " if config.HOST_NAME else ""
    special = ""
    if config.INCLUDE_SPECIAL_TODAY:
        special = ("\nToday's special days and 'on this day' events:\n"
                   + ("\n".join(f"- {h.get('year') or 'Day'}: {h['text']}" for h in history) or "(none)")
                   + "\nspecial_today: if one of these is fun for kids (an international day, a famous invention or"
                     " discovery, a space milestone), write 40-70 words about it in the show's style. Ignore wars,"
                     " disasters and religious feast days. If none fits, leave it empty.")
    show_prompt = f"""{_style_guide()}

---
Show name: {config.SHOW_NAME}. {host}
Today's stories: {'; '.join(w['title'] for w in written)}
Topics in order: {', '.join(topics)}
{special}

Write for the 10-year-old host to read aloud, in the style above:
- intro: 40-70 words.
- topic_openers: one lively line for each topic above (in that order).
- outro: 25-45 words."""
    from langchain_core.messages import HumanMessage, SystemMessage
    empty = {"intro": "", "topic_openers": [], "special_today": "", "outro": ""}
    try:
        res = _llm(ShowOut).invoke([SystemMessage(content=SYSTEM), HumanMessage(content=show_prompt)])
        tin, tout = _usage(res)
        log["input_tokens"] += tin
        log["output_tokens"] += tout
        show = res["parsed"].model_dump() if res.get("parsed") else empty
    except Exception as e:
        log["errors"]["intro/outro"] = str(e)
        show = empty

    say(0.95, "Saving…")
    detailed_md, script_md, words = _render(written, show)
    (folder / "news_detailed.md").write_text(detailed_md, encoding="utf-8")
    (folder / "youtube_script.md").write_text(script_md, encoding="utf-8")
    teleprompter = to_markers(render_teleprompter(written, show))
    save_teleprompter(folder, teleprompter)
    log["script_words"] = words
    log["script_minutes"] = round(words / config.WORDS_PER_MINUTE, 1)
    log["cost_usd"] = round((log["input_tokens"] * PRICE_IN + log["output_tokens"] * PRICE_OUT) / 1e6, 4)
    log["run_at"] = datetime.now().isoformat(timespec="seconds")
    (folder / "write_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    say(1.0, f"Done: {words} words ≈ {log['script_minutes']} min, cost ≈ ${log['cost_usd']}")
    return {"detailed": detailed_md, "script": script_md, "teleprompter": teleprompter, "log": log}


SECTION = {
    "World": "🌍 WORLD NEWS", "India": "🇮🇳 INDIA NEWS", "UAE": "🇦🇪 UAE NEWS", "Sports": "🏆 SPORTS NEWS",
    "Space & Science": "🚀 SPACE & SCIENCE NEWS", "Tech": "💻 TECHNOLOGY NEWS",
    "Weather & Nature": "🌦️ WEATHER & NATURE", "Business": "💰 MONEY & BUSINESS",
}


_STOP_TERM = {"a", "an", "the", "of", "to", "in", "on", "and"}


def term_used(term: str, text: str) -> bool:
    """Is the explained word really in this version's text? ('Humid' matches 'humidity', 'Sovereignty' 'sovereign')."""
    words = [w for w in re.findall(r"[a-z0-9]+", (term or "").lower()) if w not in _STOP_TERM]
    if not words:
        return False
    low = (text or "").lower()
    return all(re.search(r"\b" + re.escape(w[:max(4, min(len(w), 6))]), low) for w in words)


def _explainer_fits(w: dict, points_key: str) -> bool:
    """Show the 'Wait, what's…?' box only if its word appears in the text of THIS version."""
    if not (w.get("explain_term") and w.get("explain_text")):
        return False
    body = " ".join([w.get("title", ""), w.get("opener", "")] + [p["text"] for p in w.get(points_key, [])]
                    + [p["label"] for p in w.get(points_key, [])])
    return term_used(w["explain_term"], body)


def _tidy(text: str) -> str:
    """Straight quotes/apostrophes — curly ones show as odd characters in some editors and teleprompters."""
    return (text or "").translate(str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"'}))


def _story_md(k: int, w: dict, points_key: str, with_source: bool) -> list[str]:
    s = w["story"]
    out = [f"### {k}. {w['title']} {w.get('emoji', '')}".rstrip()]
    if w.get("opener"):
        out.append(w["opener"])
    out.append("")
    out += [f"- **{p['label'].rstrip(':')}:** {p['text']}" for p in w.get(points_key, [])]
    if _explainer_fits(w, points_key):
        term = w["explain_term"].strip().rstrip("?")
        out += ["", f"> **Wait, what's {term}?** {w['explain_text']}"]
    if with_source:
        pub = (s.get("published") or "")[:16].replace("T", " ")
        out += ["", f"<sub>Source: [{s['source']}]({s['url']}) · {pub} · original headline: {s['headline']}</sub>"]
    out.append("")
    return out


def _words(lines: list[str]) -> int:
    text = " ".join(l for l in lines if not l.startswith("<sub>") and not l.startswith("#"))
    return len(re.sub(r"[*_>`\-]", " ", text).split())


def _render(written: list[dict], show: dict) -> tuple[str, str, int]:
    today = datetime.now(config.LOCAL_TZ).strftime("%d %B %Y")
    openers = {o["topic"]: o["line"] for o in show.get("topic_openers", [])}

    def build(points_key: str, with_source: bool) -> list[str]:
        out = [show.get("intro", ""), ""] if show.get("intro") else []
        current = None
        for k, w in enumerate(written, 1):
            b = w["story"]["bucket"]
            if b != current:
                current = b
                out += ["---", f"## {SECTION.get(b, b)}"]
                if openers.get(b):
                    out += [f"_\"{openers[b].strip(chr(34))}\"_", ""]
            out += _story_md(k, w, points_key, with_source)
        if show.get("special_today"):
            out += ["---", "## 🗓️ SPECIAL TODAY", show["special_today"], ""]
        if show.get("outro"):
            out += ["---", show["outro"]]
        return out

    body_sc = build("script_points", with_source=False)
    words = _words(body_sc)
    sc = [f"# {config.SHOW_NAME} — YouTube script", f"_{today} · about {words} words ≈ "
          f"{words / config.WORDS_PER_MINUTE:.0f} minutes_", ""] + body_sc
    d = [f"# {config.SHOW_NAME} — detailed news", f"_{today} · {len(written)} stories_", ""] \
        + build("detailed_points", with_source=True)
    return _tidy("\n".join(d)), _tidy("\n".join(sc)), words


# ---------------------------------------------------------------------------
# Teleprompter version (plain text)
# ---------------------------------------------------------------------------

_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\U00002B00-\U00002BFF"
    "\U0000FE0F\U0000200D\U000020E3]+")


def _plain(text: str) -> str:
    """Strip markdown symbols (and emojis unless kept) so the teleprompter shows clean words."""
    text = re.sub(r"[*_`#>]+", "", _tidy(text))
    if not config.TELEPROMPTER_KEEP_EMOJIS:
        text = _EMOJI.sub("", text)
    text = text.replace("\u2014", " - ")                     # long dash reads oddly on some prompters
    return re.sub(r"[ \t]+", " ", text).strip()


def render_teleprompter(written: list[dict], show: dict) -> str:
    """One block per story: title line, then short paragraphs, blank lines between — nothing to trip over."""
    openers = {o["topic"]: o["line"] for o in show.get("topic_openers", [])}
    names = {"World": "WORLD NEWS", "India": "INDIA NEWS", "UAE": "UAE NEWS", "Sports": "SPORTS NEWS",
             "Space & Science": "SPACE AND SCIENCE NEWS", "Tech": "TECHNOLOGY NEWS",
             "Weather & Nature": "WEATHER AND NATURE", "Business": "MONEY AND BUSINESS"}
    out: list[str] = []
    if show.get("intro"):
        out += [_plain(show["intro"]), ""]
    current = None
    for k, w in enumerate(written, 1):
        b = w["story"]["bucket"]
        if b != current:
            current = b
            out += ["", names.get(b, b.upper()), ""]
            if openers.get(b):
                out += [_plain(openers[b]).strip('"'), ""]
        out += [f"{k}. {_plain(w['title']).upper()}", ""]
        if w.get("opener"):
            out += [_plain(w["opener"]), ""]
        for p in w.get("script_points", []):
            label = _plain(p["label"]).rstrip(":")
            out += [f"{label}: {_plain(p['text'])}", ""]
        if _explainer_fits(w, "script_points"):
            term = _plain(w["explain_term"]).rstrip("?")
            out += [f"Wait, what's {term}? {_plain(w['explain_text'])}", ""]
    if show.get("special_today"):
        out += ["", "SPECIAL TODAY", "", _plain(show["special_today"]), ""]
    if show.get("outro"):
        out += ["", _plain(show["outro"])]
    text = "\n".join(out).strip() + "\n"
    return re.sub(r"\n{3,}", "\n\n", text)               # never more than one empty line


# ---------------------------------------------------------------------------
# Teleprompter file: the app drops every line break, so mark breaks with visible symbols
# ---------------------------------------------------------------------------

PARA_MARK = "   ●●●   "      # between paragraphs
TITLE_MARK = "★"             # around section and story titles


def _is_title(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    return bool(letters) and line == line.upper()


def to_markers(text: str) -> str:
    """One stream of text: ●●● between paragraphs, ★ TITLE ★ around titles — survives any import."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text.replace("\r\n", "\n")) if p.strip()]
    paras = [" ".join(p.split()) for p in paras]
    return PARA_MARK.join(f"{TITLE_MARK} {p} {TITLE_MARK}" if _is_title(p) else p for p in paras)


def save_teleprompter(folder: Path, text: str) -> None:
    (folder / "youtube_script.txt").write_text(text, encoding="utf-8")
