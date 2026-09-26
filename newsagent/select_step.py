"""
Step 2: pick the TOP_N most important stories from today's candidates.

gpt-4o-mini scores every headline 1-10 in ONE call (SELECT_MODE = "single"),
or one call per topic (SELECT_MODE = "per_topic"). Code then takes the best-scored
stories per topic target, skipping near-duplicates. Run with --no-ai to use
only the free rule-based ranking.

Output (same folder as candidates):
  selected.json / selected.csv — the picks, in show order
  scores.json                  — AI score + reason for every story (shown on the review screen)
  select_log.json              — model, tokens, cost, how many came from AI vs fallback
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from . import config
from .ai_errors import AIError, friendly

TRUSTED_SOURCES = {"The Hindu", "BBC World", "BBC Science", "BBC Technology", "NASA",
                   "Khaleej Times", "Gulf News", "Space", "Space.com"}

# gpt-4o-mini list prices per 1M tokens (check openai.com/api/pricing if they change)
PRICE_IN, PRICE_OUT = 0.15, 0.60


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def scaled_targets(n: int) -> dict[str, int]:
    """Scale BUCKET_TARGETS (written for 20 stories) to n, keeping the total exactly n."""
    base = config.BUCKET_TARGETS
    total = sum(base.values())
    raw = {b: v * n / total for b, v in base.items()}
    out = {b: int(x) for b, x in raw.items()}
    for b in sorted(raw, key=lambda b: raw[b] - out[b], reverse=True)[: n - sum(out.values())]:
        out[b] += 1
    return out


# ---------------------------------------------------------------------------
# Free fallback ranking
# ---------------------------------------------------------------------------

def _score(row: dict) -> float:
    s = 0.0
    s += 2 * len([x for x in (row.get("also_in") or "").split(",") if x.strip()])  # covered by several outlets
    s += 1 if row["source"] in TRUSTED_SOURCES else 0
    s += 0.5 if row.get("summary") else 0
    return s


def rule_based_pick(rows: list[dict], n: int, already: list[str] | None = None) -> list[str]:
    """Pick ids by topic target, best score first; spare slots go to the best leftovers."""
    chosen = list(already or [])
    pool = [r for r in rows if not r["sensitive"] and r["id"] not in chosen]
    pool.sort(key=_score, reverse=True)
    counts = {b: sum(1 for r in rows if r["id"] in chosen and r["bucket"] == b) for b in config.BUCKET_TARGETS}
    for b, target in scaled_targets(n).items():
        for r in [r for r in pool if r["bucket"] == b]:
            if counts[b] >= target or len(chosen) >= n:
                break
            chosen.append(r["id"])
            counts[b] += 1
    for r in pool:
        if len(chosen) >= n:
            break
        if r["id"] not in chosen:
            chosen.append(r["id"])
    return chosen[:n]


# ---------------------------------------------------------------------------
# AI scoring
# ---------------------------------------------------------------------------
# Instead of asking for "the top 20" out of ~230 headlines in one go (the model
# lost track of ids and ignored the topic mix), the AI now SCORES every headline
# 1-10, one topic at a time (8 small calls run in parallel). Code then takes the
# best-scored stories per topic, so the topic mix is always respected and each
# score can be shown on the review screen.

class Score(BaseModel):
    n: int = Field(description="the story number exactly as given")
    score: int = Field(description="1-10, how right this story is for today's kids' show")
    why: str = Field(default="", description="max 8 words; only for scores 7-10, otherwise empty")


class Scores(BaseModel):
    scores: list[Score]


SYSTEM = """You are the editor of a daily YouTube news show for children aged 8-14 in India and the UAE, \
presented by a 10-year-old host. The show explains the REAL big news of the day simply — it does not
avoid serious world events. You score today's headlines for the show.

Score 9-10: the biggest stories of the day — the ones on every front page:
  - major world events and turning points: wars and conflicts moving forward (big attacks, drone or
    missile strikes, ceasefires, blockades like the Strait of Hormuz), peace or trade deals, summits,
    big decisions by governments, anything that affects India's or the UAE's relations with the world
  - natural disasters and weather emergencies
  - space missions, launches, discoveries; big science findings
  - big sports wins, medals, records
  - exciting things kids care about: new video games, gadgets or tech launches, animals, rescues
Score 6-8: clear, important news that is easy to explain.
Score 3-5: soft or feel-good features (crowds, travel, "places disappearing"), minor or local news,
  very technical news, or news only adults care about.
Score 1-2: crime and court cases about individuals, scams, deaths of private people, gossip,
  political name-calling or party fights (one politician attacking another), speeches with no decision,
  birthday tributes, religious events, adverts and sales, property, stock prices, company earnings,
  opinion or lifestyle pieces, live blogs, job ads.

"covered N times" means N headlines today are about this same event — the higher N, the bigger the story.
If several headlines are about the same event, give the clearest one the high score and the others at most 3.
Learn from the editor's past choices: stories like the "picked" examples score high, like the "rejected" ones low."""


def _feedback_examples() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(topic, headline) the editor picked / rejected: config lists + recent feedback files."""
    picks, rejects = list(config.EDITOR_PICKS), list(config.EDITOR_REJECTS)
    if config.LEARN_FROM_MY_PICKS:
        for f in sorted(Path("output").glob("20??-??-??/feedback.json"))[-config.FEEDBACK_DAYS:]:
            try:
                fb = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            picks += [(r["bucket"], r["headline"]) for r in fb.get("picked", [])]
            rejects += [(r["bucket"], r["headline"]) for r in fb.get("rejected", [])]
    return list(dict.fromkeys(picks)), list(dict.fromkeys(rejects))


def _examples_for(bucket: str) -> str:
    picks, rejects = _feedback_examples()
    same = [h for b, h in picks if b == bucket][-15:]
    other = [f"({b}) {h}" for b, h in picks if b != bucket][-8:]
    rej = [h for b, h in rejects if b == bucket][-12:] + [f"({b}) {h}" for b, h in rejects if b != bucket][-4:]
    out = ["Stories the editor PICKED on earlier days (for taste, not for today):"]
    out += [f"- {h}" for h in same + other] or ["- (none yet)"]
    if rej:
        out += ["Stories the editor REJECTED on earlier days:"] + [f"- {h}" for h in rej]
    return "\n".join(out)


def coverage(rows: list[dict]) -> dict[str, int]:
    """How many of today's headlines are about the same event (1 = only this one)."""
    from .filter import _stems, same_event
    count = {r["id"]: 1 + len([x for x in (r.get("also_in") or "").split(",") if x.strip()]) for r in rows}
    stems = [_stems(r["headline"]) for r in rows]
    for i, a in enumerate(rows):
        for j in range(i + 1, len(rows)):
            if not stems[i] & stems[j]:          # no shared word → cannot be the same event (fast skip)
                continue
            b = rows[j]
            if same_event(a["headline"], b["headline"]):
                count[a["id"]] += 1
                count[b["id"]] += 1
    return count


def _bucket_prompt(bucket: str, items: list[tuple[int, dict]], target: int, cover: dict[str, int]) -> str:
    lines = []
    for num, r in items:
        c = cover.get(r["id"], 1)
        tag = f" (covered {c} times)" if c > 1 else ""
        lines.append(f"{num} | {r['source']}{tag} | {r['headline']}")
    return (f"Topic: {bucket}. The show needs about {target} story(ies) from this topic today.\n\n"
            f"{_examples_for(bucket)}\n\n"
            f"Score EVERY headline below (number | source | headline):\n" + "\n".join(lines))


def _all_examples() -> str:
    picks, rejects = _feedback_examples()
    out = ["Stories the editor PICKED on earlier days (for taste, not for today):"]
    out += [f"- ({b}) {h}" for b, h in picks[-30:]] or ["- (none yet)"]
    if rejects:
        out += ["Stories the editor REJECTED on earlier days:"] + [f"- ({b}) {h}" for b, h in rejects[-20:]]
    return "\n".join(out)


def _single_prompt(number: dict[int, dict], targets: dict[str, int], cover: dict[str, int]) -> str:
    parts = [f"Today's show needs {sum(targets.values())} stories: "
             + ", ".join(f"{b} {t}" for b, t in targets.items()) + ".",
             "", _all_examples(), "",
             "Score EVERY headline below, grouped by topic (number | source | headline).",
             "Give a short 'why' only for scores 7-10; leave it empty for the rest."]
    for b in config.BUCKET_TARGETS:
        items = [(k, r) for k, r in number.items() if r["bucket"] == b]
        if not items:
            continue
        parts.append(f"\n## {b}")
        for k, r in items:
            c = cover.get(r["id"], 1)
            tag = f" (covered {c} times)" if c > 1 else ""
            parts.append(f"{k} | {r['source']}{tag} | {r['headline']}")
    return "\n".join(parts)


def ai_scores(rows: list[dict], n: int) -> tuple[dict[str, dict], dict]:
    """Return {story id: {"score": int, "why": str}} for every non-flagged story."""
    from dotenv import load_dotenv
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    load_dotenv(override=True)
    llm = ChatOpenAI(model=config.SELECT_MODEL, temperature=0).with_structured_output(
        Scores, include_raw=True, method="function_calling")
    targets = scaled_targets(n)
    pool = [r for r in rows if not r["sensitive"]]
    number = {k: r for k, r in enumerate(pool, 1)}          # short numbers are far easier for the model than hex ids
    cover = coverage(pool)
    if config.SELECT_MODE == "per_topic":
        # one call per topic, run in parallel (more calls, each sees fewer headlines)
        buckets = [b for b in config.BUCKET_TARGETS if any(r["bucket"] == b for r in pool)]
        prompts = []
        for b in buckets:
            items = [(k, r) for k, r in number.items() if r["bucket"] == b]
            prompts.append([SystemMessage(content=SYSTEM),
                            HumanMessage(content=_bucket_prompt(b, items, max(targets.get(b, 1), 1), cover))])
        results = llm.batch(prompts, config={"max_concurrency": 8}, return_exceptions=True)
    else:
        # one call for all headlines (default)
        buckets = [None]
        prompts = [[SystemMessage(content=SYSTEM), HumanMessage(content=_single_prompt(number, targets, cover))]]
        try:
            results = [llm.invoke(prompts[0], config={"run_name": "Pick top N"})]
        except Exception as e:
            results = [e]

    failures = [r for r in results if isinstance(r, Exception)]
    if failures and len(failures) == len(results):
        raise failures[0]
    scores: dict[str, dict] = {}
    tin = tout = 0
    errors = {}
    for b, res in zip(buckets, results):
        if isinstance(res, Exception) or res.get("parsed") is None:
            errors[b or "all"] = str(res if isinstance(res, Exception) else res.get("parsing_error"))[:200]
            continue
        u = getattr(res["raw"], "usage_metadata", None) or {}
        tin += u.get("input_tokens", 0)
        tout += u.get("output_tokens", 0)
        for s in res["parsed"].scores:
            r = number.get(s.n)
            if r is not None and (b is None or r["bucket"] == b):   # per-topic: ignore numbers from another topic
                scores[r["id"]] = {"score": max(1, min(10, s.score)), "why": s.why}
    log = {"model": config.SELECT_MODEL, "calls": len(prompts), "input_tokens": tin, "output_tokens": tout,
           "cost_usd": round((tin * PRICE_IN + tout * PRICE_OUT) / 1e6, 5),
           "scored": len(scores), "not_scored": len(pool) - len(scores)}
    if errors:
        log["errors"] = errors
    return scores, log


def pick_by_score(rows: list[dict], scores: dict[str, dict], n: int) -> list[str]:
    """Best-scored stories per topic target; no near-duplicates; spare slots go to the best leftovers."""
    from .filter import same_event
    min_score = config.MIN_PICK_SCORE
    ranked = sorted((r for r in rows if r["id"] in scores and scores[r["id"]]["score"] >= min_score),
                    key=lambda r: (scores[r["id"]]["score"], _score(r)), reverse=True)
    chosen: list[dict] = []

    def ok(r):
        return all(not same_event(r["headline"], c["headline"]) for c in chosen)

    for b, target in scaled_targets(n).items():
        for r in [r for r in ranked if r["bucket"] == b]:
            if sum(c["bucket"] == b for c in chosen) >= target:
                break
            if ok(r):
                chosen.append(r)
    everything = sorted((r for r in rows if r["id"] in scores),
                        key=lambda r: (scores[r["id"]]["score"], _score(r)), reverse=True)
    targets = scaled_targets(n)
    for cap in (1, n):                    # spare slots → best leftovers, at most 1 extra per topic first
        for r in everything:
            if len(chosen) >= n:
                break
            over = sum(c["bucket"] == r["bucket"] for c in chosen) - targets.get(r["bucket"], 0)
            if r not in chosen and over < cap and ok(r):
                chosen.append(r)
    return [r["id"] for r in chosen[:n]]


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def latest_run_dir() -> Path:
    dirs = sorted(p for p in Path("output").glob("20??-??-??") if (p / "candidates.json").exists())
    if not dirs:
        raise SystemExit("No candidates found — run `python -m newsagent collect` first.")
    return dirs[-1]


def run(n: int | None = None, folder: Path | None = None, use_ai: bool = True) -> list[dict]:
    n = n or config.TOP_N
    folder = folder or latest_run_dir()
    rows = json.loads((folder / "candidates.json").read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in rows}
    print(f"Picking top {n} from {len(rows)} candidates in {folder}")

    why: dict[str, str] = {}
    log: dict = {"top_n": n, "candidates": len(rows), "targets": scaled_targets(n)}
    ids: list[str] = []
    if use_ai:
        # If the AI fails, stop and say why — never swap in the rule-based pick silently.
        try:
            scores, ai_log = ai_scores(rows, n)
        except Exception as e:
            raise AIError(friendly(e)) from e
        log.update(ai_log)
        if not scores:
            raise AIError("The AI answer could not be read. Try again, or tick 'Free pick (no AI)'.")
        (folder / "scores.json").write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
        ids = pick_by_score(rows, scores, n)
        why = {i: f"{scores[i]['score']}/10 · {scores[i]['why']}" for i in ids}
    log["from_ai"] = len(ids)
    if len(ids) < n:
        ids = rule_based_pick(rows, n, already=ids)
    log["from_fallback"] = len(ids) - log["from_ai"]

    order = list(config.BUCKET_TARGETS)
    ai_rank = {i: k for k, i in enumerate(ids)}   # within a topic: highest score first
    selected = sorted((by_id[i] for i in ids),
                      key=lambda r: (order.index(r["bucket"]) if r["bucket"] in order else 99, ai_rank[r["id"]]))
    for k, r in enumerate(selected, 1):
        r["rank"] = k
        r["why"] = why.get(r["id"], "rule-based pick" if not use_ai else "added by rules (AI returned fewer)")

    (folder / "selected.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(folder / "selected.csv", "w", newline="", encoding="utf-8-sig") as f:
        cols = ["rank", "bucket", "headline", "why", "source", "also_in", "published", "url", "id"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(selected)
    log["run_at"] = datetime.now().isoformat(timespec="seconds")
    (folder / "select_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    print()
    for r in selected:
        print(f'{r["rank"]:>2}. [{r["bucket"]}] {r["headline"]}  — {r["source"]}')
    if "cost_usd" in log:
        print(f'\nAI: {log["from_ai"]} picks, {log["input_tokens"]}+{log["output_tokens"]} tokens ≈ ${log["cost_usd"]}')
    if log["from_fallback"]:
        print(f'Rule-based fill: {log["from_fallback"]} picks')
    print(f"Saved {folder / 'selected.csv'}")
    return selected
