"""
Step 2: pick the TOP_N most important stories from today's candidates.

One gpt-4o-mini call reads the headlines (about 8k tokens, roughly $0.002)
and returns the picks with a one-line reason each. Code then checks the
answer (real ids, no repeats, right count) and tops it up without AI if needed.
Run with --no-ai to use only the free rule-based ranking.

Output (same folder as candidates):
  selected.json / selected.csv — the picks, in show order
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
# AI pick
# ---------------------------------------------------------------------------

class Pick(BaseModel):
    id: str = Field(description="the story id exactly as given")
    why: str = Field(description="max 12 words: why this matters or will interest kids")


class Picks(BaseModel):
    picks: list[Pick] = Field(description="most important first")


SYSTEM = """You are the editor of a daily YouTube news show for children, presented by a 10-year-old host \
for viewers aged 8-14 in India and the UAE. From today's headlines, choose the most important stories.

What makes a story important for this show:
- big events many people are talking about today (reported by several outlets ranks higher)
- things that change daily life, or that kids would find exciting: space, science, discoveries,
  technology, sports wins and records, nature and weather, animals, inventions
- major world and India events explained simply (summits, elections results, disasters, big decisions)
- UAE news that families in the UAE would care about

Avoid: crime, violence details, political name-calling or party fights, court cases, gossip,
advertisements, lifestyle features, stock-price or company-earnings notes, opinion pieces,
and the same event twice (pick the best headline for it).

Follow the topic targets closely. If a topic has no good story, give the slot to another topic."""


def _prompt(rows: list[dict], n: int) -> str:
    targets = ", ".join(f"{b} {t}" for b, t in scaled_targets(n).items())
    lines = []
    for r in rows:
        if r["sensitive"]:
            continue
        others = len([x for x in (r.get("also_in") or "").split(",") if x.strip()])
        cover = f" (+{others} outlets)" if others else ""
        lines.append(f'{r["id"]} | {r["bucket"]} | {r["source"]}{cover} | {r["headline"]}')
    return (f"Choose exactly {n} stories.\nTopic targets: {targets}.\n\n"
            "Headlines (id | topic | source | headline):\n" + "\n".join(lines))


def ai_pick(rows: list[dict], n: int) -> tuple[list[dict], dict]:
    from dotenv import load_dotenv
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    load_dotenv(override=True)
    llm = ChatOpenAI(model=config.SELECT_MODEL, temperature=0).with_structured_output(
        Picks, include_raw=True, method="function_calling")
    res = llm.invoke([SystemMessage(content=SYSTEM), HumanMessage(content=_prompt(rows, n))])
    usage = getattr(res["raw"], "usage_metadata", None) or {}
    tin, tout = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    log = {"model": config.SELECT_MODEL, "input_tokens": tin, "output_tokens": tout,
           "cost_usd": round((tin * PRICE_IN + tout * PRICE_OUT) / 1e6, 5)}
    parsed: Picks | None = res.get("parsed")
    if parsed is None:
        log["error"] = str(res.get("parsing_error"))
        return [], log
    return [p.model_dump() for p in parsed.picks], log


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
            picks, ai_log = ai_pick(rows, n)
        except Exception as e:
            raise AIError(friendly(e)) from e
        log.update(ai_log)
        if not picks:
            raise AIError(f"The AI answer could not be read ({ai_log.get('error', 'empty answer')[:200]}). "
                          "Try again, or tick 'Free pick (no AI)'.")
        for p in picks:
            if p["id"] in by_id and p["id"] not in ids and len(ids) < n:
                ids.append(p["id"])
                why[p["id"]] = p["why"]
    log["from_ai"] = len(ids)
    ids = rule_based_pick(rows, n, already=ids)
    log["from_fallback"] = len(ids) - log["from_ai"]

    order = list(config.BUCKET_TARGETS)
    ai_rank = {i: k for k, i in enumerate(ids)}
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
