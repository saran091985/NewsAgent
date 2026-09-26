"""
NewsAgent review screen (Gradio).

  uv run python -m newsagent ui        → opens http://127.0.0.1:7860

1. Collect today's news   (free)
2. Pick top N with AI     (~$0.002) — ticks the picks below
3. Review: tick / untick stories per topic, add a story by link
4. Write scripts          (~$0.01)  — detailed version + YouTube script, with downloads
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr

from . import collect_step, config, select_step, write_step
from .ai_errors import AIError

BUCKETS = list(config.BUCKET_TARGETS)

CSS = """
.bucket-box .wrap {flex-direction: column; align-items: flex-start; gap: 4px;}
.bucket-box label {font-size: 0.93rem;}
#counter {position: sticky; top: 0; z-index: 10; background: var(--background-fill-primary);
          padding: 6px 0; border-bottom: 1px solid var(--border-color-primary);}
"""


# ---------------------------------------------------------------------------
# data helpers
# ---------------------------------------------------------------------------

def _load(folder: str | None) -> tuple[list[dict], list[str]]:
    """Candidates + currently chosen ids (final list if saved, else AI picks)."""
    if not folder or not (Path(folder) / "candidates.json").exists():
        return [], []
    f = Path(folder)
    rows = json.loads((f / "candidates.json").read_text(encoding="utf-8"))
    chosen: list[str] = []
    for name in ("final.json", "selected.json"):
        if (f / name).exists():
            chosen = [r["id"] for r in json.loads((f / name).read_text(encoding="utf-8"))]
            break
    if (f / "scores.json").exists():          # AI score per story, shown and used for sorting
        scores = json.loads((f / "scores.json").read_text(encoding="utf-8"))
        for r in rows:
            if r["id"] in scores:
                r["score"], r["score_why"] = scores[r["id"]]["score"], scores[r["id"]]["why"]
    return rows, chosen


def _label(r: dict) -> str:
    t = ""
    if r.get("published"):
        try:
            t = datetime.fromisoformat(r["published"]).astimezone(config.LOCAL_TZ).strftime("%H:%M") + " · "
        except ValueError:
            pass
    more = len([x for x in (r.get("also_in") or "").split(",") if x.strip()])
    extra = f"  (+{more} outlets)" if more else ""
    flag = "⚠️ " if r.get("sensitive") else ""
    score = f"[{r['score']}/10] " if r.get("score") else ""
    return f"{flag}{score}{t}{r['headline']} — {r['source']}{extra}"


def _groups(rows: list[dict], chosen: list[str], show_flagged: bool):
    """One CheckboxGroup update per topic + the ticked values."""
    out, values = [], []
    chosen_set = set(chosen)
    for b in BUCKETS:
        items = [r for r in rows if r["bucket"] == b and (show_flagged or not r["sensitive"] or r["id"] in chosen_set)]
        # ticked first, then highest AI score, then newest
        items.sort(key=lambda r: (r["id"] not in chosen_set, -(r.get("score") or 0),
                                  -(datetime.fromisoformat(r["published"]).timestamp() if r.get("published") else 0)))
        value = [r["id"] for r in items if r["id"] in chosen_set]
        values.append(value)
        out.append(gr.CheckboxGroup(choices=[(_label(r), r["id"]) for r in items], value=value,
                                    label=f"{b} — {len(items)} stories", show_label=False))
    return out, values


def _counter(top_n: int, values: list[list[str]]) -> str:
    targets = select_step.scaled_targets(int(top_n))
    parts = []
    for b, v in zip(BUCKETS, values):
        n, t = len(v or []), targets.get(b, 0)
        mark = "✅" if n == t else ("🔺" if n > t else "▫️")
        parts.append(f"{mark} {b} **{n}/{t}**")
    total = sum(len(v or []) for v in values)
    head = f"### Picked {total} of {int(top_n)}"
    return head + "\n" + " · ".join(parts)


def _status(folder: str | None) -> str:
    if not folder:
        return "No news collected yet — click **1. Collect today's news**."
    f = Path(folder)
    msg = [f"**Folder:** `{f}`"]
    if (f / "collect_log.json").exists():
        s = json.loads((f / "collect_log.json").read_text())
        msg.append(f"Collected {s['collected']} → in last 24 h {s['in_window']} → kept {s['after_category_filter']} "
                   f"→ unique **{s['after_dedupe']}** ({s['sensitive_flagged']} flagged ⚠️)")
    if (f / "select_log.json").exists():
        s = json.loads((f / "select_log.json").read_text())
        cost = f", cost ≈ ${s['cost_usd']}" if "cost_usd" in s else ""
        err = f" — ⚠️ AI error: {s['error'][:120]}" if s.get("error") else ""
        msg.append(f"AI picked {s.get('from_ai', 0)}, rule-based {s.get('from_fallback', 0)}{cost}{err}")
    return "  \n".join(msg)


def _latest_folder() -> str | None:
    try:
        return str(select_step.latest_run_dir())
    except SystemExit:
        return None


# ---------------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------------

def on_load(show_flagged, top_n):
    folder = _latest_folder()
    rows, chosen = _load(folder)
    groups, values = _groups(rows, chosen, show_flagged)
    return [folder, _status(folder), _counter(top_n, values), *groups]


def on_collect(hours, show_flagged, top_n, progress=gr.Progress()):
    progress(0.1, desc="Collecting from The Hindu, BBC, NASA, Gulf News, Khaleej Times…")
    _, folder = collect_step.run(hours=int(hours))
    folder = str(folder)
    rows, _ = _load(folder)
    groups, values = _groups(rows, [], show_flagged)
    return [folder, _status(folder), _counter(top_n, values), *groups]


def on_pick(folder, top_n, no_ai, show_flagged, progress=gr.Progress()):
    if not folder:
        raise gr.Error("Collect the news first.")
    progress(0.3, desc=f"Picking the top {int(top_n)}…")
    try:
        select_step.run(n=int(top_n), folder=Path(folder), use_ai=not no_ai)
    except AIError as e:
        # Show the reason and leave the page exactly as it was (no silent free pick)
        msg = f"{e} Your ticks were not changed. To pick without AI, tick 'Free pick (no AI)'."
        gr.Warning(msg, duration=None, title="AI pick failed")
        return [_status(folder) + f"  \n❌ **AI pick failed:** {msg}", gr.skip(), *[gr.skip() for _ in BUCKETS]]
    final = Path(folder) / "final.json"
    if final.exists():   # the new pick replaces your earlier saved list (kept as final.previous.json)
        final.replace(final.with_suffix(".previous.json"))
    rows, chosen = _load(folder)
    groups, values = _groups(rows, chosen, show_flagged)
    return [_status(folder), _counter(top_n, values), *groups]


def on_toggle_flagged(folder, show_flagged, top_n, *values):
    rows, _ = _load(folder)
    chosen = [i for v in values for i in (v or [])]
    groups, values = _groups(rows, chosen, show_flagged)
    return [_counter(top_n, values), *groups]


def on_add(folder, headline, url, bucket, show_flagged, top_n, *values):
    if not folder:
        raise gr.Error("Collect the news first.")
    if not headline.strip():
        raise gr.Error("Type a headline.")
    f = Path(folder) / "candidates.json"
    rows = json.loads(f.read_text(encoding="utf-8"))
    sid = hashlib.sha1((url or headline).encode()).hexdigest()[:6]
    if all(r["id"] != sid for r in rows):
        rows.append({"headline": headline.strip(), "url": url.strip(), "source": "Added by you",
                     "published": datetime.now(timezone.utc).isoformat(), "category": None, "summary": None,
                     "bucket": bucket, "sensitive": False, "also_in": "", "origin": "manual", "id": sid})
        f.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    chosen = [i for v in values for i in (v or [])] + [sid]
    groups, values = _groups(rows, chosen, show_flagged)
    return ["", "", _counter(top_n, values), *groups]


def _save_final(folder: str, ids: list[str]) -> list[dict]:
    rows, _ = _load(folder)
    by_id = {r["id"]: r for r in rows}
    old = {}
    sel = Path(folder) / "selected.json"
    if sel.exists():
        old = {r["id"]: r.get("why", "") for r in json.loads(sel.read_text(encoding="utf-8"))}
    stories = [by_id[i] for i in ids if i in by_id]
    for k, s in enumerate(stories, 1):
        s["rank"], s["why"] = k, old.get(s["id"], "picked by you")
    (Path(folder) / "final.json").write_text(json.dumps(stories, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(Path(folder) / "final.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["rank", "bucket", "headline", "source", "url", "why"], extrasaction="ignore")
        w.writeheader()
        w.writerows(stories)
    return stories


def on_save(folder, *values):
    if not folder:
        raise gr.Error("Collect the news first.")
    ids = [i for v in values for i in (v or [])]   # topic order = show order
    stories = _save_final(folder, ids)
    gr.Info(f"Saved {len(stories)} stories to final.csv")
    return str(Path(folder) / "final.csv")


def on_write(folder, *values, progress=gr.Progress()):
    if not folder:
        raise gr.Error("Collect the news first.")
    ids = [i for v in values for i in (v or [])]
    if not ids:
        raise gr.Error("Tick at least one story.")
    stories = _save_final(folder, ids)
    try:
        res = write_step.run(Path(folder), stories, progress=progress)
    except AIError as e:
        msg = f"{e} Your list was saved to final.csv."
        gr.Warning(msg, duration=None, title="Writing failed")
        return gr.skip(), gr.skip(), f"❌ **Writing failed:** {msg}", gr.skip()
    log = res["log"]
    notes = [f"**{log['script_words']} words ≈ {log['script_minutes']} minutes** · cost ≈ ${log['cost_usd']}"]
    if log["errors"]:
        notes.append("⚠️ Could not write: " + "; ".join(log["errors"]))
    if log["number_warnings"]:
        notes.append("🔎 **Check these numbers** — they are not in the article text:\n" +
                     "\n".join(f"- {h[:70]} → {', '.join(n)}" for h, n in log["number_warnings"].items()))
    if log["thin_source"]:
        notes.append(f"ℹ️ {len(log['thin_source'])} stories had little article text (written from headline/summary) — double-check them.")
    f = Path(folder)
    files = [str(f / "youtube_script.md"), str(f / "news_detailed.md"), str(f / "final.csv")]
    return res["script"], res["detailed"], "\n\n".join(notes), files


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------

def build() -> gr.Blocks:
    with gr.Blocks(title="NewsAgent — Kids News") as demo:
        folder = gr.State(None)
        gr.Markdown("# 📰 NewsAgent — today's news for the kids' show")

        with gr.Row():
            top_n = gr.Slider(5, 40, value=config.TOP_N, step=1, label="How many stories (top N)")
            hours = gr.Slider(6, 48, value=config.WINDOW_HOURS, step=1, label="News from the last … hours")
            with gr.Column(min_width=160):
                no_ai = gr.Checkbox(False, label="Free pick (no AI)")
                show_flagged = gr.Checkbox(False, label="Show ⚠️ flagged stories")
        with gr.Row():
            b_collect = gr.Button("1. Collect today's news", variant="secondary")
            b_pick = gr.Button("2. Pick top N with AI", variant="secondary")
        status = gr.Markdown()

        with gr.Tabs():
            with gr.Tab("3. Review stories"):
                counter = gr.Markdown(elem_id="counter")
                groups = []
                for b in BUCKETS:
                    with gr.Accordion(b, open=True):
                        groups.append(gr.CheckboxGroup(choices=[], label=b, show_label=False,
                                                       elem_classes="bucket-box"))
                with gr.Accordion("➕ Add a story you found elsewhere", open=False):
                    with gr.Row():
                        add_head = gr.Textbox(label="Headline", scale=3)
                        add_url = gr.Textbox(label="Link", scale=3)
                        add_bucket = gr.Dropdown(BUCKETS, value="Space & Science", label="Topic", scale=1)
                    b_add = gr.Button("Add and tick it")
                with gr.Row():
                    b_save = gr.Button("💾 Save list only")
                    b_write = gr.Button("4. ✍️ Write scripts for the ticked stories", variant="primary")
                saved_file = gr.File(label="final.csv", visible=True)
            with gr.Tab("YouTube script"):
                notes = gr.Markdown()
                script_md = gr.Markdown()
            with gr.Tab("Detailed version"):
                detailed_md = gr.Markdown()
            with gr.Tab("Downloads"):
                files = gr.Files(label="Today's files")

        # wiring
        demo.load(on_load, [show_flagged, top_n], [folder, status, counter, *groups])
        b_collect.click(on_collect, [hours, show_flagged, top_n], [folder, status, counter, *groups])
        b_pick.click(on_pick, [folder, top_n, no_ai, show_flagged], [status, counter, *groups])
        show_flagged.change(on_toggle_flagged, [folder, show_flagged, top_n, *groups], [counter, *groups])
        for g in groups:
            g.change(lambda n, *v: _counter(n, list(v)), [top_n, *groups], counter)
        top_n.change(lambda n, *v: _counter(n, list(v)), [top_n, *groups], counter)
        b_add.click(on_add, [folder, add_head, add_url, add_bucket, show_flagged, top_n, *groups],
                    [add_head, add_url, counter, *groups])
        b_save.click(on_save, [folder, *groups], saved_file)
        b_write.click(on_write, [folder, *groups], [script_md, detailed_md, notes, files])
    return demo


def main(port: int = 7860, share: bool = False):
    build().launch(server_name="127.0.0.1", server_port=port, share=share, css=CSS,
                   theme=gr.themes.Soft(), inbrowser=True)
