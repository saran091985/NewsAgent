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

from . import collect_step, config, images_step, select_step, write_step
from .ai_errors import AIError

BUCKETS = list(config.BUCKET_TARGETS)

TOPIC_COLORS = {
    "World": "#3b82f6", "India": "#f97316", "UAE": "#16a34a", "Sports": "#ef4444",
    "Space & Science": "#8b5cf6", "Tech": "#06b6d4", "Weather & Nature": "#14b8a6", "Business": "#eab308",
}
TOPIC_ICONS = {
    "World": "🌍", "India": "🇮🇳", "UAE": "🇦🇪", "Sports": "🏆", "Space & Science": "🚀",
    "Tech": "💻", "Weather & Nature": "🌦️", "Business": "💰",
}


def _slug(b: str) -> str:
    return "".join(ch for ch in b.lower() if ch.isalnum())


CSS = """
.gradio-container {max-width: 1280px !important; width: 100% !important; margin: auto;}
body, .gradio-container {background: linear-gradient(180deg, #fdf4ff 0%, #eff6ff 40%, #f0fdfa 100%) !important;}
.dark body, .dark .gradio-container {background: #0f172a !important;}

/* header banner */
#hero {border-radius: 22px; padding: 22px 28px; color: #fff; margin-bottom: 6px;
       background: linear-gradient(120deg, #7c3aed 0%, #db2777 45%, #f59e0b 100%);
       box-shadow: 0 10px 30px rgba(124,58,237,.25);}
#hero h1 {margin: 0; font-size: 2rem; color: #fff;}
#hero p {margin: 6px 0 0; opacity: .95; font-size: 1.02rem;}
#hero .steps span {display: inline-block; background: rgba(255,255,255,.22); border-radius: 999px;
                   padding: 3px 12px; margin: 10px 6px 0 0; font-weight: 600; font-size: .9rem;}

/* panels */
#controls {border-radius: 18px; background: rgba(255,255,255,.75); padding: 8px; box-shadow: 0 4px 18px rgba(0,0,0,.05);}
.dark #controls {background: rgba(30,41,59,.7);}

/* step buttons */
.step-btn button, button.step-btn {border: none !important; color: #fff !important; font-weight: 700 !important;
      font-size: 1.05rem !important; border-radius: 14px !important; min-height: 52px;
      box-shadow: 0 6px 16px rgba(0,0,0,.12); transition: transform .08s ease;}
.step-btn:hover {transform: translateY(-1px);}
#btn-collect {background: linear-gradient(90deg, #0ea5e9, #6366f1) !important;}
#btn-pick {background: linear-gradient(90deg, #a855f7, #ec4899) !important;}
#btn-write {background: linear-gradient(90deg, #f97316, #ef4444) !important;}
#btn-save {background: linear-gradient(90deg, #10b981, #14b8a6) !important;}
#btn-images {background: linear-gradient(90deg, #0ea5e9, #22c55e) !important;}
/* long prompts wrap instead of scrolling sideways */
.prose pre, .prose pre code, .img-card pre, .img-card code {white-space: pre-wrap !important; word-break: break-word;}
.img-card {border-radius: 16px !important; padding: 10px 14px !important; margin-bottom: 12px;
            box-shadow: 0 4px 14px rgba(0,0,0,.06); border-left: 8px solid #22c55e !important;}
#btn-add {background: linear-gradient(90deg, #64748b, #334155) !important;}
button:disabled {filter: grayscale(.4); opacity: .8; cursor: progress !important;}

/* status + counter */
#status {border-radius: 14px; padding: 4px 14px; background: rgba(255,255,255,.7);}
.dark #status {background: rgba(30,41,59,.7);}
#counter {position: sticky; top: 0; z-index: 10; padding: 8px 12px; border-radius: 14px;
          background: rgba(255,255,255,.92); box-shadow: 0 4px 14px rgba(0,0,0,.06);}
.dark #counter {background: rgba(15,23,42,.92);}
#write-status {min-height: 8px;}
#teleprompter textarea {font-size: 1.15rem !important; line-height: 1.6 !important;}
.pwrap {margin: 6px 0 4px;}
.ptext {font-weight: 700; margin-bottom: 6px;}
.pbar {height: 16px; border-radius: 999px; background: rgba(148,163,184,.25); overflow: hidden;}
.pfill {height: 100%; border-radius: 999px; transition: width .4s ease;}

/* topic cards */
.topic {border-radius: 16px !important; border: none !important; overflow: hidden;
        border-left: 8px solid var(--tc) !important; box-shadow: 0 4px 14px rgba(0,0,0,.06);
        background: linear-gradient(90deg, color-mix(in srgb, var(--tc) 10%, transparent), transparent 60%) !important;}
.topic > button, .topic .label-wrap {font-weight: 800 !important; font-size: 1.1rem !important; color: var(--tc) !important;}
.bucket-box .wrap {flex-direction: column; align-items: flex-start; gap: 5px;}
.bucket-box label {font-size: 0.95rem; border-radius: 12px !important; transition: background .1s;}
.bucket-box label:hover {background: color-mix(in srgb, var(--tc) 12%, transparent) !important;}
.bucket-box label.selected {background: var(--tc) !important; color: #fff !important; border-color: var(--tc) !important;}
.bucket-box label.selected span {color: #fff !important;}
""" + "\n".join(f".topic-{_slug(b)} {{--tc: {c};}}" for b, c in TOPIC_COLORS.items())

HERO = f"""
<div id="hero">
  <h1>📰 {config.SHOW_NAME} — News Studio</h1>
  <p>Today's real news, picked and written for kids aged 8–14.</p>
  <div class="steps"><span>1 · Collect</span><span>2 · Pick</span><span>3 · Review</span><span>4 · Write</span></div>
</div>
"""

THEME = gr.themes.Soft(
    primary_hue="violet", secondary_hue="pink", neutral_hue="slate",
    radius_size="lg", font=[gr.themes.GoogleFont("Nunito"), "ui-sans-serif", "sans-serif"],
)


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
        parts.append(f"{mark} {TOPIC_ICONS.get(b, '')} {b} **{n}/{t}**")
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


def _save_feedback(folder: str, by_id: dict, ai_ids: set, my_ids: set) -> None:
    """Remember what you kept, added and removed compared with the AI — the AI learns from it next time."""
    def rows(ids):
        return [{"bucket": by_id[i]["bucket"], "headline": by_id[i]["headline"]} for i in ids if i in by_id]
    fb = {"saved_at": datetime.now().isoformat(timespec="seconds"),
          "picked": rows(my_ids),                       # everything in your final list
          "added": rows(my_ids - ai_ids),               # you ticked, the AI had not
          "rejected": rows(ai_ids - my_ids)}            # the AI ticked, you removed
    (Path(folder) / "feedback.json").write_text(json.dumps(fb, ensure_ascii=False, indent=2), encoding="utf-8")


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
    log = Path(folder) / "select_log.json"
    ai_made = log.exists() and json.loads(log.read_text()).get("from_ai", 0) > 0
    _save_feedback(folder, by_id, set(old) if ai_made else set(ids), set(ids))
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
    return gr.File(value=str(Path(folder) / "final.csv"), visible=True)


def _bar(frac: float, text: str, color: str = "linear-gradient(90deg,#f97316,#ec4899,#8b5cf6)") -> str:
    pct = max(3, min(100, int(frac * 100)))
    return (f'<div class="pwrap"><div class="ptext">{text} <b>{pct}%</b></div>'
            f'<div class="pbar"><div class="pfill" style="width:{pct}%;background:{color}"></div></div></div>')


def on_write(folder, *values):
    """Generator: shows a live progress bar under the button while the scripts are written."""
    import threading
    import time
    skip = gr.skip()
    if not folder:
        raise gr.Error("Collect the news first.")
    ids = [i for v in values for i in (v or [])]
    if not ids:
        raise gr.Error("Tick at least one story.")
    stories = _save_final(folder, ids)

    state = {"frac": 0.0, "desc": "Starting…", "res": None, "err": None}

    def progress(frac, desc=""):
        state["frac"], state["desc"] = frac, desc

    def work():
        try:
            state["res"] = write_step.run(Path(folder), stories, progress=progress)
        except Exception as e:      # AIError or anything unexpected
            state["err"] = e

    t = threading.Thread(target=work, daemon=True)
    t.start()
    while t.is_alive():
        yield skip, skip, skip, skip, _bar(state["frac"], f"✍️ {state['desc']}"), skip, skip
        time.sleep(0.5)

    if state["err"] is not None:
        msg = f"{state['err']} Your list was saved to final.csv."
        gr.Warning(msg, duration=None, title="Writing failed")
        yield skip, skip, f"❌ **Writing failed:** {msg}", skip, \
            _bar(1, f"❌ Writing failed — {msg}", "#ef4444"), skip, skip
        return
    res = state["res"]
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
    files = [str(f / "youtube_script.docx"), str(f / "youtube_script.rtf"), str(f / "youtube_script.txt"),
             str(f / "youtube_script.md"), str(f / "news_detailed.md"),
             str(f / "final.csv")]
    done = _bar(1, f"✅ Done! {log['script_words']} words ≈ {log['script_minutes']} minutes — see the 🎬 YouTube script tab.",
                "linear-gradient(90deg,#10b981,#14b8a6)")
    yield res["script"], res["detailed"], "\n\n".join(notes), files, done, gr.Tabs(selected="script"), \
        res["teleprompter"]


# ---------------------------------------------------------------------------
# images (only when you click the button)
# ---------------------------------------------------------------------------

def on_images(folder, source, upload, per_story, creative_commons):
    """Search + download only (no AI). Prompts are written per story with the ✨ buttons."""
    import threading
    import time
    skip = gr.skip()
    if source == "Upload a file":
        if not upload:
            raise gr.Error("Choose a file to upload first (final.csv, final.json, youtube_script.txt/.md or a .txt list).")
        topics = images_step.topics_from_file(upload if isinstance(upload, str) else upload.name)
        target = collect_step.run_dir(datetime.now(config.LOCAL_TZ))
    else:
        if not folder:
            raise gr.Error("No news collected yet today — collect, pick and save a list first, or upload a file.")
        target = Path(folder)
        topics = images_step.topics_from_today(target)
        if not topics:
            raise gr.Error("Today's folder has no saved list yet — click 💾 Save list only or ✍️ Write scripts first.")
    if not topics:
        raise gr.Error("No topics found in that file.")

    state = {"frac": 0.0, "desc": "Starting…", "res": None, "err": None}

    def progress(frac, desc=""):
        state["frac"], state["desc"] = frac, desc

    def work():
        try:
            state["res"] = images_step.run(target, topics, per_story=int(per_story),
                                           creative_commons=bool(creative_commons), progress=progress)
        except Exception as e:
            state["err"] = e

    t = threading.Thread(target=work, daemon=True)
    t.start()
    while t.is_alive():
        yield _bar(state["frac"], f"🖼️ {state['desc']}"), skip, skip
        time.sleep(0.5)
    if state["err"] is not None:
        msg = str(state["err"])
        gr.Warning(msg, duration=None, title="Finding images failed")
        yield _bar(1, f"❌ {msg}", "#ef4444"), skip, skip
        return
    res = state["res"]
    n_pics = sum(len(r["options"]) for r in res["results"])
    done = _bar(1, f"✅ {n_pics} pictures for {res['log']['stories']} stories · search cost ≈ "
                   f"${res['log']['cost_usd']} · use ✨ under a story only if you need a Gemini prompt",
                "linear-gradient(90deg,#10b981,#14b8a6)")
    yield done, {"folder": str(target), "n": time.time()}, [str(target / "images.zip"), str(target / "image_prompts.md")]


def _img_state_for(folder):
    """On page load: show today's pictures if they were already searched."""
    import time
    if folder and images_step.load_results(Path(folder)):
        return {"folder": folder, "n": time.time()}
    return {"folder": None}


def _prompt_writer(folder: str, index: int):
    def write():
        try:
            images_step.write_prompts(Path(folder), index)
        except AIError as e:
            gr.Warning(str(e), duration=None, title="Writing prompts failed")
            return gr.skip(), gr.skip()
        r = images_step.load_results(Path(folder))[index]
        return images_step.prompts_md(r), gr.Button(value="🔄 Rewrite Gemini prompts")
    return write


# ---------------------------------------------------------------------------
# past runs
# ---------------------------------------------------------------------------

FILE_NOTES = {
    "youtube_script.docx": "📺 teleprompter script (Word — keeps line breaks)",
    "youtube_script.rtf": "📺 teleprompter script (Rich Text — keeps line breaks)",
    "youtube_script.txt": "📺 teleprompter script (plain text)",
    "image_prompts.md": "🖼️ pictures + Gemini prompts", "images.zip": "🖼️ all pictures",
    "youtube_script.md": "🎬 YouTube script", "news_detailed.md": "📚 detailed version",
    "final.csv": "✅ your final list", "selected.csv": "✨ AI picks", "candidates.csv": "📥 all collected stories",
}


def _run_dates() -> list[str]:
    root = config.OUTPUT_DIR
    if not root.exists():
        return []
    return sorted((p.name for p in root.glob("20??-??-??") if p.is_dir()), reverse=True)


def _zip_day(day: str) -> str | None:
    """Zip one day's folder into a temp file (kept out of the output folder)."""
    import shutil
    import tempfile
    src = config.OUTPUT_DIR / day
    if not src.exists():
        return None
    base = Path(tempfile.mkdtemp(prefix="newsagent_")) / f"NewsAgent_{day}"
    return shutil.make_archive(str(base), "zip", root_dir=src)


def on_past_refresh(current=None):
    dates = _run_dates()
    value = current if current in dates else (dates[0] if dates else None)
    return gr.Dropdown(choices=dates, value=value)


def on_past_select(day):
    if not day:
        return "No saved days yet.", "", "", None, None
    f = config.OUTPUT_DIR / day
    script = (f / "youtube_script.md").read_text(encoding="utf-8") if (f / "youtube_script.md").exists() \
        else "_No YouTube script was written on this day._"
    detailed = (f / "news_detailed.md").read_text(encoding="utf-8") if (f / "news_detailed.md").exists() \
        else "_No detailed version was written on this day._"
    names = sorted(p.name for p in f.iterdir() if p.is_file())
    lines = [f"### 📅 {day} — {len(names)} files"]
    for n in names:
        if n in FILE_NOTES:
            lines.append(f"- **{n}** — {FILE_NOTES[n]}")
    if (f / "write_log.json").exists():
        log = json.loads((f / "write_log.json").read_text(encoding="utf-8"))
        lines.append(f"\n{log.get('stories', '?')} stories · {log.get('script_words', '?')} words ≈ "
                     f"{log.get('script_minutes', '?')} min · cost ≈ ${log.get('cost_usd', '?')}")
    files = [str(f / n) for n in names if n.endswith((".md", ".csv", ".txt", ".docx", ".rtf", ".zip"))]
    return "\n".join(lines), script, detailed, files, _zip_day(day)


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------

def _copy_button() -> dict:
    """Copy button on a Textbox — the argument name differs between Gradio versions."""
    import inspect
    params = inspect.signature(gr.Textbox).parameters
    if "buttons" in params:
        return {"buttons": ["copy"]}
    if "show_copy_button" in params:
        return {"show_copy_button": True}
    return {}


def _busy(label: str):
    return lambda: gr.Button(value=label, interactive=False)


def _ready(label: str):
    return lambda: gr.Button(value=label, interactive=True)


def build() -> gr.Blocks:
    L_COLLECT, L_PICK = "1 · 📥 Collect today's news", "2 · ✨ Pick top N with AI"
    L_WRITE, L_SAVE = "4 · ✍️ Write scripts for the ticked stories", "💾 Save list only"
    with gr.Blocks(title=f"{config.SHOW_NAME} — News Studio") as demo:
        folder = gr.State(None)
        gr.HTML(HERO)

        with gr.Row(elem_id="controls"):
            top_n = gr.Slider(5, 40, value=config.TOP_N, step=1, label="🔢 How many stories (top N)")
            hours = gr.Slider(6, 48, value=config.WINDOW_HOURS, step=1, label="⏱️ News from the last … hours")
            with gr.Column(min_width=180):
                no_ai = gr.Checkbox(False, label="Free pick (no AI)")
                show_flagged = gr.Checkbox(False, label="Show ⚠️ flagged stories")
        with gr.Row():
            b_collect = gr.Button(L_COLLECT, elem_id="btn-collect", elem_classes="step-btn")
            b_pick = gr.Button(L_PICK, elem_id="btn-pick", elem_classes="step-btn")
        status = gr.Markdown(elem_id="status")

        with gr.Tabs() as tabs:
            with gr.Tab("3 · 📝 Review stories", id="review"):
                counter = gr.Markdown(elem_id="counter")
                groups = []
                for b in BUCKETS:
                    with gr.Accordion(f"{TOPIC_ICONS.get(b, '')} {b}", open=True,
                                      elem_classes=["topic", f"topic-{_slug(b)}"]):
                        groups.append(gr.CheckboxGroup(choices=[], label=b, show_label=False,
                                                       elem_classes="bucket-box"))
                with gr.Accordion("➕ Add a story you found elsewhere", open=False):
                    with gr.Row():
                        add_head = gr.Textbox(label="Headline", scale=3)
                        add_url = gr.Textbox(label="Link", scale=3)
                        add_bucket = gr.Dropdown(BUCKETS, value="Space & Science", label="Topic", scale=1)
                    b_add = gr.Button("➕ Add and tick it", elem_id="btn-add", elem_classes="step-btn")
                with gr.Row():
                    b_save = gr.Button(L_SAVE, elem_id="btn-save", elem_classes="step-btn")
                    b_write = gr.Button(L_WRITE, elem_id="btn-write", elem_classes="step-btn")
                write_status = gr.HTML(elem_id="write-status")   # live progress bar shows here
                # appears only after "Save list only" — a download link for your saved list
                saved_file = gr.File(label="⬇️ Your saved list (final.csv)", visible=False, height=90)
            with gr.Tab("🎬 YouTube script", id="script"):
                notes = gr.Markdown()
                script_md = gr.Markdown()
            with gr.Tab("📺 Teleprompter", id="teleprompter"):
                gr.Markdown("Plain text for your teleprompter app — no `*`, `#` or other symbols. "
                            "Copy it with the button in the corner, or download it from ⬇️ Downloads. If your app joins "
                            "all the lines together, import **youtube_script.docx** (or **.rtf**) instead of the .txt — "
                            "those keep every line and paragraph break.")
                teleprompter = gr.Textbox(show_label=False, lines=28, max_lines=60, interactive=False,
                                          elem_id="teleprompter", **_copy_button())
            with gr.Tab("📚 Detailed version", id="detailed"):
                detailed_md = gr.Markdown()
            with gr.Tab("⬇️ Downloads", id="downloads"):
                files = gr.Files(label="Today's files")
            with gr.Tab("🖼️ Images", id="images"):
                gr.Markdown("**🔍 Find images** gets 2-3 pictures per story that fit **half of a 16:9 video** "
                            "(960×1080, 8:9) — about $0.001 per story, no AI. If a story's pictures aren't good "
                            "enough, click **✨ Write Gemini prompts** under it: you get a prompt to *create* a "
                            "realistic picture and one to *enhance* each found picture in Gemini.")
                with gr.Row():
                    img_source = gr.Radio(["Today's final list", "Upload a file"], value="Today's final list",
                                          label="Stories from", scale=2)
                    img_upload = gr.File(label="Upload final.csv / final.json / youtube_script.txt / a .txt list",
                                         file_types=[".csv", ".json", ".txt", ".md"], visible=False, scale=3)
                with gr.Row():
                    img_count = gr.Slider(2, 3, value=3, step=1, label="Pictures per story")
                    img_cc = gr.Checkbox(False, label="Only Creative Commons pictures (free to reuse, fewer results)")
                b_images = gr.Button("🔍 Find images", elem_id="btn-images", elem_classes="step-btn")
                img_status = gr.HTML()
                img_files = gr.Files(label="⬇️ images.zip (originals + 8:9 crops) and image_prompts.md")
                img_state = gr.State({"folder": None})

                @gr.render(inputs=img_state)
                def show_image_cards(st):
                    folder_path = (st or {}).get("folder")
                    results = images_step.load_results(Path(folder_path)) if folder_path else []
                    if not results:
                        gr.Markdown("_No pictures yet — click **🔍 Find images**._")
                        return
                    for i, r in enumerate(results):
                        with gr.Group(elem_classes="img-card"):
                            gr.Markdown(f"### {i + 1}. {r['headline']}")
                            pics = [(o.get("half") or o["file"], f"{j} · fit {int((o.get('fit') or 0) * 100)}%")
                                    for j, o in enumerate(r["options"], 1) if o.get("file")]
                            if pics:
                                gr.Gallery(value=pics, columns=3, height=320, object_fit="contain",
                                           show_label=False, allow_preview=True)
                            gr.Markdown(images_step.options_md(r))
                            has = bool(r.get("prompts"))
                            btn = gr.Button("🔄 Rewrite Gemini prompts" if has else
                                            "✨ Write Gemini prompts for this story (≈ $0.0005)", size="sm")
                            out = gr.Markdown(images_step.prompts_md(r))
                            btn.click(_prompt_writer(folder_path, i), None, [out, btn])
            with gr.Tab("📂 Past runs", id="past"):
                with gr.Row():
                    past_day = gr.Dropdown(choices=[], label="📅 Pick a day", scale=4)
                    b_past_refresh = gr.Button("🔄 Refresh list", scale=1)
                past_info = gr.Markdown()
                with gr.Row():
                    past_zip = gr.File(label="⬇️ Download the whole day (.zip)")
                    past_files = gr.Files(label="Or download single files")
                with gr.Tabs():
                    with gr.Tab("🎬 YouTube script"):
                        past_script = gr.Markdown()
                    with gr.Tab("📚 Detailed version"):
                        past_detailed = gr.Markdown()

        # wiring — buttons are greyed out while they work, the progress bar shows under them
        demo.load(on_load, [show_flagged, top_n], [folder, status, counter, *groups]) \
            .then(_img_state_for, folder, img_state)
        img_source.change(lambda v: gr.File(visible=v == "Upload a file"), img_source, img_upload)
        L_IMAGES = "🔍 Find images"
        b_images.click(_busy("⏳ Finding pictures… please wait"), None, b_images) \
            .then(on_images, [folder, img_source, img_upload, img_count, img_cc],
                  [img_status, img_state, img_files], show_progress="hidden") \
            .then(_ready(L_IMAGES), None, b_images)
        past_out = [past_info, past_script, past_detailed, past_files, past_zip]
        demo.load(on_past_refresh, None, past_day).then(on_past_select, past_day, past_out)
        b_past_refresh.click(on_past_refresh, past_day, past_day).then(on_past_select, past_day, past_out)
        past_day.input(on_past_select, past_day, past_out)
        b_collect.click(_busy("⏳ Collecting news…"), None, b_collect) \
            .then(on_collect, [hours, show_flagged, top_n], [folder, status, counter, *groups]) \
            .then(_ready(L_COLLECT), None, b_collect)
        b_pick.click(_busy("⏳ Picking the top stories…"), None, b_pick) \
            .then(on_pick, [folder, top_n, no_ai, show_flagged], [status, counter, *groups]) \
            .then(_ready(L_PICK), None, b_pick)
        show_flagged.change(on_toggle_flagged, [folder, show_flagged, top_n, *groups], [counter, *groups])
        for g in groups:
            g.change(lambda n, *v: _counter(n, list(v)), [top_n, *groups], counter, show_progress="hidden")
        top_n.change(lambda n, *v: _counter(n, list(v)), [top_n, *groups], counter, show_progress="hidden")
        b_add.click(on_add, [folder, add_head, add_url, add_bucket, show_flagged, top_n, *groups],
                    [add_head, add_url, counter, *groups])
        b_save.click(on_save, [folder, *groups], saved_file)
        b_write.click(_busy("⏳ Writing scripts… please wait"), None, b_write) \
            .then(on_write, [folder, *groups], [script_md, detailed_md, notes, files, write_status, tabs,
                                                teleprompter],
                  show_progress="hidden") \
            .then(_ready(L_WRITE), None, b_write) \
            .then(on_past_refresh, past_day, past_day)
    return demo


def main(port: int | None = None, host: str | None = None, share: bool = False):
    """Local: http://127.0.0.1:7860. On Railway/Render: host 0.0.0.0 and the PORT they give."""
    import os
    port = port or int(os.getenv("PORT", 7860))
    host = host or os.getenv("HOST") or ("0.0.0.0" if os.getenv("PORT") else "127.0.0.1")
    user, pw = os.getenv("APP_USERNAME"), os.getenv("APP_PASSWORD")
    auth = (user, pw) if user and pw else None
    build().queue(default_concurrency_limit=1).launch(
        server_name=host, server_port=port, share=share, css=CSS, theme=THEME,
        allowed_paths=[str(config.OUTPUT_DIR.resolve())],
        auth=auth, auth_message=f"{config.SHOW_NAME} — News Studio" if auth else None,
        inbrowser=host == "127.0.0.1")
