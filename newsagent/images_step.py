"""
Images for the stories — only runs when you click "Find images" on the 🖼️ Images tab.

For each story:
  1. one Google Images search (Serper, ~$0.001) for the headline
  2. pick the best 2-3 pictures for HALF of a 16:9 video frame (960×1080, portrait 8:9):
     big enough, close to that shape, from different websites, no watermarked stock sites
  3. download them and also save a ready-cropped 8:9 copy
  4. ONLY when you click "Write Gemini prompts" for a story: one small gpt-4o-mini call writes
       - create: a PHOTO-REALISTIC new picture of the story (no real faces, no text, nothing scary)
       - enhance: one per found picture, to attach it in Gemini and make it high-resolution 8:9

Output: output/<date>/images/ (pictures + *_half.jpg crops), images.json, image_prompts.md, images.zip
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests
from pydantic import BaseModel, Field

from . import config
from .ai_errors import AIError, friendly

HALF_W, HALF_H = 960, 1080                    # half of a 1920×1080 frame
TARGET_ASPECT = HALF_W / HALF_H               # 0.89 (a little taller than wide)

# watermarked stock-photo sites — their previews are unusable
STOCK_DOMAINS = ("shutterstock", "gettyimages", "istockphoto", "alamy", "dreamstime", "depositphotos",
                 "123rf", "adobestock", "stock.adobe", "vectorstock", "pond5", "bigstockphoto", "canstockphoto")


# ---------------------------------------------------------------------------
# Topics: today's list or an uploaded file
# ---------------------------------------------------------------------------

def topics_from_today(folder: Path) -> list[dict]:
    for name in ("final.json", "selected.json"):
        f = folder / name
        if f.exists():
            rows = json.loads(f.read_text(encoding="utf-8"))
            return [{"headline": r["headline"], "bucket": r.get("bucket", "")} for r in rows]
    return []


def topics_from_file(path: str) -> list[dict]:
    """final.csv / final.json from this app, youtube_script.txt/.md, or any text file with one topic per line."""
    p = Path(path)
    text = p.read_text(encoding="utf-8-sig", errors="ignore")
    if p.suffix.lower() == ".json":
        rows = json.loads(text)
        return [{"headline": r.get("headline") or r.get("title") or str(r), "bucket": r.get("bucket", "")}
                for r in rows]
    if p.suffix.lower() == ".csv":
        rows = list(csv.DictReader(io.StringIO(text)))
        key = next((k for k in ("headline", "title", "topic", "Headline", "Title", "Topic") if rows and k in rows[0]), None)
        if key:
            return [{"headline": r[key], "bucket": r.get("bucket", "")} for r in rows if r.get(key)]
        return [{"headline": next(iter(r.values())), "bucket": ""} for r in rows]
    # scripts: numbered story titles ("1. The Big Squeeze" / "### 1. The Big Squeeze 🌊")
    numbered = re.findall(r"^\s*(?:#+\s*)?\d+[.)]\s+(.+?)\s*$", text, flags=re.M)
    lines = numbered or [l.strip("-•* \t") for l in text.splitlines() if len(l.strip()) > 8]
    from .write_step import _EMOJI
    return [{"headline": _EMOJI.sub("", re.sub(r"[*_#`]", "", l)).strip(), "bucket": ""} for l in lines[:40]]


# ---------------------------------------------------------------------------
# 1-2. Search and pick
# ---------------------------------------------------------------------------

def search_images(query: str, creative_commons: bool) -> list[dict]:
    key = os.getenv("SERPER_API_KEY")
    if not key:
        raise AIError("No SERPER_API_KEY found — add it to .env (or Railway Variables) to search images.")
    body = {"q": query, "num": 30}
    if creative_commons:
        body["tbs"] = "il:cl"                    # Google Images → usage rights: Creative Commons
    try:
        r = requests.post("https://google.serper.dev/images", json=body,
                          headers={"X-API-KEY": key, "Content-Type": "application/json"}, timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        raise AIError(f"Image search failed: {e}") from e
    return r.json().get("images", [])


def fit_score(w: int, h: int) -> float:
    """1.0 = perfect for a 960×1080 half frame; lower for too small or the wrong shape."""
    if not w or not h:
        return 0.0
    shape = max(0.0, 1 - abs(math.log((w / h) / TARGET_ASPECT)) / math.log(2.5))   # 2.5× off → 0
    size = min(1.0, min(w / HALF_W, h / HALF_H) ** 0.5)                              # big enough → 1
    return round(shape * 0.55 + size * 0.45, 3)


def pick_images(results: list[dict], k: int) -> list[dict]:
    good = []
    for r in results:
        url = r.get("imageUrl", "")
        dom = (r.get("domain") or urlparse(r.get("link", "")).netloc).lower()
        w, h = int(r.get("imageWidth") or 0), int(r.get("imageHeight") or 0)
        if not url or any(s in dom or s in url for s in STOCK_DOMAINS) or min(w, h) < 400:
            continue
        good.append({**r, "domain": dom, "w": w, "h": h, "fit": fit_score(w, h)})
    good.sort(key=lambda r: r["fit"], reverse=True)
    picked, seen = [], set()
    for r in good:                                    # one picture per website, for variety
        if r["domain"] in seen:
            continue
        seen.add(r["domain"])
        picked.append(r)
        if len(picked) == k:
            break
    return picked


# ---------------------------------------------------------------------------
# 3. Download + ready-cropped half-frame copy
# ---------------------------------------------------------------------------

def download(opt: dict, dest: Path) -> dict:
    from PIL import Image
    try:
        r = requests.get(opt["imageUrl"], headers=config.HTTP_HEADERS, timeout=20)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content))
        img.load()
    except Exception:
        # many sites block downloads; fall back to Google's thumbnail so you still see the option
        try:
            r = requests.get(opt.get("thumbnailUrl", ""), headers=config.HTTP_HEADERS, timeout=15)
            img = Image.open(io.BytesIO(r.content))
            img.load()
            opt["thumb_only"] = True
        except Exception:
            return opt
    img = img.convert("RGB")
    opt["w"], opt["h"] = img.size
    opt["fit"] = fit_score(*img.size)
    img.save(dest.with_suffix(".jpg"), quality=92)
    opt["file"] = str(dest.with_suffix(".jpg"))
    # centre-crop to 8:9 and size to 960×1080 (never enlarge — enlarging is what the Gemini prompt is for)
    w, h = img.size
    if w / h > TARGET_ASPECT:
        nw = int(h * TARGET_ASPECT)
        img = img.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
    else:
        nh = int(w / TARGET_ASPECT)
        img = img.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))
    if img.size[0] > HALF_W:
        img = img.resize((HALF_W, HALF_H), Image.LANCZOS)
    half = dest.with_name(dest.name + "_half").with_suffix(".jpg")
    img.save(half, quality=92)
    opt["half"] = str(half)
    opt["half_size"] = f"{img.size[0]}×{img.size[1]}"
    return opt


# ---------------------------------------------------------------------------
# 4. Gemini prompts (AI)
# ---------------------------------------------------------------------------

class Enhance(BaseModel):
    option: int = Field(description="picture option number as given")
    prompt: str = Field(description="Gemini prompt to use with that attached picture")


class TopicPrompts(BaseModel):
    create_prompt: str = Field(description="Gemini prompt for a NEW photo-realistic picture of the story")
    enhance_prompts: list[Enhance] = Field(default_factory=list)


PROMPT_SYSTEM = """You write image prompts for Google Gemini for a kids' YouTube news show (ages 8-14).
The picture fills HALF of a 16:9 video frame: portrait 8:9, 1920x2160 pixels; the host stands on the
other half, so keep the main subject centred.

create_prompt (70-110 words): a PHOTO-REALISTIC news photograph of the story's scene — it must look like
  a real photo taken by a press photographer, NOT a cartoon, 3D render or illustration.
  Describe: the real place and setting, what is happening, time of day and natural light, camera and lens
  (e.g. "shot on a full-frame DSLR, 35mm lens, f/4"), realistic textures and colours, depth of field,
  "documentary news photography, ultra-detailed, 8:9 portrait, 1920x2160".
  Rules: no identifiable real people — show people from behind, far away, as a crowd or as hands only;
  no readable text, signs, captions, logos or brand names anywhere in the picture (they come out wrong);
  nothing violent, bloody or frightening — for conflicts show calm, factual scenes (ships in a strait,
  empty conference table with flags, trucks leaving a base at sunrise, a map on a desk).
enhance_prompts (40-70 words each, one per found picture): the user attaches that picture in Gemini.
  Ask Gemini to keep the same subject, scene and composition, keep it photo-realistic, turn it into a sharp
  high-resolution 1920x2160 8:9 portrait, extend the background naturally if the shape needs it, and
  improve lighting, colour and detail while removing blur and noise. Do not ask to remove watermarks or logos."""


def _prompts_llm():
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    load_dotenv(override=True)
    return ChatOpenAI(model=config.WRITE_MODEL, temperature=0.5).with_structured_output(
        TopicPrompts, include_raw=True, method="function_calling")


def _topic_prompt(topic: dict, opts: list[dict]) -> str:
    lines = [f"Picture {i}: \"{o.get('title', '')}\" from {o.get('domain', '')} ({o.get('w')}x{o.get('h')})"
             for i, o in enumerate(opts, 1)]
    found = "\n".join(lines) if lines else "(no pictures found — only write create_prompt)"
    return (f"News story: {topic['headline']}\nTopic: {topic.get('bucket') or 'news'}\n\n"
            f"Pictures found for it:\n{found}\n\n"
            "Write one create_prompt and one enhance prompt for EVERY picture above.")


# ---------------------------------------------------------------------------
# Saved results (images.json) — so prompts can be written later, per story
# ---------------------------------------------------------------------------

def load_results(folder: Path) -> list[dict]:
    f = Path(folder) / "images.json"
    if not f.exists():
        return []
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except ValueError:
        return []


def save_results(folder: Path, results: list[dict]) -> None:
    folder = Path(folder)
    (folder / "images.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (folder / "image_prompts.md").write_text(_render(results), encoding="utf-8")


# ---------------------------------------------------------------------------
# Run: search + download only (no AI). Prompts are written per story on request.
# ---------------------------------------------------------------------------

def run(folder: Path, topics: list[dict], per_story: int = 3, creative_commons: bool = False,
        progress=None) -> dict:
    def say(frac, msg):
        print(msg)
        if progress:
            progress(frac, msg)

    img_dir = folder / "images"
    if img_dir.exists():
        shutil.rmtree(img_dir, ignore_errors=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    n = len(topics)
    log = {"stories": n, "searches": 0, "errors": {}}

    def gather(i_topic):
        i, t = i_topic
        opts = pick_images(search_images(t["headline"], creative_commons), per_story)
        for j, o in enumerate(opts, 1):
            download(o, img_dir / f"{i:02d}_{j}")
        return i, [o for o in opts if o.get("file")]

    found: dict[int, list[dict]] = {}
    say(0.02, f"Searching pictures for story 0 of {n}…")
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(gather, (i, t)) for i, t in enumerate(topics, 1)]
        for done, f in enumerate(as_completed(futs), 1):
            try:
                i, opts = f.result()
                found[i] = opts
            except AIError:
                raise
            except Exception as e:
                log["errors"][f"search {done}"] = str(e)[:200]
            log["searches"] += 1
            say(0.02 + 0.93 * done / n, f"Searching pictures for story {done} of {n}…")

    keep = ("title", "imageUrl", "link", "domain", "w", "h", "fit", "file", "half", "half_size", "thumb_only")
    results = [{"headline": t["headline"], "bucket": t.get("bucket", ""),
                "options": [{k: o.get(k) for k in keep} for o in found.get(i, [])],
                "prompts": None}
               for i, t in enumerate(topics, 1)]
    say(0.97, "Saving…")
    save_results(folder, results)
    shutil.make_archive(str(folder / "images"), "zip", root_dir=img_dir)
    log["cost_usd"] = round(log["searches"] * 0.001, 4)
    log["run_at"] = datetime.now().isoformat(timespec="seconds")
    (folder / "images_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    say(1.0, f"Done: {sum(len(r['options']) for r in results)} pictures for {n} stories")
    return {"results": results, "log": log}


def write_prompts(folder: Path, index: int) -> dict:
    """One small AI call for ONE story (index starts at 0). Saved into images.json / image_prompts.md."""
    results = load_results(folder)
    if not 0 <= index < len(results):
        raise AIError("That story is no longer in the list — search again.")
    r = results[index]
    from langchain_core.messages import HumanMessage, SystemMessage
    try:
        res = _prompts_llm().invoke([SystemMessage(content=PROMPT_SYSTEM),
                                     HumanMessage(content=_topic_prompt(r, r["options"]))])
    except Exception as e:
        raise AIError(friendly(e)) from e
    if not res.get("parsed"):
        raise AIError("The AI answer could not be read — click again.")
    p = res["parsed"]
    r["prompts"] = {"create": p.create_prompt, "enhance": {e.option: e.prompt for e in p.enhance_prompts}}
    save_results(folder, results)
    return r["prompts"]


def prompts_md(r: dict) -> str:
    """Markdown for one story's prompts (code blocks have copy buttons)."""
    p = r.get("prompts")
    if not p:
        return ""
    out = ["**✨ Create a new realistic picture** — paste into [Gemini](https://gemini.google.com/app)",
           "```text", p["create"], "```"]
    enh = p.get("enhance") or {}
    for j, _ in enumerate(r["options"], 1):
        text = enh.get(j) or enh.get(str(j))
        if text:
            out += [f"**🔧 Picture {j}: attach it in [Gemini](https://gemini.google.com/app) and use**",
                    "```text", text, "```"]
    return "\n".join(out)


def options_md(r: dict) -> str:
    lines = []
    for j, o in enumerate(r["options"], 1):
        note = " · small preview only" if o.get("thumb_only") else ""
        lines.append(f"**{j}.** fit {int((o.get('fit') or 0) * 100)}% · {o.get('w')}×{o.get('h')} · "
                     f"[{o.get('domain')}]({o.get('link') or o.get('imageUrl')}) · "
                     f"[full picture]({o.get('imageUrl')}){note}")
    return "  \n".join(lines) or "_No usable picture found — write a prompt to create one._"


def _render(results: list[dict]) -> str:
    out = ["# 🖼️ Pictures and Gemini prompts",
           "_Half of a 16:9 frame = 960×1080 (8:9). Found pictures belong to their websites — check the "
           "licence before using one, or create your own with the prompt._", ""]
    for i, r in enumerate(results, 1):
        out += ["---", f"## {i}. {r['headline']}", "", options_md(r), ""]
        if r.get("prompts"):
            out += [prompts_md(r), ""]
    return "\n".join(out)
