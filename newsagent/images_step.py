"""
Images for the stories — only runs when you click "Find images" on the 🖼️ Images tab.

For each story:
  1. one Google Images search (Serper, ~$0.001) for the headline
  2. pick the best 2-3 pictures for HALF of a 16:9 video frame (960×1080, portrait 8:9):
     big enough, close to that shape, from different websites, no watermarked stock sites
  3. download them and also save a ready-cropped 8:9 copy
  4. one gpt-4o-mini call writes two Gemini prompts per picture:
       - create: make a NEW original picture like it (safe to use, no copyright worries)
       - enhance: attach the found picture in Gemini and turn it into a high-resolution 8:9 version

Output: output/<date>/images/ (pictures + *_half.jpg crops), image_prompts.md, images.zip
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
PRICE_IN, PRICE_OUT = 0.15, 0.60

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

class OptionPrompts(BaseModel):
    option: int = Field(description="option number as given")
    create_prompt: str = Field(description="Gemini prompt to create a NEW original image like this option")
    enhance_prompt: str = Field(description="Gemini prompt to use with the attached found image")


class StoryPrompts(BaseModel):
    prompts: list[OptionPrompts]


PROMPT_SYSTEM = """You write image prompts for Google Gemini for a kids' YouTube news show (ages 8-14).
The picture fills HALF of a 16:9 video frame: portrait 8:9, 1920×2160 pixels (or 960×1080), the host
stands on the other half, so keep the main subject centred with some space around it.

create_prompt (60-100 words): a brand-new ORIGINAL picture showing the same scene or idea as the option.
  Bright, friendly, detailed; say the style (photo-realistic or colourful 3D illustration), lighting,
  composition and "8:9 portrait, 1920x2160, high resolution". Never show real, identifiable people,
  logos, brand names or text in the picture — use symbols instead (flags, maps, objects, silhouettes).
  Nothing scary or violent: for conflicts show calm symbols (ships on a sea route, a peace table, a map).
enhance_prompt (40-80 words): the user attaches the found picture in Gemini. Ask Gemini to keep the
  same subject and composition, make it a sharp high-resolution 1920x2160 8:9 portrait, extend the
  background naturally if the shape needs it, improve lighting and colour, remove blur and noise.
  Do not ask to remove watermarks or logos."""


def _prompts_llm():
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    load_dotenv(override=True)
    return ChatOpenAI(model=config.WRITE_MODEL, temperature=0.6).with_structured_output(
        StoryPrompts, include_raw=True, method="function_calling")


def _story_prompt(topic: dict, opts: list[dict]) -> str:
    lines = [f"Option {i}: \"{o.get('title', '')}\" from {o.get('domain', '')} ({o.get('w')}×{o.get('h')})"
             for i, o in enumerate(opts, 1)]
    if not lines:
        lines = ["Option 1: (no picture found — invent a fitting scene)"]
    return (f"News story: {topic['headline']}\nTopic: {topic.get('bucket') or 'news'}\n\n"
            "Pictures found for it:\n" + "\n".join(lines) +
            "\n\nWrite create_prompt and enhance_prompt for EVERY option.")


# ---------------------------------------------------------------------------
# Run
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
    log = {"stories": n, "searches": 0, "input_tokens": 0, "output_tokens": 0, "errors": {}}

    # 1-3: search, pick, download — a few stories at a time
    def gather(i_topic):
        i, t = i_topic
        results = search_images(t["headline"], creative_commons)
        opts = pick_images(results, per_story)
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
            say(0.02 + 0.6 * done / n, f"Searching pictures for story {done} of {n}…")

    # 4: prompts
    from langchain_core.messages import HumanMessage, SystemMessage
    llm = _prompts_llm()

    def write(i):
        return llm.invoke([SystemMessage(content=PROMPT_SYSTEM),
                           HumanMessage(content=_story_prompt(topics[i - 1], found.get(i, [])))])

    prompts: dict[int, dict[int, dict]] = {}
    say(0.65, "Writing Gemini prompts…")
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(write, i): i for i in range(1, n + 1)}
        for done, f in enumerate(as_completed(futs), 1):
            i = futs[f]
            try:
                res = f.result()
            except Exception as e:
                if done == n and not prompts:
                    raise AIError(friendly(e)) from e
                log["errors"][f"prompts {i}"] = str(e)[:200]
                continue
            u = getattr(res.get("raw"), "usage_metadata", None) or {}
            log["input_tokens"] += u.get("input_tokens", 0)
            log["output_tokens"] += u.get("output_tokens", 0)
            if res.get("parsed"):
                prompts[i] = {p.option: p.model_dump() for p in res["parsed"].prompts}
            say(0.65 + 0.3 * done / n, f"Writing Gemini prompts {done} of {n}…")

    say(0.97, "Saving…")
    md = _render(topics, found, prompts)
    (folder / "image_prompts.md").write_text(md, encoding="utf-8")
    zip_path = shutil.make_archive(str(folder / "images"), "zip", root_dir=img_dir)
    log["cost_usd"] = round((log["input_tokens"] * PRICE_IN + log["output_tokens"] * PRICE_OUT) / 1e6
                            + log["searches"] * 0.001, 4)
    log["run_at"] = datetime.now().isoformat(timespec="seconds")
    (folder / "images_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    gallery = []
    for i in sorted(found):
        for j, o in enumerate(found[i], 1):
            gallery.append((o.get("half") or o["file"], f"{i}.{j} · {topics[i - 1]['headline'][:60]} · "
                                                        f"{o['w']}×{o['h']} · fit {int(o['fit'] * 100)}%"))
    say(1.0, f"Done: {sum(len(v) for v in found.values())} pictures for {n} stories")
    return {"markdown": md, "gallery": gallery, "zip": zip_path,
            "prompts_file": str(folder / "image_prompts.md"), "log": log}


def _render(topics: list[dict], found: dict[int, list[dict]], prompts: dict[int, dict[int, dict]]) -> str:
    out = ["# 🖼️ Pictures and Gemini prompts",
           f"_Half of a 16:9 frame = 960×1080 (8:9). `fit` shows how well a picture matches that "
           f"shape and size. Found pictures belong to their websites — check the licence before using one, "
           f"or use the **create** prompt to make your own._", ""]
    for i, t in enumerate(topics, 1):
        out += ["---", f"## {i}. {t['headline']}", ""]
        opts = found.get(i, [])
        ps = prompts.get(i, {})
        if not opts:
            out += ["_No usable picture found — use the create prompt below._", ""]
            p = ps.get(1)
            if p:
                out += ["**✨ Create in Gemini**", "```text", p["create_prompt"], "```", ""]
            continue
        for j, o in enumerate(opts, 1):
            note = " · only a small preview could be downloaded" if o.get("thumb_only") else ""
            out += [f"### Option {j} — fit {int(o['fit'] * 100)}%",
                    f"{o.get('title', '')}  ",
                    f"{o['w']}×{o['h']} · [{o['domain']}]({o.get('link', '')}) · "
                    f"[full picture]({o['imageUrl']}){note}  ",
                    f"Files: `images/{Path(o['file']).name}` and cropped `images/{Path(o.get('half', '')).name}`", ""]
            p = ps.get(j)
            if p:
                out += ["**✨ Create a new picture in Gemini**", "```text", p["create_prompt"], "```",
                        "**🔧 Attach this picture in Gemini and use**", "```text", p["enhance_prompt"], "```", ""]
    return "\n".join(out)
