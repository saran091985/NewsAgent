# Hosting NewsAgent on Railway

The app is the Gradio page in `newsagent/ui.py`. Railway builds it with Railpack
(it sees `pyproject.toml` + `uv.lock` and installs with uv, Python from `.python-version`)
and starts it with the command in `railway.json`: `python -m newsagent ui`.
The app listens on `0.0.0.0:$PORT` automatically when Railway sets `PORT`.

## One-time setup
1. Push the repo to GitHub: `git push origin main`.
2. railway.com → **New Project → Deploy from GitHub repo** → pick `NewsAgent`.
3. **Variables** (service → Variables). Copy from your `.env`:
   | Variable | Needed for |
   |---|---|
   | `OPENAI_API_KEY` | picking + writing |
   | `SERPER_API_KEY` | extra facts for thin stories |
   | `APP_USERNAME`, `APP_PASSWORD` | **login screen — set these, the URL is public** |
   | `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, `LANGSMITH_ENDPOINT` | optional tracing |
   | `TZ=Asia/Dubai` | optional, log times in UAE time |
4. **Volume** (service → Settings → Volumes → New volume), mount path **`/app/output`**.
   Without it, the daily folders (and the AI's learning from your picks) are wiped on every redeploy.
5. **Networking → Generate Domain** to get a public `https://….up.railway.app` link.

Every `git push` to `main` redeploys automatically.

## Notes
- Never commit `.env` — it is in `.gitignore`; keys live only in Railway Variables.
- If a source stops returning stories on Railway but works on your PC, that site may be blocking
  cloud servers — check the "Collected …" line on the page after step 1.
- Cost: Railway's usage-based plan; the app is idle most of the day.
