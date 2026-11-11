# CareAssist — Operations Runbook

**Audience:** whoever maintains or operates this app. Six
months from now, with no memory of how any of this works.

**Last verified:** 2026-11-10
**Live app:** https://careassist-web.onrender.com
**API:** https://careassist-api.onrender.com
**Repo:** https://github.com/Thomasayanfeoluwa/HealthAssistant-AI

---

## Table of contents

1. [What this app is (30 seconds)](#1-what-this-app-is-30-seconds)
2. [Local development](#2-local-development)
3. [Deploying](#3-deploying)
4. [Rolling back](#4-rolling-back)
5. [Rotating keys and secrets](#5-rotating-keys-and-secrets)
6. [Re-indexing the medical book](#6-re-indexing-the-medical-book)
7. [Common failures and fixes](#7-common-failures-and-fixes)
8. [Monitoring and quotas](#8-monitoring-and-quotas)
9. [Emergency shutdown](#9-emergency-shutdown)
10. [What NOT to do](#10-what-not-to-do)
11. [Architecture in one diagram](#11-architecture-in-one-diagram)

---

## 1. What this app is (30 seconds)

A FastAPI backend + React frontend that answers health questions by:

1. Embedding the question via **Pinecone Inference** (`multilingual-e5-large`)
2. Retrieving the top 3 chunks from **`carebot-e5`** (3,424 vectors from `Medical_book.pdf`)
3. If the top score is below 0.4, searching the web via **Tavily**
4. Generating the answer via **Groq** (`openai/gpt-oss-120b`), falling back to **OpenRouter** if Groq fails

Everything runs on **Render free tier**:
- `careassist-api` — Python web service (512 MB RAM, 1 worker)
- `careassist-web` — React static site
- No Redis (removed at Stage 8.3 due to Render's 1-instance free tier limit)

---

## 2. Local development

### First-time setup

```powershell
# Clone
git clone https://github.com/Thomasayanfeoluwa/HealthAssistant-AI.git
cd HealthAssistant-AI

# Python env
uv venv --python 3.11 --seed .venv
. .\.venv\Scripts\Activate.ps1
cd backend
pip install -r requirements-dev.txt
cd ..

# Frontend
cd frontend
npm ci
cd ..

# Fill secrets (copy template, fill in real values)
Copy-Item .env.example .env
# Edit .env with your actual keys

# Frontend env
Copy-Item frontend/.env.local.example frontend/.env.local  # if exists
# Otherwise create it manually:
#   VITE_API_BASE_URL=http://localhost:8000
#   VITE_PILOT_AUTH_TOKEN=<same value as PILOT_AUTH_TOKEN in .env>
```

### Running locally

**Terminal 1 — backend:**
```powershell
cd backend
. ..\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 — frontend:**
```powershell
cd frontend
. ..\.venv\Scripts\Activate.ps1
npm run dev
```

Open http://localhost:5173.

**Critical:** run `--reload` in dev. Without it, `.py` file edits do not take effect until you manually restart.

### Running the tests

```powershell
cd backend
python -m pytest tests/ -q           # 41 tests, ~15s
python -m app.eval.run_eval          # 5-case eval, ~30s, real API calls
```

The eval uses real Groq / Pinecone / Tavily credits. Do not run it in a loop.

---

## 3. Deploying

### Automatic (default)

Render watches `main`. Every `git push origin main` triggers a rebuild+deploy of both services.

```powershell
# On your laptop
git add -A
git commit -m "your change"
git push origin main

# Watch the build
# https://dashboard.render.com → careassist-api → Events
```

Build times:
- Backend: **~3 minutes** (after B3 dependency cleanup)
- Frontend: **~2 minutes**

### Manual redeploy without a code change

Useful for:
- After changing an env var (env changes do not auto-redeploy)
- After re-indexing (no code change)

```
Render dashboard → careassist-api → Manual Deploy → Deploy latest commit
```

Same for `careassist-web`.

### Forcing a re-sync of `render.yaml`

If you changed `render.yaml` (added a service, changed a plan):

```
Render dashboard → Blueprint (top of sidebar) → Sync
```

---

## 4. Rolling back

### Scenario A — a bad deploy just landed

```
Render dashboard → careassist-api → Events
Find the last known-good deploy (before the bad one)
Click the three-dot menu on that deploy → "Redeploy"
```

Takes ~30 seconds. No git changes needed — Render keeps previous Docker images.

### Scenario B — a bad commit is on `main`

```powershell
# Local rollback
git log --oneline -10                       # find the bad commit SHA
git revert <bad-commit-sha>                 # creates a new commit that undoes it
git push origin main
```

**Do not use `git reset --hard` on a pushed commit.** It rewrites history and forces `--force push`, which breaks any other contributors. `git revert` is the correct tool.

### Scenario C — the entire repo is broken

```powershell
# Find a known-good tag (we do not have any yet — this is why you tag releases!)
git tag -l

# If no tags exist, find a good commit
git log --oneline -20

# Deploy that specific commit on Render without touching main:
# Render dashboard → careassist-api → Settings → "Deploy specific commit"
```

**Recommendation:** start tagging releases after major milestones:

```powershell
git tag -a v0.1.0-pilot -m "First pilot release"
git push origin v0.1.0-pilot
```

---

## 5. Rotating keys and secrets

There are **three places** every secret is stored. All three must be updated together or something breaks.

| Where | What it stores | Who needs it |
|---|---|---|
| Local `.env` | All keys | Your laptop only |
| Render env vars | All keys | The deployed app |
| GitHub Actions secrets | All keys | CI eval job (when billing is fixed) |

### Rotation procedure (using `GROQ_API_KEY` as example)

1. **Generate new key** at the provider (groq.com/console → API Keys → create new)
2. **Update local** `.env`: `GROQ_API_KEY=<new value>`
3. **Verify locally** still works:
   ```powershell
   cd backend
   python -m app.eval.run_eval --limit 1
   ```
4. **Update Render**: dashboard → `careassist-api` → Environment → edit `GROQ_API_KEY` → Save and Deploy
5. **Update GitHub**: repo → Settings → Secrets → Actions → update `GROQ_API_KEY`
6. **Delete the old key** at the provider (do this LAST, after confirming the new one works everywhere)

### Which keys affect which service

| Key | Used by | If missing |
|---|---|---|
| `GROQ_API_KEY` | Backend | Falls back to OpenRouter (still works) |
| `OPENROUTER_API_KEY` | Backend | If Groq also fails, all chats return 500 |
| `PINECONE_API_KEY` | Backend | No PDF retrieval, web-only answers |
| `TAVILY_API_KEY` | Backend | No web fallback, PDF-only answers |
| `PILOT_AUTH_TOKEN` | **Both** backend AND frontend | All chats 401 |
| `SESSION_SECRET` | Backend | Existing sessions invalidate, new ones created |
| `VITE_PILOT_AUTH_TOKEN` | Frontend only | Frontend refuses to send any chat ("token not set") |

**Critical:** `PILOT_AUTH_TOKEN` (backend) and `VITE_PILOT_AUTH_TOKEN` (frontend) must be **byte-identical**. If they drift, you get `chat failed: 401`.

**Critical:** after updating `VITE_PILOT_AUTH_TOKEN`, the frontend must be **rebuilt** — Vite bakes `VITE_*` vars into the JS bundle at build time. A Render env var change on `careassist-web` triggers an auto-rebuild.

---

## 6. Re-indexing the medical book

Needed when:
- You replace `Data/Medical_book.pdf` with a new/different PDF
- You want to switch embedding models
- The `carebot-e5` index is deleted or corrupted

### Current state

- PDF: `Data/Medical_book.pdf` (16 MB, 637 pages)
- Chunk size: 1000 chars, 200 overlap
- Total chunks: 3,424
- Embedding model: `multilingual-e5-large` (Pinecone Inference, 1024 dims)
- Index name: `carebot-e5`

### Re-index procedure

```powershell
cd backend
. ..\.venv\Scripts\Activate.ps1

# Confirm .env has: PINECONE_API_KEY=... and PINECONE_INDEX=carebot-e5
python -m app.scripts.build_index
```

Timeline: **~2 minutes** (Pinecone Inference embedding ~45s + upsert ~40s).

Expected output ends with:
```
pinecone_build_done name=carebot-e5 vectors=3424 dim=1024
build_done total_elapsed=~90s
```

### If the PDF changes significantly

1. Drop the old index on Pinecone console: https://app.pinecone.io → carebot-e5 → Delete Index
2. Update `PINECONE_INDEX` in `.env` and `render.yaml` if you want a new name (e.g. `carebot-e5-v2`)
3. Update the eval `expected_pages` in `backend/app/eval/dataset.jsonl` — page numbers will not match a new PDF
4. Run `python -m app.scripts.build_index` — creates the new index, embeds 3424 new chunks
5. Update `PINECONE_INDEX` on Render dashboard → Environment → Save and deploy
6. Verify: run one chat, check the answer still has `[PDF: ... page ...]` citations

---

## 7. Common failures and fixes

### `chat failed: 401 unauthorized`

**Cause:** `PILOT_AUTH_TOKEN` on Render does not match `VITE_PILOT_AUTH_TOKEN` on the frontend.

**Diagnose:**
```powershell
# On the frontend, open DevTools → Network → POST /api/chat → Request Headers
# Look at the Authorization: Bearer <value> — copy it
# Then compare with Render dashboard → careassist-api → PILOT_AUTH_TOKEN
```

**Fix:** copy the exact value from `careassist-api`'s `PILOT_AUTH_TOKEN` into `careassist-web`'s `VITE_PILOT_AUTH_TOKEN`. Save, redeploy the frontend.

**Local variant:** in `backend/`, uvicorn may still have an old `PILOT_AUTH_TOKEN` in its environment from a prior session. Run `echo $env:PILOT_AUTH_TOKEN` in the uvicorn terminal — if it prints a value, that value overrides `.env`. Fix: `Remove-Item Env:PILOT_AUTH_TOKEN` and restart uvicorn.

---

### `chat failed: 502 Bad Gateway`

**Cause (most likely):** OOM. The backend process exceeded 512 MB and was killed mid-request.

**Diagnose:** Render dashboard → `careassist-api` → Logs → look for `Out of memory (used over 512Mi)`.

**Fix:** If you reintroduced a local model (any `sentence-transformers`, `torch`, or `transformers` import), remove it. The current design uses Pinecone Inference — no local model — so this should not happen again. If it does, check:
- `backend/requirements.txt` does not contain `torch`, `transformers`, or `sentence-transformers`
- `backend/app/services/embedder.py` uses `Pinecone(api_key=...).inference.embed()`, not `SentenceTransformer()`

---

### `healthz` returns 200 but `/api/chat` returns 500

**Cause:** retrieval or LLM is failing, but the health check (which does not call them) still works.

**Diagnose:** Render dashboard → `careassist-api` → Logs. Look for the last few lines before the 500:
- `rag_pdf_retrieve_failed` → Pinecone issue
- `web_search_failed` → Tavily issue
- `llm_primary_failed` followed by `llm_call ... fallback=True` → Groq failed, OpenRouter saved it (this is fine)
- `llm_primary_failed` followed by `unhandled_error` → both LLMs failed

**Common causes:**
- Pinecone key expired → regenerate, update Render
- Groq rate limit exceeded (see §8) → wait until next minute or next day
- Groq/OpenRouter both down → temporary, try again in 5 minutes

---

### First message takes 30–50 seconds, subsequent ones are 2–3 seconds

**Cause:** Render free tier cold start. After 15 minutes of inactivity, the container is put to sleep. The first request wakes it up.

**Not a bug.** This is the free tier trade-off.

**Mitigation:**
- Before showing a live demo, click the site yourself to warm it up (one `/healthz` request is enough to wake the container, but the embedder still needs to warm up on the first chat)
- Upgrade to Render Starter ($7/month) — no cold starts

---

### `pinecone.core...UpsertError: Vector dimension 1024 does not match the dimension of the index 384`

**Cause:** You changed the embedder from `all-MiniLM-L6-v2` (384d) to `multilingual-e5-large` (1024d), but the index name still points to the old `carebot` index.

**Fix:** Set `PINECONE_INDEX=carebot-e5` in `.env` and in the Render `careassist-api` environment variables.

---

### `pinecone.core...UnauthorizedException: Invalid API key`

**Cause:** Pinecone key is wrong or expired.

**Fix:** Regenerate at https://app.pinecone.io → API Keys → Create. Update `.env` and Render env var.

---

### `web_search_failed error_type=Unauthorized`

**Cause:** Tavily key expired or invalid.

**Fix:** Regenerate at https://app.tavily.com → API Keys. Update `.env` and Render env var.

**Note:** this does not break chat — `RAGService` catches it, returns PDF chunks only. Users still get answers, just without web fallback.

---

### `llm_primary_failed error_type=RateLimitError` followed by successful fallback

**Cause:** Groq free tier rate limit hit — 8,000 tokens/minute.

**Not a bug.** OpenRouter fallback worked. Users see a slower answer (~4s instead of ~2s) but the answer is correct.

**Diagnose:** Render logs → look at `llm_primary_failed`. If it appears on most requests, you are over quota.

**Fix:** wait for the next minute (per-minute limit) or next day (per-day limit). To increase: upgrade Groq Dev Tier at https://console.groq.com/settings/billing.

---

### Frontend shows `API status: ok` but `chat failed: Failed to fetch`

**Cause:** CORS. The browser is blocking the response even though the server sent it.

**Diagnose:**
```powershell
# Should print {"status":"ok"}
curl.exe -i https://careassist-api.onrender.com/healthz
```

If curl works but browser does not, it is CORS. Check:
```
Render dashboard → careassist-api → Environment → CORS_ORIGINS
```

Should be exactly:
```
https://careassist-web.onrender.com
```

No trailing slash. No `http://`. If it is empty or wrong, fix it and redeploy.

---

### `{"detail":"Not Found"}` on the root URL

**Cause:** There is no route at `/`. Only `/healthz` and `/api/chat` exist.

**Not a bug.** Ignore.

---

## 8. Monitoring and quotas

### Where to check each service

| Service | Dashboard | What to look at |
|---|---|---|
| **Render (backend)** | https://dashboard.render.com → careassist-api → Metrics | CPU, memory, request rate |
| **Render (logs)** | https://dashboard.render.com → careassist-api → Logs | Live log tail, filter by `error` |
| **Groq** | https://console.groq.com → Usage | Daily request count, tokens, remaining quota |
| **Pinecone** | https://app.pinecone.io → Usage | Vector count, read units, inference tokens |
| **Tavily** | https://app.tavily.com → Usage | Credits remaining this month |
| **GitHub** | repo → Actions | CI status (when billing is fixed) |

### Free tier limits

| Service | Limit | What happens at the limit |
|---|---|---|
| **Groq** | 1,000 requests/day, 8,000 tokens/min, 200,000 tokens/day | Falls back to OpenRouter (still works) |
| **Pinecone Inference** | 5M tokens/month | Quota exception → no PDF retrieval, web-only |
| **Pinecone index** | 2 GB storage, 1M read units/month | Quota exception |
| **Tavily** | 1,000 credits/month (1 per search) | Returns empty → PDF-only answers |
| **Render (API)** | 512 MB RAM, 0.1 CPU, 750 hrs/mo, spins down after 15 min idle | OOM if exceeded; 30–50s cold start after idle |
| **Render (web)** | 100 GB bandwidth/mo | Static site blocked until next cycle |

**Realistic capacity for a pilot:**
- ~200 conversations/day (Groq request limit)
- ~5,000 user messages/month before Tavily starts returning empty (only ~1 in 5 messages triggers web fallback, so ~25,000 messages before Tavily is even a concern)

### When to upgrade

| Symptom | Upgrade to | Cost |
|---|---|---|
| Consistent 502 or 500 with no clear bug | Render Starter (2 GB RAM) | $7/mo |
| Cold-start is annoying during demos | Render Starter | $7/mo |
| Hitting Groq 1,000/day | Groq Dev Tier | pay-as-you-go |
| Hitting Tavily 1,000/month | Tavily paid tier | $0.008/credit |
| Needing multiple workers | Add Redis back (paid KV) | $10/mo |

---

## 9. Emergency shutdown

If the app is doing something dangerous (answering medical questions wrong, being abused, leaked secret):

### Option A — disable chat only, keep the UI

Add an env var on Render `careassist-api`:
```
LLM_MOCK=true
```
Save and deploy. Now every chat returns `[mock] You said: ...` — no real LLM call, no retrieval. Frontend still loads, healthz still green.

To restore: set `LLM_MOCK=false` and redeploy.

### Option B — take the API offline

```
Render dashboard → careassist-api → Settings → Suspend Service
```

Frontend will show "API unreachable" but will not crash. To restore: resume the service.

### Option C — take everything offline

Suspend both `careassist-api` and `careassist-web`.

### Option D — rotate `PILOT_AUTH_TOKEN`

Fastest way to kick out every existing user (requires them to get the new token from you):

1. Generate a new random 43-char string
2. Update Render `careassist-api` → `PILOT_AUTH_TOKEN` → Save and deploy
3. Update Render `careassist-web` → `VITE_PILOT_AUTH_TOKEN` → Save and deploy
4. Update local `.env` and `frontend/.env.local` if you continue developing

All currently-open browser sessions get 401 on next request.

---

## 10. What NOT to do

### ❌ Do not commit `.env` or `frontend/.env.local`

Both are in `.gitignore`. If you ever see them in `git status --short` before a commit, something is wrong. Verify with:
```powershell
git check-ignore .env backend/.env frontend/.env.local   # all should print the path
```

### ❌ Do not add back `sentence-transformers` / `torch` / `transformers`

The whole reason the app fits in 512 MB is that Pinecone hosts the embedding model. If you add a local model back, you will OOM.

### ❌ Do not run `uvicorn --workers 2` without adding Redis

Two workers means two independent in-memory session stores → user history flips between workers. Either stay at `--workers 1` (current) or add Redis back and set `REDIS_URL`.

### ❌ Do not put user message content in logs

The `test_logs.py` regression test enforces this. If you add a `print(user_message)` or `logger.info(f"... {user_message}")`, that test fails and it is a real privacy bug.

### ❌ Do not change `SESSION_SECRET` on a running system

Doing so invalidates every existing user's session cookie. They will not lose data (there is no persisted data), but they will need to wait 30 minutes or clear cookies.

### ❌ Do not use `git reset --hard` on pushed commits

Use `git revert` instead. `reset --hard` on shared history requires `--force` and breaks any other contributor.

### ❌ Do not run `python -m app.eval.run_eval` in a loop

Each run uses real Groq / Pinecone / Tavily credits. 5 cases = ~30 seconds and ~$0.001. Twenty runs in a debug loop = $0.02 and 100 Groq requests. Not free but not free either.

---

## 11. Architecture in one diagram

```text
┌──────────────────────────────────────────────────────────────────┐
│  User's browser                                                  │
│  └─ https://careassist-web.onrender.com (React static site)     │
│     ├─ POST /api/chat  ── with session cookie + Bearer token ──┐ │
└────────────────────────────────────────────────────────────────│─┘
                                                                 │
┌────────────────────────────────────────────────────────────────▼─┐
│  careassist-api (FastAPI on Render, 512 MB, 1 worker)            │
│                                                                  │
│  1. CORS check                                                   │
│  2. Request-ID middleware                                        │
│  3. Body-size limit (16 KB)                                      │
│  4. Rate limit (30/min, in-memory)                               │
│  5. Session cookie → session_id                                  │
│  6. Load session history from RAM                                │
│                                                                  │
│  7. RAGService.retrieve(question):                               │
│     ├─► Pinecone Inference  → 1024-dim vector                    │
│     ├─► Pinecone Index      → top 3 chunks                       │
│     └─► if top score < 0.4:                                      │
│         └─► Tavily search   → 5 web snippets                     │
│                                                                  │
│  8. LLMService.complete(question + context + history):           │
│     ├─► Groq (openai/gpt-oss-120b)     ← primary                 │
│     └─► OpenRouter (llama-3.3-70b)     ← fallback                │
│                                                                  │
│  9. Save exchange to session history                             │
│ 10. Return { reply, model, sources, request_id }                 │
└──────────────────────────────────────────────────────────────────┘
```

---

## 12. Known limitations (honest list)

1. **Single worker.** All session history is in-memory. Adding a second worker requires Redis.
2. **No persistence across restarts.** Render restarts the container periodically; all sessions are lost.
3. **Shared `PILOT_AUTH_TOKEN`.** Every pilot user uses the same token. No per-user identity.
4. **No clinical validation of answers.** `docs/eval/latest.md` documents the current eval. Medical correctness beyond citation grounding is a Stage 9 pre-pilot deliverable — see `docs/eval/review.md` (to be written).
5. **512 MB RAM ceiling.** Fine for current design. Add any local model → OOM.
6. **Cold starts.** 30–50s after 15 min idle. Not fixable on free tier.
7. **No CI/CD.** GitHub Actions is billing-locked. Deploys are `git push` → Render auto-deploy. No test gate before merge.
8. **`--reload` only in dev.** Production runs without it.

---

## Contact

**Operator:** Idowu Thomas
**Email:** ayanfeoluwadegoke@gmail.com
**Privacy notice:** [docs/privacy.md](privacy.md)