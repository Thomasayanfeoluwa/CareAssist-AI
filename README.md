# CareAssist AI

An educational health-information assistant that answers health questions from a 637-page medical textbook, cites exact page numbers, and falls back to web search when the book does not cover a topic.

**Live pilot:** https://careassist-web.onrender.com  
**Status:** Closed pilot. Educational use only. Not a medical device.

> **Not for diagnosis, treatment, or triage.** Every answer ends with a reminder to consult a qualified health professional. See [docs/privacy.md](docs/privacy.md) for what data leaves your browser.

---

## What it does

You ask: *"What are the side effects of aspirin?"*

It answers:

> Aspirin can cause a range of side effects... The most common ones include stomachache or heartburn, loss of appetite, and small amounts of blood in the stool **[PDF: Medical book, page 393]**.
>
> **Reminder:** Always consult a qualified health professional for your specific situation.

Plus three clickable source chips (`PDF p.393`, `PDF p.393`, `PDF p.189`) that open a hoverable preview of the exact passage the model cited.

Under the hood:

1. **Retrieve** — embed the question with Pinecone Inference (`multilingual-e5-large`, 1024 dims), search a 3,424-chunk index built from `Data/Medical_book.pdf`, take the top 3 chunks by cosine similarity
2. **Fallback** — if the best score is below 0.4, search the web via Tavily for current information
3. **Generate** — call Groq's `openai/gpt-oss-120b`, falling back to OpenRouter's `llama-3.3-70b-instruct` on failure
4. **Cite** — every PDF chunk carries its page number, so citations are grounded in what was actually retrieved

---

## Stack

| Layer          | Choice                                      | Why                                              |
|----------------|---------------------------------------------|--------------------------------------------------|
| Backend        | FastAPI + Uvicorn                           | Typed request/response, async, OpenAPI           |
| Frontend       | React 18 + TypeScript + Vite + Tailwind     | Typed, fast builds, no framework bloat           |
| Retrieval      | Pinecone (1024-dim, cosine)                 | Serverless vector DB with hosted embeddings      |
| Embeddings     | Pinecone Inference `multilingual-e5-large`  | No local model → fits in 512 MB RAM              |
| LLM primary    | Groq `openai/gpt-oss-120b`                  | Fast, cheap, good medical Q&A                    |
| LLM fallback   | OpenRouter `llama-3.3-70b-instruct`         | Different provider, different model family       |
| Web fallback   | Tavily                                      | LLM-optimized search, returns clean snippets     |
| Session memory | In-process (single worker)                  | 30-min TTL, no external dependency               |
| Hosting        | Render (free tier)                          | Backend + frontend + free HTTPS                  |

---

## Local development

### Prerequisites

- **Python 3.11** (3.12 will likely work; 3.13 may not)
- **Node 20** or higher
- **uv** (optional, faster than `venv`; `python -m venv` works too)
- A **Pinecone** account and API key
- A **Groq** API key
- A **Tavily** API key
- An **OpenRouter** API key (for fallback)

### Setup

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

# Fill secrets
Copy-Item .env.example .env
# Edit .env with real values

# Frontend env (gitignored)
# Create frontend/.env.local with:
#   VITE_API_BASE_URL=http://localhost:8000
#   VITE_PILOT_AUTH_TOKEN=<same value as PILOT_AUTH_TOKEN in .env>
```

### Run

Two terminals, both with the venv active:

```powershell
# Terminal 1 — backend
cd backend
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2 — frontend
cd frontend
npm run dev
```

Open http://localhost:5173.

The `--reload` flag is required during development. Without it, `.py` changes do not take effect until you restart.

### Test

```powershell
cd backend
python -m pytest tests/ -q           # 41 tests, ~15s, no API calls
python -m app.eval.run_eval          # 5-case eval, ~30s, uses real API credits
```

---

## Deploy

Deploys are handled by Render's Blueprint system reading `render.yaml`:

- `careassist-api` — FastAPI backend (Python 3.11)
- `careassist-web` — React static site (built with `npm ci && npm run build`)

Every `git push origin main` triggers a rebuild + deploy of both services.

### First deploy

1. Fork or clone this repo to your own GitHub
2. In Render: **New + → Blueprint →** select the repo
3. Fill in the `sync: false` secrets in the dashboard  
   (`GROQ_API_KEY`, `OPENROUTER_API_KEY`, `PINECONE_API_KEY`, `TAVILY_API_KEY`, `PILOT_AUTH_TOKEN`)
4. After the deploy is live, run the index build once  
   (see [docs/runbook.md](docs/runbook.md) §6)

Full deployment, rollback, and troubleshooting steps live in [docs/runbook.md](docs/runbook.md).

---

## Documentation

- **[Privacy notice](docs/privacy.md)** — every third party your message reaches, retention, deletion
- **[Operations runbook](docs/runbook.md)** — deploy, rollback, key rotation, common failures, quota limits
- **[Evaluation report](docs/eval/latest.md)** — current eval results with scope disclosure

---

## What this is NOT

- ❌ **Not a medical device.** Not FDA, EMA, NAFDAC, or any regulator cleared.
- ❌ **Not for diagnosis, treatment, or dosing.** Every answer directs users to a clinician.
- ❌ **Not HIPAA or GDPR compliant.** Do not enter real patient data. See [docs/privacy.md](docs/privacy.md).
- ❌ **Not multi-tenant.** Every pilot user shares the same bearer token. Sessions are isolated by cookie only.
- ❌ **Not a substitute for a clinician.** Ever.

---

## Known limitations

| Limitation                                                  | Workaround / Future                                             |
|-------------------------------------------------------------|-----------------------------------------------------------------|
| Single worker, in-memory sessions                           | Add Redis + `--workers 2` when upgrading                        |
| Cold start after 15 min idle (~30-50s)                      | Upgrade to Render Starter ($7/mo)                               |
| Eval covers 5 cases with clinically-unreviewed page numbers | Clinical review is a Stage 9 pre-pilot deliverable              |
| No CI/CD (GitHub Actions billing-locked)                    | Render auto-deploy on git push; wire actions when billing fixed |
| 512 MB RAM ceiling                                          | Comfortable at current design; no headroom for local models     |

---

## Contributing

This is a closed pilot, not an open-source project. Do not open PRs without contacting the operator first.

If you are the operator (hi, future Thomas), see the [runbook](docs/runbook.md) for how to make changes safely.

---

## License

MIT License. See [LICENSE](LICENSE) for the full text.

---

## Contact

**Operator:** Idowu Thomas  
**Email:** ayanfeoluwadegoke@gmail.com  
**Privacy:** [docs/privacy.md](docs/privacy.md)
```
