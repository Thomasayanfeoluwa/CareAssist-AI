# CareAssist

An educational health-information assistant that answers health questions from a 637-page medical textbook, cites exact page numbers, and falls back to web search when the book does not cover a topic.

**Live pilot:** https://careassist-web.onrender.com  
**Status:** Closed pilot. Educational use only. Not a medical device.

> **Not for diagnosis, treatment, or triage.** Every answer ends with a reminder to consult a qualified health professional. See [docs/privacy.md](docs/privacy.md) for what data leaves your browser.

---

## What it does

You ask: *"What are the side effects of aspirin?"*

It answers:

> Aspirin can cause a range of side effects... The most common ones include
> stomachache or heartburn, loss of appetite, and small amounts of blood in
> the stool **[PDF: Medical book, page 393]**.
>
> **Reminder:** Always consult a qualified health professional for your
> specific situation.

Plus clickable source chips (`PDF p.393`, `PDF p.393`, `PDF p.189`)
that open a hoverable preview of the exact passage the model cited.

Under the hood, each request runs through six steps:

1. **Rewrite** — if the question is a short follow-up (e.g. "what about
   children?"), a lightweight LLM rewrites it into a standalone query that
   preserves the original topic. Anchors on the previous *user* message, not
   the assistant's reply, and falls back to string concatenation if the
   rewrite drifts.
2. **Retrieve** — embed the query with Pinecone Inference
   (`multilingual-e5-large`, 1024 dims) and search an index built
   from `Data/Medical_book.pdf` (`carebot-e5`) for the top 3 chunks by cosine similarity.
3. **Judge relevance** — an LLM reviews the retrieved chunks using a mechanical anchor check
   (verifying specific guidelines, years, agencies, doses, or schedules).
   Scores are evaluated on a 1–5 scale: scores of **4 or 5** ("passages directly answer with
   the specific facts requested") are accepted as sufficient. Scores of **3 or below**
   (or missing anchors) fire the web fallback, ensuring the assistant never serves a false
   answer from topically-adjacent but incomplete book passages.
4. **Web fallback (Tavily)** — if the PDF chunks do not fully answer the
   question, search the web for current, authoritative medical information. Used for topics the
   textbook does not cover (recent guidelines, vaccine schedules, drug
   comparisons).
5. **Generate** — call Groq's `openai/gpt-oss-120b`. On failure the service
   retries up to 3 times, honouring Groq's `retry-after` hint before each
   retry. If all retries fail, fall back to OpenRouter (`meta-llama/llama-3.3-70b-instruct` or `openai/gpt-4o-mini`).
6. **Cite** — citations are strictly grounded in what was retrieved: exact page numbers for PDF passages (`[PDF: Medical book, page X]`) and source URLs when web fallback is utilized.

---

## Stack

| Layer          | Choice                                      | Why                                              |
|----------------|---------------------------------------------|--------------------------------------------------|
| Backend        | FastAPI + Uvicorn                           | Typed request/response, async, OpenAPI           |
| Frontend       | React 18 + TypeScript + Vite + Tailwind     | Typed, fast builds, no framework bloat           |
| Retrieval      | Pinecone (1024-dim, cosine, `carebot-e5`)   | Serverless vector DB with hosted embeddings      |
| Embeddings     | Pinecone Inference `multilingual-e5-large`  | No local model → fits in 512 MB RAM              |
| LLM primary    | Groq `openai/gpt-oss-120b`                  | Fast, cheap, good medical Q&A                    |
| LLM fallback   | OpenRouter `llama-3.3-70b-instruct` / `gpt-4o-mini` | Different provider & model family for failover resilience |
| Web fallback   | Tavily                                      | LLM-optimized search, returns clean snippets     |
| Session memory | In-memory / Redis (`RedisSessionStore`)     | 30-min TTL; in-memory for dev, Redis for multi-worker prod |
| CI/CD          | GitHub Actions                              | Ruff, MyPy strict, Pytest, Vite build, Eval gate & Render deploy hook |
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
git clone https://github.com/Thomasayanfeoluwa/CareAssist-AI.git
cd CareAssist-AI

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

### Test & Quality Checks

```powershell
# Backend lint, typecheck, and unit tests
cd backend
ruff check app tests                 # Linting and style rules
mypy app                             # Strict type checking across backend
python -m pytest tests/ -q           # 41 tests, ~13s, no API calls

# Automated evaluation (requires live API keys, 30s pacing, ~2.5 min)
python -m app.eval.run_eval          # 5-case RAG eval (retrieval, citation grounding, dosage check, LLM fact judge)

# Frontend typecheck & build
cd ../frontend
npm run typecheck                    # TypeScript compiler verification
npm run build                        # Production bundle build
```

---

## Deploy & CI/CD Pipeline

Continuous Integration and Deployment is handled via **GitHub Actions** (`.github/workflows/ci.yml`) and **Render**:

1. **Backend CI**: Runs on all pushes and pull requests (`Python 3.11`). Enforces `ruff check`, `mypy app` (strict mode), and `pytest -q` (41 unit tests).
2. **Frontend CI**: Runs on all pushes and pull requests (`Node 20`). Enforces `npm run typecheck` and `npm run build`.
3. **Automated Evaluation Gate**: On push to `main`, executes the 5-case evaluation suite (`python -m app.eval.run_eval --limit 5`) with `LLM_TEMPERATURE=0.0` for deterministic scoring. Generates and uploads the eval report artifact (`docs/eval/latest.md`).
4. **Deploy Gate**: Once `backend`, `frontend`, and `eval` jobs all pass, GitHub Actions triggers the Render deploy webhook (`RENDER_DEPLOY_HOOK_URL`) to deploy:
   - `careassist-api` — FastAPI backend (Python 3.11, Virginia region)
   - `careassist-web` — React static site

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
| In-memory sessions by default in dev                        | `RedisSessionStore` is implemented; set `REDIS_URL` + `--workers 2` when upgrading |
| Cold start after 15 min idle (~30-50s)                      | Upgrade to Render Starter ($7/mo)                               |
| Eval covers 5 automated benchmark cases                     | Pages verified in `dataset.jsonl`; ongoing clinical review documented in `docs/eval/` |
| 512 MB RAM ceiling on free tier                             | Cloud embeddings via Pinecone Inference preserve low RAM footprint |

---

## Contributing

This is a closed pilot, not an open-source project. Do not open PRs without contacting the operator first.

For operational guidance and safe changes, see the [runbook](docs/runbook.md).

---

## License

MIT License. See [LICENSE](LICENSE) for the full text.

---

## Contact

**Operator:** Idowu Thomas  
**Email:** ayanfeoluwadegoke@gmail.com  
**Privacy:** [docs/privacy.md](docs/privacy.md)
```
