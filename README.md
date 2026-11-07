# CareAssist AI

Educational health-information assistant. Closed pilot. Shared bearer-token auth.

## Stack
- Frontend: React 18 + TypeScript + Vite + Tailwind (Render static site)
- Backend: FastAPI + Uvicorn (Render web service)
- LLM: LiteLLM → Groq `llama-3.3-70b-versatile` (primary), OpenRouter (fallback)
- Retrieval: Pinecone, `sentence-transformers/all-MiniLM-L6-v2`
- Fallback search: Google Custom Search
- Session memory: Redis (per-session, TTL 30 min, 5 exchanges)

## Local setup
1. `cp .env.example .env` and fill every required key.
2. `make dev` — starts backend on :8000 and frontend on :5173.
3. Open http://localhost:5173.

## Deploy
Push to `main`. `render.yaml` provisions `careassist-api`, `careassist-web`,
and `careassist-redis`. Set the `sync: false` secrets in the Render dashboard
on first deploy.

## Limitations (pilot)
- Not a medical device. No diagnosis, no dosing, no personalized treatment.
- Educational content only. Users are directed to a clinician for decisions.
- Evaluation coverage is documented in `docs/eval/latest.md`.

## Legacy
Prior Flask and Streamlit implementations are in `legacy/`. They are not
deployed and are kept for reference only. See `legacy/README.md`.