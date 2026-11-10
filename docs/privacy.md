# CareAssist AI — Privacy Notice

**Last updated:** 2026-11-10
**Status:** Closed pilot. Educational use only. Not a medical device.

This document explains what data CareAssist AI processes, where it goes, how long it is kept, what we can and cannot do about removal, and how to reach us with a privacy question.

---

## Short version

- Your **messages are not stored** on our servers beyond 30 minutes of in-session memory (single session, cleared on restart).
- Your messages **are sent to third-party AI services** to generate an answer: Groq (primary), OpenRouter (fallback), Pinecone (retrieval), Tavily (web fallback).
- We **do not** sell, share, or use your messages to train any model.
- We **do not** have a database of past conversations. Refreshing the page starts a new session.
- If you would not put it in a web search box, do not put it here.

**This system is not HIPAA, GDPR, or any other regulatory framework compliant.** It is a pilot. Do not use it with real identifying medical information.

---

## 1. What we collect

### 1.1 When you send a message

| Data | Where it lives | How long |
|---|---|---|
| Your message text | Sent to Groq/OpenRouter/Pinecone/Tavily (see §3) | Immediate processing only |
| Your message text | In-process session memory (RAM on our server) | Up to 30 minutes, cleared on restart |
| Assistant's reply | Sent back to your browser | Immediate |
| Assistant's reply | In-process session memory | Up to 30 minutes |
| Session ID | Signed cookie in your browser | 30 minutes |
| Request metadata (timestamp, IP, response size) | Render access log | ~7 days (Render default) |

**What we do NOT collect:** name, email, phone, location, device IDs, or any account credentials.

### 1.2 When you browse the site

| Data | Where | Why |
|---|---|---|
| `careassist_session` cookie | Your browser | Signed session ID so the server can return your own chat history to you |
| Cloudflare logs | Cloudflare (Render's CDN) | Standard web traffic logs |

The cookie is:
- **HttpOnly** — JavaScript cannot read it
- **SameSite=Lax** — not sent on cross-site requests
- **Secure** — sent only over HTTPS
- **Signed** — tampering is rejected server-side
- **Not tracked across sites** — no analytics, no ads

We do **not** use Google Analytics, Facebook Pixel, or any third-party tracking.

---

## 2. What we do NOT collect

- ❌ No account registration
- ❌ No email addresses
- ❌ No passwords
- ❌ No IP addresses stored beyond what Render's access log contains
- ❌ No device fingerprints
- ❌ No behavioral analytics
- ❌ No cookies beyond the single session cookie above
- ❌ No data used for training any model

---

## 3. Third parties — exactly what is sent, to whom

Every chat request goes through three or four external services. Here is the complete list.

### 3.1 Groq (primary language model)

- **Purpose:** Generate the answer to your question.
- **What is sent:**
  - Your question (verbatim)
  - The top 3 retrieved passages from the medical book (from Pinecone)
  - The last up to 5 turns of your conversation in this session
  - A fixed system prompt that instructs the model to answer educationally, cite sources, refuse personalized advice, and escalate emergencies
- **What is NOT sent:** your IP address, your cookie, your browser details
- **Retention by Groq:** Per Groq's terms, API inputs/outputs are not used for model training and are retained for abuse monitoring for **up to 30 days**. See https://groq.com/privacy-policy/
- **Model:** `openai/gpt-oss-120b` (as of last update)

### 3.2 OpenRouter (fallback language model)

- **Purpose:** Generate the answer only if Groq fails or rate-limits.
- **What is sent:** Same as Groq — your question, retrieved passages, recent history, system prompt.
- **Retention by OpenRouter:** Per their policy, not used for training by the default routing. See https://openrouter.ai/privacy
- **Model:** `meta-llama/llama-3.3-70b-instruct`

### 3.3 Pinecone (vector database + embedding service)

- **Purpose (retrieval):** Find the 3 most relevant passages from `Medical_book.pdf` for your question.
- **What is sent (embedding step):** Your question, to Pinecone's **Inference API** (`multilingual-e5-large` model) to convert it into a numeric vector.
- **What is returned:** The 3 best-matching text chunks from the pre-indexed medical book, plus their page numbers.
- **What is NOT sent:** your cookie, your IP, your previous messages.
- **Retention:** Processed per Pinecone's privacy policy — see https://www.pinecone.io/privacy-policy/

### 3.4 Tavily (web search fallback)

- **Purpose:** When the medical book does not contain strong matches for your question, search the web for relevant health information.
- **What is sent:** **Only your question.** No context, no history.
- **What is returned:** Up to 5 short snippets with URLs.
- **When it fires:** Only if the top PDF match has a similarity score below 0.4.
- **Retention:** Tavily retains query logs for 30 days for abuse monitoring. See https://tavily.com/privacy

### 3.5 Render (hosting)

- **Purpose:** Hosts the web app and the backend.
- **What is stored:** Standard HTTP access logs (URL path, status code, timestamp, IP). These do **not** include your message content — our application logs never write user content (verified by our test suite).
- **Retention:** ~7 days on free tier. See https://render.com/privacy

### 3.6 GitHub (source code)

- **Purpose:** Hosts the code.
- **What is stored:** The public source code. **No user data.**
- **Note:** The code repository is public, but no chat data, no logs, and no secrets are in the repository. The `.env` file is gitignored and has never been committed.

---

## 4. Data flow diagram

```text
Your browser
   │
   │ "What are the side effects of aspirin?"
   ▼
[1] Frontend (careassist-web.onrender.com)
   │  POST /api/chat  +  session cookie
   ▼
[2] Backend (careassist-api.onrender.com)
   │
   ├──► [3] Pinecone Inference       — embeds your question (1024-dim vector)
   │        ◄── 1024-dim vector
   │
   ├──► [4] Pinecone Index "carebot-e5"
   │        ◄── top 3 chunks from Medical_book.pdf (page 393, 393, 189)
   │
   │    if top score < 0.4:
   ├──► [5] Tavily                   — sends your question to web search
   │        ◄── 5 web snippets
   │
   ├──► [6] Groq (or [7] OpenRouter fallback)
   │        ◄── generated answer
   │
   ▼
[2] Backend streams answer + sources to browser
   │
   ▼
Your browser renders the answer with citations
```
---

## 5. Retention — the complete table

| Data | Where | How long | Who can see it |
|---|---|---|---|
| Your message | RAM on the API server | ≤ 30 minutes, or until restart | Server process only |
| Assistant reply | RAM on the API server | ≤ 30 minutes, or until restart | Server process only |
| Session cookie | Your browser | 30 minutes | Your browser + server |
| HTTP access logs | Render | ~7 days | Render ops |
| Groq API request | Groq servers | ~30 days | Groq |
| OpenRouter API request (if fallback) | OpenRouter servers | Per their policy | OpenRouter |
| Pinecone inference request | Pinecone servers | ~30 days | Pinecone |
| Tavily search (if fallback) | Tavily servers | ~30 days | Tavily |
| Database of conversations | **Does not exist** | N/A | Nobody |

**On our servers there is no database, no file, and no log that contains your chat messages.**

---

## 6. Your rights

Because this system has no user accounts and no stored conversations, the
traditional "data subject rights" apply differently here. Below is what we
can actually do, and what we cannot.

### 6.1 What we can do

**Clear your active session now** — if you asked a question and want the
in-memory session dropped immediately (rather than waiting for its 30-minute
expiry), email us with the subject "CareAssist session clear" and we will
clear it on the running server within 1 hour during business hours.

**Answer what was transmitted** — we can tell you exactly what data was sent
where during your visit. The complete answer is always the same shape:

  - Your message text → sent to Groq (and OpenRouter if Groq was rate-limited)
  - Your message text → sent to Pinecone's Inference API (converted to a vector)
  - Your message text → sent to Tavily (only if PDF retrieval was weak)
  - Your message and the assistant's reply → held in server RAM for ≤ 30 minutes
  - Your IP, timestamp, and request path → in Render's access log for ~7 days

We can confirm the general shape, but we **cannot** confirm a specific
individual message from a specific individual user, because:
  - We do not verify who you are (no accounts, no logins)
  - Access logs contain no message content, so we cannot search them
  - We have no visibility into Groq/Pinecone/Tavily's internal logs

### 6.2 What we cannot do

**We cannot delete your data from third-party services.** Groq, OpenRouter,
Pinecone, and Tavily each have their own retention policies (see §3). We do
not have the ability to force them to delete specific records, and we do not
have a "customer support" relationship with them that would allow such a
request to be honored.

If you want to make a deletion request directly to a third party, here are
their privacy contacts:

  - Groq: https://groq.com/privacy-policy/
  - OpenRouter: https://openrouter.ai/privacy
  - Pinecone: https://www.pinecone.io/privacy-policy/
  - Tavily: https://tavily.com/privacy

**We cannot verify your identity.** Because there are no accounts, there is no
way for us to know that the person emailing us is the same person who sent a
particular message. This is a deliberate design choice — collecting identity
would create the very data-retention problem we are trying to avoid — but it
also means "delete my message" is not a request we can act on reliably.

**We cannot delete from Render's access logs.** Those are managed by Render
per their retention policy (~7 days). After 7 days, they are gone
automatically. We have no mechanism to trigger deletion sooner.

### 6.3 What you can do yourself

If you want certainty about what is retained:

  - **Now:** clear your browser cookies for careassist-web.onrender.com. This
    ends your session; server memory expires within 30 minutes.
  - **In 30 minutes:** your message and the assistant's reply are gone from
    our server's RAM.
  - **In 7 days:** the access-log entry is gone from Render.
  - **In 30 days:** any request/response logs held by Groq, Pinecone, or
    Tavily are gone per their standard abuse-monitoring retention.

### 6.4 The honest summary

If you do not want your health question processed by third-party AI services
and retained by them for up to 30 days, **do not use this app**. There is no
technical mechanism that lets you use CareAssist without your message reaching
Groq, Pinecone, and possibly Tavily. That is the fundamental trade-off of any
cloud-hosted LLM application.

For anything not covered above — or if you believe this notice is inaccurate
— email **ayanfeoluwadegoke@gmail.com** with the subject "CareAssist privacy question"
and we will respond within 7 days with an honest answer.
---

## 7. Children

CareAssist is not intended for users under 18. Do not submit questions on behalf of a minor without a parent/guardian's consent.

---

## 8. What this system is NOT

- ❌ **Not a medical device.** Not FDA, EMA, NAFDAC, or any other regulator cleared.
- ❌ **Not for diagnosis, treatment, or triage.**
- ❌ **Not HIPAA compliant.** Do not enter patient-identifying information.
- ❌ **Not GDPR compliant in a formal sense.** This is a pilot; a formal DPA has not been signed with any third party for this use.
- ❌ **Not a substitute for a clinician.** Every answer ends with "consult a qualified health professional."
- ❌ **Not multi-tenant.** All pilot users share the same `PILOT_AUTH_TOKEN`. There is no per-user identity on the server. The only isolation is by browser session cookie.

**Do not use CareAssist with real patient data.** Do not use it for a medical decision. Do not use it as your only source.

---

## 9. Changes to this notice

Any change to this document will be committed to the public repository at `github.com/Thomasayanfeoluwa/HealthAssistant-AI/blob/main/docs/privacy.md` with a visible commit message. The "Last updated" date at the top reflects the most recent change.

---

## 10. Contact

**Operator:** Thomas Ayanfeoluwa (pilot operator)
**Email:** **ayanfeoluwadegoke@gmail.com**
**Repository:** https://github.com/Thomasayanfeoluwa/HealthAssistant-AI
**Live app:** https://careassist-web.onrender.com

For anything related to this notice — access, deletion, correction, or a question about what data was transmitted — use the email above.

---

## 11. One more honest disclosure

This document is written by the developer of a pilot project, not by a lawyer. It represents the actual system behavior accurately and completely, but it is not legal advice. If you are deploying CareAssist in a regulated context (healthcare provider, insurer, government), you **must** engage qualified counsel and a data protection officer before using it with real users.