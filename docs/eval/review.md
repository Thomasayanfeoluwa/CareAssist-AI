# CareAssist — Evaluation Review Log

**Purpose:** record what has been verified about the eval dataset, by whom, and what remains for clinical sign-off.

---

## Verification status

| Case | Pages verified? | Expected facts written? | Clinically reviewed? |
|---|---|---|---|
| aspirin-side-effects | ✅ 2026-11-11 (I) | ✅ | ❌ |
| hypertension-treatment | ✅ 2026-11-11 (I) | ✅ | ❌ |
| type2-diabetes-causes | ✅ 2026-11-11 (I) | ✅ | ❌ |
| redflag-chest-pain | N/A (no page citations) | N/A (only refusal check) | ❌ |
| dosage-trap-aspirin | N/A (no page citations) | N/A (only dosage check) | ❌ |

---

## Page verification detail

Pages were read directly from `Data/Medical_book.pdf` and classified as
**Yes / Partially / No** based on whether the page discusses the case's topic.

### Page decisions from 2026-11-11 review

| Page | Case | Verdict | Reason |
|---|---|---|---|
| 189 | aspirin-side-effects | **No** (removed) | Page is about narcotic analgesics and NSAIDs in general; aspirin mentioned once in passing ("NSAIDs are contraindicated in patients allergic to aspirin"). Not about aspirin. |
| 217 | hypertension-treatment | Yes | Entire page on treating high blood pressure with ACE inhibitors |
| 218 | hypertension-treatment | Yes | Continuation of ACE inhibitor section (precautions, interactions) |
| 219 | hypertension-treatment | **Partially** (removed) | First half finishes ACE inhibitor entry; second half is about the ACE *test* for sarcoidosis. Ambiguous — removed to avoid false positives. |
| 275 | type2-diabetes-causes | **No** (removed) | Page is about drug interactions and KEY TERMS; no diabetes content |
| 276 | type2-diabetes-causes | Yes | Full Type I vs Type II discussion with insulin resistance |
| 293 | hypertension-treatment | Yes | Start of "Antihypertensive drugs" entry with drug classes |
| 392 | aspirin-side-effects | Yes | Full Aspirin entry (purpose, description, start of precautions) |
| 393 | aspirin-side-effects | Yes | Continuation of Aspirin entry (side effects) |

---

## What "verified" means in this document

- ✅ **Pages verified** — I read the actual page text and confirmed it discusses the topic. This rules out false positives in the `retrieval_hit` check.
- ✅ **Expected facts written** — the developer wrote 3-5 factual claims that a medically correct answer should contain, based on standard medical knowledge. These claims are checked by an LLM-as-judge.
- ❌ **Clinically reviewed** — a person with medical training has read the actual model replies and confirmed they are medically sound.

**Only the last item requires medical training.** The first two can be done by anyone who can read.

---

## What remains for clinical sign-off (S9.1c)

For each case, a reviewer (medical student, nurse, pharmacist, or physician) needs to:

1. Open `docs/eval/latest.md`
2. Read the case's question, expected facts, and the actual model reply
3. Answer: **"Are the facts stated in this reply medically accurate and appropriately stated?"**
4. Mark "Reviewed: [name], [date]" or note concerns

Estimated time: **30 minutes total for 5 cases.**

### Contact for reviewer

If you are a medical professional willing to review 5 short Q&A cases for a pilot project, email **ayanfeoluwadegoke@gmail.com** with the subject "CareAssist clinical review". You will receive:
- A one-page document with each question, expected facts, and the model's reply
- 30 minutes of work
- No financial compensation (pilot project)
- Full credit in this document

---

## Known limitations of this review process

1. **The expected facts were written by the developer, not a clinician.** They represent standard medical knowledge but may miss important nuances that a trained reviewer would catch.
2. **The LLM-as-judge can fail** — if the judge model misinterprets a fact or the reply, `facts_present` may pass a bad answer or fail a good one. Spot-checking the eval report manually is still recommended.
3. **5 cases is not a statistically meaningful sample.** A 30+ case dataset (S9.2) is the proper scale for a real pilot, and is deferred.
4. **Passing every check does not mean the answer is correct.** It means the answer:
   - Cited a page that was actually retrieved
   - Contained the expected factual claims
   - Did not leak dosages
   - Did not use brand names
   - Escalated when a red-flag question was asked
   
   A medically wrong answer that happens to contain all the "expected facts" (because we wrote them wrong) would still pass. That is why S9.1c matters.