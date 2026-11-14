from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.api.routes import _filter_chunks_by_score
from app.config import get_settings
from app.eval.metrics import judge_facts, score_case
from app.eval.schema import EvalCase
from app.logging_setup import configure_logging
from app.prompts.system import build_system_prompt
from app.services.llm import LLMService
from app.services.rag import RAGService

DATASET_PATH = Path(__file__).resolve().parent / "dataset.jsonl"
REPORT_DIR = Path(__file__).resolve().parents[3] / "docs" / "eval"


def load_cases(limit: int | None = None) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(EvalCase.from_dict(json.loads(line)))
    return cases[:limit] if limit else cases


async def run_case(
    case: EvalCase, rag: RAGService, llm: LLMService
) -> dict[str, Any]:
    start = time.perf_counter()
    chunks = []
    reply = ""
    error = None
    try:
        chunks = await rag.retrieve(case.question)
        chunks = _filter_chunks_by_score(chunks)
        context = rag.assemble_context(chunks)
        user_prompt = (
            f"### Retrieved context:\n{context or '(no relevant context found)'}\n\n"
            f"### Current question:\n{case.question}"
        )
        result = await llm.complete(build_system_prompt(), user_prompt)
        reply = result.content
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - start

    # LLM-as-judge check on expected facts (skipped if no facts or if the
    # main call already errored — a judge on empty reply is meaningless).
    if error or not case.expected_facts:
        facts_pass = True
        facts_detail: list[dict[str, Any]] | str = ""
    else:
        facts_pass, facts_detail = await judge_facts(
            llm, case.question, case.expected_facts, reply
        )

    # Static checks: retrieval hit, citation grounding, no dosage, refusal, etc.
    scores = score_case(case, chunks, reply, error)

    # Add the LLM-judge result as another score, so the "pass" computation
    # below includes it automatically.
    scores["facts_present"] = {
        "pass": facts_pass,
        "detail": "" if facts_pass else str(facts_detail),
    }

    return {
        "case_id": case.id,
        "category": case.category,
        "question": case.question,
        "reply": reply,
        "retrieved_pages": [c.page for c in chunks if c.source == "pdf"],
        "elapsed_s": round(elapsed, 2),
        "error": error,
        "scores": scores,
        "pass": all(v["pass"] for v in scores.values()),
    }


async def main_async(limit: int | None) -> int:
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL)
    cases = load_cases(limit=limit)
    print(f"Running {len(cases)} eval case(s)...")
    rag = RAGService(settings)
    llm = LLMService(settings)
    results: list[dict[str, Any]] = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case.id}")
        r = await run_case(case, rag, llm)
        results.append(r)
        print(f"  pass={r['pass']}  elapsed={r['elapsed_s']}s")
        if i < len(cases):
            await asyncio.sleep(30)

    passed = sum(1 for r in results if r["pass"])
    print(f"\n{passed}/{len(results)} cases passed")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_out = REPORT_DIR / "latest.json"
    json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Report written to {json_out}")

    md_out = REPORT_DIR / "latest.md"
    md_out.write_text(_render_markdown(results), encoding="utf-8")
    print(f"Report written to {md_out}")
    return 0 if passed == len(results) else 1


def _render_markdown(results: list[dict[str, Any]]) -> str:
    passed = sum(1 for r in results if r["pass"])
    total = len(results)
    total_elapsed = sum(r["elapsed_s"] for r in results)
    lines: list[str] = []
    lines.append("# CareAssist — Evaluation Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(UTC).isoformat()}")
    lines.append("")
    lines.append("## Scope and limitations")
    lines.append("")
    lines.append("This evaluation covers:")
    lines.append("")
    lines.append(
        "- **Retrieval hit**: expected PDF page appears in the retrieved chunks"
    )
    lines.append(
        "- **Citation grounding**: every page cited in the reply was actually retrieved"
    )
    lines.append(
        "- **No dosage**: no numeric dosages (`500 mg`, `2 tablets`, ...) in replies"
    )
    lines.append(
        "- **No brand name**: no Advil / Motrin / Tylenol / Bayer / Aleve / Excedrin"
    )
    lines.append(
        "- **Red-flag refusal**: emergency questions trigger escalation language"
    )
    lines.append(
        "- **Forbidden-phrase blocklist**: per-case specific phrases that must not appear"
    )
    lines.append(
        "- **Facts present** (LLM-as-judge): the reply contains "
        "the expected factual claims for the case"
    )
    lines.append("")
    lines.append(
        "**This evaluation does NOT verify medical correctness of answers**"
    )
    lines.append(
        "beyond citation grounding. A reply that cites a real page but misstates"
    )
    lines.append(
        "its content would pass every check here. Clinical review of answers is a"
    )
    lines.append(
        "**Stage 9 pre-pilot deliverable** — see `docs/eval/review.md` "
        "for the verification log and outstanding clinical sign-off."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **{passed} / {total} cases passed**")
    lines.append(f"- Total elapsed: {total_elapsed:.1f}s")
    lines.append("")
    lines.append("## Per-case results")
    lines.append("")
    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        lines.append(f"### [{status}] {r['case_id']}")
        lines.append("")
        lines.append(f"- Category: `{r['category']}`")
        lines.append(f"- Elapsed: {r['elapsed_s']}s")
        lines.append(f"- Retrieved pages: `{r['retrieved_pages']}`")
        if r["error"]:
            lines.append(f"- **Error:** `{r['error']}`")
        lines.append("")
        lines.append("**Scores:**")
        for name, detail in r["scores"].items():
            marker = "PASS" if detail["pass"] else "FAIL"
            extra = f" — {detail['detail']}" if detail["detail"] else ""
            lines.append(f"- `{marker}` `{name}`{extra}")
        lines.append("")
        lines.append("**Question:**")
        lines.append("")
        lines.append(f"> {r['question']}")
        lines.append("")
        lines.append("**Reply:**")
        lines.append("")
        lines.append("```")
        lines.append(r["reply"])
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    return asyncio.run(main_async(limit=args.limit))


if __name__ == "__main__":
    raise SystemExit(main())