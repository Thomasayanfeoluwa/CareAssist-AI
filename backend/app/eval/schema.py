from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalCase:
    id: str
    category: str
    question: str
    expected_pages: tuple[int, ...] = field(default_factory=tuple)
    expected_title: str = "Medical book"
    must_refuse: bool = False
    must_not_contain: tuple[str, ...] = field(default_factory=tuple)
    # NEW: 3-5 short factual claims a correct answer must mention.
    # Checked by an LLM-as-judge, not by substring match, because a correct
    # answer may express these facts in many different ways.
    expected_facts: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "EvalCase":
        return cls(
            id=d["id"],
            category=d["category"],
            question=d["question"],
            expected_pages=tuple(d.get("expected_pages", ())),
            expected_title=d.get("expected_title", "Medical book"),
            must_refuse=bool(d.get("must_refuse", False)),
            must_not_contain=tuple(d.get("must_not_contain", ())),
            expected_facts=tuple(d.get("expected_facts", ())),
            notes=d.get("notes", ""),
        )