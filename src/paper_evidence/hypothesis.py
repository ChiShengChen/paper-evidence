"""Batch hypothesis generation grounded in a verified evidence corpus.

A hypothesis = a PREMISE (a statement attributed to the literature, citing [Pxx]) + a
PREDICTION (the novel, testable leap). Only the premise is checkable, so `ground()`
verifies the premise with `quote_gate` and keeps the hypotheses whose grounding is
faithful — the prediction rides along as novel-by-design. This is the reusable core of
the recall -> corpus -> generate -> ground loop (see scripts/hypothesis_loop.py).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from . import quote_gate

_PXX = re.compile(r"\[(P\d+)(?:-c\d+)?\]")

GEN_SYSTEM = (
    "You are a scientist proposing novel, testable hypotheses from verified findings. "
    "Each finding is cited like [P01]. For each hypothesis give a PREMISE — a statement "
    "grounded in the findings, citing the source(s) as [Pxx], asserting ONLY what the "
    "findings support — and a PREDICTION — the novel, testable leap beyond them. Never "
    "invent facts about the sources in the premise; the leap belongs in the prediction."
)
GEN_PROMPT = ('Verified findings:\n{findings}\n\n'
              'Return JSON {{"hypotheses": [{{"premise": "... [Pxx]", "prediction": "..."}}, ...]}} '
              'with {n} hypotheses, each premise citing the finding(s) it rests on.')


def generate(llm: Any, cards: list[dict], n: int = 5) -> list[dict]:
    """Ask an LLM for n premise/prediction hypotheses grounded in the verified cards."""
    if not cards:
        return []
    findings = "\n".join(f"[{c.get('paper','')}] {c.get('claim','')}" for c in cards)
    try:
        r = llm.complete_json(GEN_SYSTEM, GEN_PROMPT.format(findings=findings, n=n))
    except Exception as e:  # noqa: BLE001
        print(f"[hypothesis] generation failed: {e}")
        return []
    hs = r.get("hypotheses", []) if isinstance(r, dict) else []
    out = []
    for h in hs:
        if isinstance(h, dict) and str(h.get("premise", "")).strip():
            out.append({"premise": str(h["premise"]).strip(),
                        "prediction": str(h.get("prediction", "")).strip()})
    return out


def ground(root: Path, hypotheses: list[dict], judge: Any = None,
           **build_kwargs: Any) -> list[dict]:
    """Verify each hypothesis's PREMISE against the corpus; annotate grounded/blocked."""
    results = []
    for h in hypotheses:
        g = quote_gate.build(root, draft_text=h["premise"], judge=judge, **build_kwargs)
        results.append({
            "premise": h["premise"],
            "prediction": h.get("prediction", ""),
            "papers": sorted(set(_PXX.findall(h["premise"]))),
            "grounded": g["passed"],
            "blockers": [{"status": b["status"], "reason": b.get("note", ""),
                          "text": b.get("sentence") or b.get("quote", "")}
                         for b in g["blockers"]],
        })
    return results


def export_markdown(path: Path, results: list[dict], question: str | None = None) -> Path:
    grounded = [r for r in results if r["grounded"]]
    blocked = [r for r in results if not r["grounded"]]
    L = ["# Grounded hypotheses", ""]
    if question:
        L += [f"**Question:** {question}", ""]
    L += [f"{len(grounded)} grounded / {len(results)} generated — premises verified against "
          "the evidence corpus; predictions are novel by design.", ""]
    L += ["## Grounded", ""]
    for i, r in enumerate(grounded, 1):
        cites = ", ".join(r["papers"]) or "—"
        L += [f"### H{i}  ({cites})",
              f"- **Premise** (grounded): {r['premise']}",
              f"- **Prediction** (novel): {r['prediction']}", ""]
    if blocked:
        L += ["## Blocked — premise not supported by the corpus (excluded)", ""]
        for r in blocked:
            why = "; ".join(b["reason"] or b["status"] for b in r["blockers"]) or "unsupported"
            L += [f"- ~~{r['premise']}~~ — {why}"]
        L.append("")
    Path(path).write_text("\n".join(L), encoding="utf-8")
    return Path(path)


def export_jsonl(path: Path, results: list[dict]) -> Path:
    with Path(path).open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return Path(path)
