#!/usr/bin/env python3
"""End-to-end: search literature to saturation -> build corpus -> generate & ground hypotheses.

    recall (coverage) -> build_evidence (land + card) -> verify -> generate -> ground -> export

Exports a Markdown + JSONL of hypotheses whose PREMISE is verified against the corpus
(the PREDICTION is the novel leap, kept as-is). Needs an LLM key (generation) and — for
the strongest grounding — a second provider key so the faithfulness judge is cross-family.
The search/fetch layer needs the paper-deep-search skill installed.

Usage:
  python scripts/hypothesis_loop.py --query "Drosophila mushroom body memory" \
      --query "fly olfactory learning circuit" \
      --question "How is olfactory memory encoded in the fly?" \
      --max-papers 6 --n 5 --out fly_hypotheses.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from paper_evidence import evidence, hypothesis, quote_gate, recall  # noqa: E402
from paper_evidence.llm import api_key_available, get_client  # noqa: E402


def _load_verified(root: Path) -> list[dict]:
    p = root / "data" / "evidence" / "cards_verified.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", action="append", dest="queries", required=True)
    ap.add_argument("--question", default=None, help="research question (for query expansion + export)")
    ap.add_argument("--sources", default="arxiv,openalex,semanticscholar,pubmed,europepmc,crossref",
                    help="skill sources, or 'all'")
    ap.add_argument("--max-per-source", type=int, default=20)
    ap.add_argument("--sat-k", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--snowball-max", type=int, default=120)
    ap.add_argument("--max-papers", type=int, default=6, help="papers to land full text for")
    ap.add_argument("--max-cards", type=int, default=6)
    ap.add_argument("--n", type=int, default=5, help="hypotheses to generate")
    ap.add_argument("--semantic", action="store_true",
                    help="anchor snippets by embedding similarity to --question (needs Gemini key)")
    ap.add_argument("--expand", action="store_true", help="LLM query expansion toward saturation")
    ap.add_argument("--no-recall", action="store_true", help="skip the coverage/snowball stage")
    ap.add_argument("--skill-dir", default=None)
    ap.add_argument("--root", default=".", help="output root (default: cwd)")
    ap.add_argument("--out", default=None, help="Markdown output (default: <root>/hypotheses.md)")
    ap.add_argument("--email", default=None)
    a = ap.parse_args()

    root = Path(a.root)
    if not api_key_available():
        sys.exit("[loop] an LLM key is required for generation (set PAPER_EVIDENCE_PROVIDER + key).")
    llm = get_client()
    judge = quote_gate.make_judge()
    print(f"[loop] generator={llm.provider} | judge={getattr(judge, 'provider', None) or 'none'}\n")

    # 1) coverage: grow the ledger to saturation + snowball (unless skipped)
    if not a.no_recall:
        recall.run(root=root, queries=a.queries, sources=a.sources,
                   max_per_source=a.max_per_source, sat_k=a.sat_k, seed_n=a.seeds,
                   snowball_max=a.snowball_max, skill_dir=a.skill_dir, contact_email=a.email,
                   research_question=a.question, expand_llm=(llm if a.expand else None))
        # 2) land + card from the recall-grown ledger (no re-search)
        ev = evidence.land_and_card(root, queries=a.queries, max_papers=a.max_papers,
                                    skill_dir=a.skill_dir, max_cards=a.max_cards, contact_email=a.email,
                                    question=a.question, semantic=a.semantic)
    else:
        ev = evidence.run(root=root, queries=a.queries, sources=a.sources,
                          max_per_source=a.max_per_source, max_papers=a.max_papers,
                          skill_dir=a.skill_dir, max_cards=a.max_cards, contact_email=a.email,
                          question=a.question, semantic=a.semantic)

    # 3) verify cards -> cards_verified.jsonl
    quote_gate.build(root, judge=judge)
    cards = _load_verified(root)
    print(f"\n[loop] {ev['landed']} paper(s) landed, {len(cards)} verified card(s)")
    if not cards:
        sys.exit("[loop] no verified evidence cards — nothing to ground hypotheses in.")

    # 4) generate + 5) ground
    hyps = hypothesis.generate(llm, cards, n=a.n)
    results = hypothesis.ground(root, hyps, judge=judge)
    grounded = [r for r in results if r["grounded"]]

    # 6) export
    out_md = Path(a.out) if a.out else root / "hypotheses.md"
    hypothesis.export_markdown(out_md, results, question=a.question)
    hypothesis.export_jsonl(out_md.with_suffix(".jsonl"), grounded)

    print(f"\n=== {len(grounded)}/{len(results)} hypotheses grounded ===")
    for i, r in enumerate(grounded, 1):
        print(f"  H{i} ({','.join(r['papers'])}): {r['premise'][:90]}")
    print(f"\nexported -> {out_md}  (+ {out_md.with_suffix('.jsonl').name})")


if __name__ == "__main__":
    main()
