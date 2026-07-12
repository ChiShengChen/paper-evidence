#!/usr/bin/env python3
"""Grow the evidence ledger to saturation, snowball it, and measure recall.

Search query variants until >=k consecutive batches add nothing new, chase citations
from the best seeds, then diff a survey's reference list against the ledger to report a
recall number and the papers still missing. Shares <root>/data/evidence/_work with
build_evidence.py.

Usage:
  python scripts/recall_audit.py --query "Drosophila connectome" --query "fly brain circuit" --sat-k 3
  python scripts/recall_audit.py --query "..." --question "How does the fly encode odor?" --expand
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from paper_evidence import recall  # noqa: E402
from paper_evidence.llm import api_key_available, get_client  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", action="append", dest="queries", required=True)
    ap.add_argument("--sources", default="arxiv,openalex,semanticscholar,pubmed,europepmc,crossref",
                    help="comma list of skill sources, or 'all' for every keyword source")
    ap.add_argument("--max-per-source", type=int, default=25)
    ap.add_argument("--sat-k", type=int, default=3,
                    help="stop after this many consecutive zero-yield batches")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--snowball-max", type=int, default=150)
    ap.add_argument("--question", default=None, help="research question for LLM query expansion")
    ap.add_argument("--expand", action="store_true", help="let the LLM propose new variants")
    ap.add_argument("--skill-dir", default=None)
    ap.add_argument("--root", default=".", help="output root (default: cwd)")
    ap.add_argument("--email", default=None)
    a = ap.parse_args()

    expand_llm = get_client() if (a.expand and api_key_available()) else None
    if a.expand and expand_llm is None:
        print("[recall] --expand requested but no LLM key; using provided queries only.")

    s = recall.run(root=Path(a.root), queries=a.queries, sources=a.sources,
                   max_per_source=a.max_per_source, sat_k=a.sat_k, seed_n=a.seeds,
                   snowball_max=a.snowball_max, skill_dir=a.skill_dir,
                   contact_email=a.email, research_question=a.question, expand_llm=expand_llm)

    print("\n=== recall summary ===")
    print(f"ledger: {s['ledger_size']} | saturated: {s['saturated']} | snowball +{s['snowball_new']}")
    if s["audit"]:
        au = s["audit"]
        print(f"recall vs survey: {au['recall']} ({au['n_found']}/{au['n_refs']}), "
              f"{len(au['misses'])} missing")


if __name__ == "__main__":
    main()
