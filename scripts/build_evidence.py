#!/usr/bin/env python3
"""Search literature, land full texts, extract verbatim evidence cards, verify them.

Drives the paper-deep-search skill (search -> dedupe -> fetch -> snippets) and extracts
evidence cards with an LLM, then verifies every quote against its source. Output lands
under <root>/data/evidence/ where quote_gate.build() reads it.

Usage:
  python scripts/build_evidence.py --query "Drosophila mushroom body memory" --max-papers 6
  python scripts/build_evidence.py --query "..." --sources all --no-llm
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from paper_evidence import evidence  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", action="append", dest="queries", required=True,
                    help="search query (native syntax); repeat for several")
    ap.add_argument("--sources", default="arxiv,openalex,semanticscholar,pubmed,europepmc,crossref",
                    help="comma list of skill sources, or 'all' for every keyword source")
    ap.add_argument("--max-per-source", type=int, default=25)
    ap.add_argument("--max-papers", type=int, default=8)
    ap.add_argument("--terms", action="append", default=[])
    ap.add_argument("--max-cards", type=int, default=6)
    ap.add_argument("--skill-dir", default=None,
                    help=f"paper-deep-search scripts dir (default: {evidence.DEFAULT_SKILL_DIR})")
    ap.add_argument("--no-llm", action="store_true", help="land texts only, skip card extraction")
    ap.add_argument("--root", default=".", help="output root (default: cwd)")
    ap.add_argument("--email", default=None, help="CONTACT_EMAIL for polite API access")
    a = ap.parse_args()

    s = evidence.run(root=Path(a.root), queries=a.queries, sources=a.sources,
                     max_per_source=a.max_per_source, max_papers=a.max_papers, terms=a.terms,
                     skill_dir=a.skill_dir, use_llm=not a.no_llm, max_cards=a.max_cards,
                     contact_email=a.email)
    print(f"\nEvidence corpus: {s['landed']} paper(s), {s['n_cards']} card(s) in {s['evidence_dir']}")


if __name__ == "__main__":
    main()
