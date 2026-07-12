#!/usr/bin/env python3
"""Deep-read ONE paper: PDF / URL / arXiv id -> verified evidence cards + a grounded summary.

Self-contained — no search, no ledger, no paper-deep-search skill. Anchors snippets
semantically (if --question and a Gemini key) or by keyword, extracts verbatim cards,
verifies them (verbatim + numbers-in-context + optional cross-family judge), and writes a
summary built only from the verified cards.

Usage:
  python scripts/deep_read_paper.py --arxiv 2605.10817 --question "what does it claim about AUROC?"
  python scripts/deep_read_paper.py --pdf paper.pdf --terms dopamine --terms "mushroom body"
  python scripts/deep_read_paper.py --url https://example.org/paper.pdf --out read.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from paper_evidence import deepread, evidence, quote_gate, semantic  # noqa: E402
from paper_evidence.llm import api_key_available, get_client  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdf", help="local .pdf (or already-extracted .txt)")
    src.add_argument("--url", help="URL to a PDF")
    src.add_argument("--arxiv", help="arXiv id, e.g. 2605.10817")
    ap.add_argument("--question", default=None, help="orients semantic anchoring + summary")
    ap.add_argument("--terms", action="append", default=[], help="keyword anchor (repeat)")
    ap.add_argument("--paper-id", default="P01", help="card paper id (default P01)")
    ap.add_argument("--max-cards", type=int, default=8)
    ap.add_argument("--no-semantic", action="store_true", help="force keyword anchoring")
    ap.add_argument("--root", default=".", help="output root (default: cwd)")
    ap.add_argument("--out", default=None, help="Markdown output (default: <root>/deep_read.md)")
    ap.add_argument("--email", default=None)
    a = ap.parse_args()

    root = Path(a.root)
    pno = a.paper_id
    ev = root / "data" / "evidence"
    (ev / "papers").mkdir(parents=True, exist_ok=True)

    # 1) get text — structured (section-tagged, headers/footers dropped) for cleaner grounding
    print("[deep-read] loading text…", file=sys.stderr)
    doc = deepread.load_structured(pdf=a.pdf, url=a.url, arxiv=a.arxiv, contact_email=a.email)
    text = doc.text()
    (ev / "papers" / f"{pno}.txt").write_text(text, encoding="utf-8")
    print(f"[deep-read] {len(text)} chars, {len(doc.sections())} section(s)", file=sys.stderr)

    if not api_key_available():
        sys.exit("[deep-read] an LLM key is required to extract cards (set a provider + key).")
    llm = get_client()

    # 2) anchor snippets: semantic (question + key) or keyword
    use_sem = bool(a.question) and not a.no_semantic
    snips = ""
    if use_sem:
        try:
            snips = "\n---\n".join(semantic.semantic_windows(text, a.question, k=20))[:14000]
            print("[deep-read] semantic anchoring", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f"[deep-read] semantic unavailable ({e}); keyword anchoring", file=sys.stderr)
            use_sem = False
    if not snips:
        terms = a.terms or evidence.query_terms([a.question] if a.question else [])
        wins = semantic.keyword_windows(text, terms) if terms else []
        snips = "\n---\n".join(wins)[:14000] or text[:12000]     # last resort: document head

    # 3) extract cards (verbatim self-repair) -> 4) verify
    cards = evidence.extract_cards_llm(llm, pno, snips, max_cards=a.max_cards, source_text=text)
    for c in cards:                                   # tag each card with its REAL section
        if (sec := doc.section_for(c.get("quote", ""))):
            c["section"] = sec
    evidence.write_jsonl(ev / "cards.jsonl", cards)
    gate = quote_gate.build(root, judge=quote_gate.make_judge())
    verified = [json.loads(l) for l in (ev / "cards_verified.jsonl").read_text().splitlines() if l.strip()]
    print(f"[deep-read] {len(verified)}/{len(cards)} cards verified", file=sys.stderr)

    # 5) synthesize from verified cards only
    summary = deepread.synthesize(llm, verified, question=a.question)

    out = Path(a.out) if a.out else root / "deep_read.md"
    lines = [f"# Deep read — {pno}", ""]
    if a.question:
        lines += [f"**Question:** {a.question}", ""]
    lines += ["## Summary (from verified cards only)", "", summary, "",
              f"## Verified evidence cards ({len(verified)})", ""]
    for c in verified:
        lines += [f"- **[{c['card_id']}]** {c.get('claim','')}",
                  f"  - quote: \"{c.get('quote','')}\""]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{summary}\n\nexported -> {out}")


if __name__ == "__main__":
    main()
