#!/usr/bin/env python3
"""Drosophila neural-function hypothesis generator — the generate -> verify skeleton.

The pattern paper-evidence exists for: let an LLM propose hypotheses grounded in a
literature corpus, then keep ONLY the ones whose GROUNDING the sources actually support,
and BLOCK the ones that misread the literature — before any of it reaches your write-up.

    generate (LLM)  ->  verify the premise (quote_gate)  ->  grounded hypotheses only

Key idea: a hypothesis is a PREMISE (a statement attributed to the literature, cited
[Pxx]) plus a PREDICTION (a novel, testable leap). Only the *premise* is checkable —
the prediction is novel by design and is expected to be unsupported yet. So we verify the
premise and let the prediction ride along, flagged as novel. Verifying the whole sentence
would wrongly reject every real hypothesis (the leap is never "in the paper").

This file is self-contained: it writes a tiny toy evidence corpus (two fly papers + their
verified cards), then runs the verify step. Swap the toy corpus for one built by
`scripts/build_evidence.py`, and swap the canned hypotheses for your own generator.

Run:
    python examples/fly_hypothesis_gen.py                 # offline: canned hypotheses, no key
    python examples/fly_hypothesis_gen.py --llm           # generate hypotheses with an LLM
                                                          #   (set PAPER_EVIDENCE_PROVIDER + key)

Verification tiers (automatic, based on what keys you have):
  * quoted claims  -> checked verbatim against the sources        (no key needed)
  * numbers        -> must sit next to their quote                (no key needed)
  * cited sentences (no quote marks) -> a cross-family LLM judge must confirm the source
    supports them                                                 (needs a 2nd provider key)
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from paper_evidence import quote_gate  # noqa: E402
from paper_evidence.llm import api_key_available, get_client  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. A tiny toy evidence corpus (normally produced by scripts/build_evidence.py)
# --------------------------------------------------------------------------- #
SOURCES = {
    "P01": (
        "The mushroom body is the principal olfactory learning center in Drosophila. "
        "Dopaminergic neurons of the PPL1 cluster convey aversive teaching signals to "
        "mushroom body compartments. "
        "Blocking dopaminergic input during training abolishes aversive memory formation."
    ),
    "P02": (
        "Kenyon cells are the intrinsic neurons of the mushroom body. "
        "Sparse odor coding by Kenyon cells supports discrimination of similar odors. "
        "The anterior paired lateral (APL) neuron provides feedback inhibition that "
        "maintains coding sparseness."
    ),
}

# Evidence cards: each quote is copied verbatim from its source (the extraction step
# would normally do this; quote_gate re-greps them to be sure).
CARDS = [
    {"card_id": "P01-c1", "paper": "P01",
     "claim": "PPL1 dopaminergic neurons carry aversive teaching signals to the mushroom body.",
     "quote": "Dopaminergic neurons of the PPL1 cluster convey aversive teaching signals to "
              "mushroom body compartments.", "numbers": []},
    {"card_id": "P01-c2", "paper": "P01",
     "claim": "Dopaminergic input is required during training for aversive memory formation.",
     "quote": "Blocking dopaminergic input during training abolishes aversive memory formation.",
     "numbers": []},
    {"card_id": "P02-c1", "paper": "P02",
     "claim": "Sparse coding by Kenyon cells supports discrimination of similar odors.",
     "quote": "Sparse odor coding by Kenyon cells supports discrimination of similar odors.",
     "numbers": []},
    {"card_id": "P02-c2", "paper": "P02",
     "claim": "APL feedback inhibition maintains Kenyon-cell coding sparseness.",
     "quote": "The anterior paired lateral (APL) neuron provides feedback inhibition that "
              "maintains coding sparseness.", "numbers": []},
]


def write_corpus(root: Path) -> None:
    ev = root / "data" / "evidence"
    (ev / "papers").mkdir(parents=True, exist_ok=True)
    for pno, text in SOURCES.items():
        (ev / "papers" / f"{pno}.txt").write_text(text, encoding="utf-8")
    with (ev / "cards.jsonl").open("w", encoding="utf-8") as f:
        for c in CARDS:
            f.write(json.dumps(c) + "\n")


# --------------------------------------------------------------------------- #
# 2. Generate hypotheses (LLM if asked, else a canned mix of good + fabricated)
# --------------------------------------------------------------------------- #
GEN_SYSTEM = (
    "You are a Drosophila neuroscientist proposing novel, testable hypotheses. You are "
    "given verified findings, each with a source id like [P01]. For each hypothesis give a "
    "PREMISE — a statement grounded in the findings, citing the source(s) as [Pxx], stating "
    "only what the findings support — and a PREDICTION — the novel, testable leap beyond "
    "them. Do not invent facts about the sources in the premise."
)


def generate_llm(llm) -> list[dict]:
    findings = "\n".join(f"[{c['paper']}] {c['claim']}" for c in CARDS)
    prompt = (f"Verified findings:\n{findings}\n\n"
              'Return JSON {"hypotheses": [{"premise": "... [Pxx]", "prediction": "..."}, ...]} '
              "with 3 hypotheses.")
    r = llm.complete_json(GEN_SYSTEM, prompt)
    hs = r.get("hypotheses", []) if isinstance(r, dict) else []
    return [h for h in hs if isinstance(h, dict) and h.get("premise")]


def generate_canned() -> list[dict]:
    return [
        # faithful premise (verbatim from P01) + a novel prediction -> KEEP
        {"premise": "Blocking dopaminergic input during training abolishes aversive memory "
                    "formation [P01].",
         "prediction": "so boosting PPL1 dopamine during training will strengthen aversive memory."},
        # faithful premise (paraphrase backed by P02-c2) + novel prediction -> KEEP
        {"premise": "APL feedback inhibition maintains Kenyon-cell coding sparseness [P02].",
         "prediction": "so silencing APL will degrade discrimination of similar odors."},
        # FABRICATED premise with a made-up quote — blocked even offline (verbatim scan)
        {"premise": 'Kenyon cells "encode visual place memory in the central complex" [P02].',
         "prediction": "so they also store navigational spatial maps."},
        # OVERSTATED premise — P01 never claims 'sole center for every modality' -> BLOCK (judge)
        {"premise": "The mushroom body is the sole memory center for every sensory modality "
                    "in the fly [P01].",
         "prediction": "so all learning must route through it."},
    ]


# --------------------------------------------------------------------------- #
# 3. Verify each hypothesis and report
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", action="store_true", help="generate hypotheses with an LLM")
    a = ap.parse_args()

    root = Path(tempfile.mkdtemp(prefix="fly_demo_"))
    write_corpus(root)

    if a.llm and api_key_available():
        print(f"[gen] generating hypotheses with {get_client().provider}…\n")
        hypotheses = generate_llm(get_client())
    else:
        if a.llm:
            print("[gen] no LLM key; using canned hypotheses.\n")
        hypotheses = generate_canned()

    judge = quote_gate.make_judge()          # cross-family; None if only one provider has a key
    print(f"[verify] judge: {getattr(judge, 'provider', None) or 'none (verbatim-only)'}\n")

    kept, blocked = [], []
    for i, h in enumerate(hypotheses, 1):
        # verify ONLY the premise (the grounding); the prediction is novel by design
        gate = quote_gate.build(root, draft_text=h["premise"], judge=judge)
        ok = gate["passed"]
        (kept if ok else blocked).append(h)
        print(f"{'✅ KEEP ' if ok else '🚫 BLOCK'} H{i}")
        print(f"   premise    : {h['premise']}")
        print(f"   prediction : {h['prediction']}  (novel — not checked)")
        for b in gate["blockers"]:
            why = b.get("note") or "quote not found in any source"
            print(f"   ↳ {b['status']}: {why}")
        print()

    print(f"=== {len(kept)} grounded, {len(blocked)} blocked ===")
    print("Keep the grounded hypotheses; their premise is faithful to the literature and "
          "their prediction is a fair, testable leap. The blocked ones misread the sources.")
    print(f"(toy corpus written under {root})")


if __name__ == "__main__":
    main()
