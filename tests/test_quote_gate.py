"""Quote gate (long-text reasoning) tests.

Property under test: a verbatim quote may only enter the manuscript if it traces to a
landed source text (EXACT/NORMALIZED), mirroring the number-traceability guarantee.
"""

import json
from pathlib import Path

import pytest

from paper_evidence import quote_gate

SRC = (
    "Section 4.2 Results. The frequency-domain model reached an accuracy of 0.873 on "
    "the held-out EEG cohort, outperforming the time-domain baseline by 0.11. "
    "We attribute the gain to spectral disentanglement of motor rhythms."
)


def _make_evidence(root, cards, source=SRC, paper="P01"):
    ev = root / "data" / "evidence"
    (ev / "papers").mkdir(parents=True, exist_ok=True)
    (ev / "papers" / f"{paper}.txt").write_text(source, encoding="utf-8")
    with (ev / "cards.jsonl").open("w", encoding="utf-8") as f:
        for c in cards:
            f.write(json.dumps(c) + "\n")
    return ev


# --------------------------------------------------------------------------- #
# verify_quote primitive
# --------------------------------------------------------------------------- #
def test_exact_and_normalized_pass():
    v = quote_gate.verify_quote("accuracy of 0.873", SRC)
    assert v["status"] == "EXACT" and v["ok"]
    # case / whitespace / hyphenation differences still pass as NORMALIZED
    v = quote_gate.verify_quote("Frequency-Domain   MODEL reached", SRC)
    assert v["status"] == "NORMALIZED" and v["ok"]


def test_fabricated_quote_fails():
    v = quote_gate.verify_quote("achieved state-of-the-art on ImageNet", SRC)
    assert v["status"] == "FAIL" and not v["ok"]


def test_number_mismatch_is_num_fail():
    # quote text is present but the number it leans on is not in the source
    v = quote_gate.verify_quote("outperforming the time-domain baseline",
                                SRC, numbers=["0.99"])
    assert v["status"] == "NUM_FAIL" and not v["ok"]


def test_numbers_must_be_near_quote():
    filler = "lorem ipsum dolor sit amet consectetur " * 12  # ~470 chars > window
    src = ("The encoder reached 0.912 macro-F1 on the sleep cohort. " + filler +
           " A separate unrelated ablation table reports 0.500 elsewhere.")
    # number inside the quoted sentence -> near -> passes
    near = quote_gate.verify_quote("The encoder reached 0.912 macro-F1 on the sleep cohort",
                                   src, numbers=["0.912"])
    assert near["ok"] and near["status"] in ("EXACT", "NORMALIZED")
    # number that exists in the paper but far from the quote -> NUM_FAIL
    far = quote_gate.verify_quote("The encoder reached 0.912 macro-F1 on the sleep cohort",
                                  src, numbers=["0.500"])
    assert not far["ok"] and far["status"] == "NUM_FAIL"


class _Judge:
    def __init__(self, supported, boom=False):
        self.supported, self.boom, self.calls = supported, boom, 0

    def complete_json(self, system, prompt, **kw):
        self.calls += 1
        if self.boom:
            raise RuntimeError("judge down")
        return {"supported": self.supported, "reason": "test verdict"}


def test_faithfulness_judge_drops_unsupported_card(tmp_path):
    _make_evidence(tmp_path, [
        {"card_id": "P01-c1", "paper": "P01", "quote": "accuracy of 0.873",
         "claim": "the model cured cancer", "numbers": ["0.873"]},
    ])
    j = _Judge(supported=False)
    r = quote_gate.build(tmp_path, draft_text="", judge=j)
    assert j.calls == 1                              # judge ran on the verbatim-OK card
    assert r["n_cards_passed"] == 0 and r["n_cards_failed"] == 1
    assert r["card_warnings"][0]["status"] == "UNFAITHFUL"
    assert r["passed"]                               # uncited -> warning, not a blocker


def test_faithfulness_judge_keeps_supported_card(tmp_path):
    _make_evidence(tmp_path, [
        {"card_id": "P01-c1", "paper": "P01", "quote": "accuracy of 0.873",
         "claim": "the model reached 0.873 accuracy", "numbers": ["0.873"]},
    ])
    r = quote_gate.build(tmp_path, draft_text="", judge=_Judge(supported=True))
    assert r["n_cards_passed"] == 1


def test_faithfulness_judge_failure_is_soft(tmp_path):
    # judge infrastructure down -> card falls back to verbatim-only, not dropped
    _make_evidence(tmp_path, [
        {"card_id": "P01-c1", "paper": "P01", "quote": "accuracy of 0.873",
         "claim": "anything", "numbers": ["0.873"]},
    ])
    r = quote_gate.build(tmp_path, draft_text="", judge=_Judge(supported=False, boom=True))
    assert r["n_cards_passed"] == 1


# --------------------------------------------------------------------------- #
# cited-claim check — unquoted paraphrases about a cited paper
# --------------------------------------------------------------------------- #
class _ContentJudge:
    """Rejects support iff a trigger substring appears in the prompt (else supports)."""
    def __init__(self, reject):
        self.reject = reject.lower()

    def complete_json(self, system, prompt, **kw):
        return {"supported": self.reject not in prompt.lower(), "reason": "test verdict"}


def _good_card():
    return {"card_id": "P01-c1", "paper": "P01", "quote": "accuracy of 0.873",
            "claim": "the model reached 0.873 accuracy", "numbers": ["0.873"]}


def test_cited_claim_unsupported_is_blocker(tmp_path):
    _make_evidence(tmp_path, [_good_card()])
    draft = "The prior model reportedly cured cancer according to its authors [P01]."
    r = quote_gate.build(tmp_path, draft_text=draft, judge=_ContentJudge("cured cancer"))
    assert not r["passed"] and r["n_cited_failed"] == 1
    b = r["blockers"][0]
    assert b["status"] == "UNSUPPORTED_CLAIM" and b["paper"] == "P01"


def test_cited_claim_supported_passes(tmp_path):
    _make_evidence(tmp_path, [_good_card()])
    draft = "The prior model reached 0.873 accuracy on the shared benchmark [P01]."
    r = quote_gate.build(tmp_path, draft_text=draft, judge=_ContentJudge("zzz_never"))
    assert r["passed"] and r["n_cited_claims"] >= 1 and r["n_cited_failed"] == 0


def test_cited_claim_with_no_card_is_warning(tmp_path):
    _make_evidence(tmp_path, [_good_card()])
    draft = "A different system achieved strong results across many downstream tasks [P99]."
    r = quote_gate.build(tmp_path, draft_text=draft, judge=_ContentJudge("zzz"))
    assert r["passed"]  # NO_EVIDENCE is a warning, not a freeze
    assert r["cited_warnings"] and r["cited_warnings"][0]["paper"] == "P99"


def test_cited_claim_skipped_without_judge(tmp_path):
    _make_evidence(tmp_path, [_good_card()])
    draft = "The prior model reportedly cured cancer according to its authors [P01]."
    r = quote_gate.build(tmp_path, draft_text=draft, judge=None)
    assert r["passed"] and r["n_cited_claims"] == 0  # paraphrase check needs a judge


def test_scan_cited_claims_maps_tex_cite_keys():
    cards = [{"paper": "P01", "quote": "accuracy of 0.873"}]
    draft = r"The model reportedly cured cancer in every clinical trial \cite{arxiv2605}."
    res = quote_gate.scan_cited_claims(draft, cards, citekey_to_paper={"arxiv2605": "P01"},
                                       judge=_ContentJudge("cured cancer"))
    assert res and res[0]["paper"] == "P01" and res[0]["status"] == "UNSUPPORTED_CLAIM"


def test_citemap_from_ledger(tmp_path):
    work = tmp_path / "data" / "evidence" / "_work"
    work.mkdir(parents=True)
    (work / "ledger.jsonl").write_text(
        '{"paper_no": "P01", "arxiv_id": "2605.10817"}\n'
        '{"paper_no": "P02", "arxiv_id": "2603.04478"}\n', encoding="utf-8")
    m = quote_gate.citemap_from_ledger(tmp_path)
    assert m == {"arxiv260510817": "P01", "arxiv260304478": "P02"}


# --------------------------------------------------------------------------- #
# gate: cards + draft scan
# --------------------------------------------------------------------------- #
def test_gate_skipped_without_corpus(tmp_path):
    r = quote_gate.build(tmp_path, draft_text="anything")
    assert r["skipped"] and r["passed"]


def test_gate_passes_clean_cards(tmp_path):
    _make_evidence(tmp_path, [
        {"card_id": "P01-c1", "paper": "P01", "quote": "accuracy of 0.873",
         "numbers": ["0.873"]},
    ])
    r = quote_gate.build(tmp_path, draft_text="")
    assert r["passed"] and r["n_cards_passed"] == 1 and r["n_failed"] == 0
    verified = (tmp_path / "data" / "evidence" / "cards_verified.jsonl").read_text()
    assert "P01-c1" in verified


def test_bad_card_is_warning_not_blocker(tmp_path):
    # an unverifiable card that is NOT cited in the prose is dropped, not blocked
    _make_evidence(tmp_path, [
        {"card_id": "P01-c1", "paper": "P01", "quote": "cured all disease"},
    ])
    r = quote_gate.build(tmp_path, draft_text="")
    assert r["passed"] and r["n_cards_failed"] == 1 and r["n_draft_failed"] == 0
    assert r["card_warnings"][0]["card_id"] == "P01-c1"
    # dropped from the citable pool
    assert (tmp_path / "data" / "evidence" / "cards_verified.jsonl").read_text() == ""


def test_gate_blocks_fabricated_draft_quote(tmp_path):
    _make_evidence(tmp_path, [])  # sources present, no cards
    draft = 'Prior work notes "a completely invented claim not present anywhere in the source text".'
    r = quote_gate.build(tmp_path, draft_text=draft)
    assert not r["passed"] and r["n_draft_quotes"] == 1 and r["n_draft_failed"] == 1
    assert r["blockers"][0]["status"] == "FAIL"


def test_gate_accepts_faithful_draft_quote(tmp_path):
    _make_evidence(tmp_path, [])
    draft = 'They report the model "reached an accuracy of 0.873 on the held-out EEG cohort".'
    r = quote_gate.build(tmp_path, draft_text=draft)
    assert r["passed"] and r["n_draft_quotes"] == 1


