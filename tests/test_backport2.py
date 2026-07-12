"""Tests for the pieces backported from flyhypo: skill-free Semantic Scholar snowball, and
the evidence_state grade attached to each quote_gate card result."""
import json

from paper_evidence import quote_gate, recall


# --------------------------------------------------------------------------- #
# skill-free S2 snowball
# --------------------------------------------------------------------------- #
def test_s2_seed_ref_routing():
    assert recall._s2_seed_ref("12345678") == "PMID:12345678"
    assert recall._s2_seed_ref("arxiv:2605.10817v2") == "ARXIV:2605.10817"
    assert recall._s2_seed_ref("2605.10817") == "ARXIV:2605.10817"
    assert recall._s2_seed_ref("10.1038/nature14539") == "DOI:10.1038/nature14539"
    assert recall._s2_seed_ref("doi:10.1/x") == "DOI:10.1/x"
    assert recall._s2_seed_ref("nonsense") is None


def test_snowball_s2_dedups(monkeypatch):
    def fake_edges(ref, direction, limit):
        a = {"source": "s2-snowball", "query": "q", "title": "A", "abstract": "",
             "year": "", "doi": "10.1/a", "arxiv_id": "", "pmid": "", "citations": "",
             "url": "", "pdf_url": ""}
        b = dict(a, title="B", doi="10.1/b")
        return [a] if direction == "refs" else [a, b]   # 'a' in both -> deduped
    monkeypatch.setattr(recall, "_s2_edges", fake_edges)
    out = recall.snowball_s2(["12345678"], direction="both")
    assert sorted(r["doi"] for r in out) == ["10.1/a", "10.1/b"]
    # composes with the existing pure recall_diff
    d = recall.recall_diff(out, [{"doi": "10.1/a"}])
    assert d["n_refs"] == 2 and d["n_found"] == 1


# --------------------------------------------------------------------------- #
# evidence_state grade attached to card results
# --------------------------------------------------------------------------- #
def test_card_results_carry_grade(tmp_path):
    ev = tmp_path / "data" / "evidence"
    (ev / "papers").mkdir(parents=True)
    (ev / "papers" / "P01.txt").write_text(
        "The model reached an accuracy of 0.873 on the held-out cohort.", encoding="utf-8")
    (ev / "cards.jsonl").write_text(json.dumps(
        {"card_id": "P01-c1", "paper": "P01", "claim": "acc 0.873",
         "quote": "reached an accuracy of 0.873", "numbers": ["0.873"]}) + "\n", encoding="utf-8")
    g = quote_gate.build(tmp_path, abstract_only=True)
    r = g["card_results"][0]
    assert r["ok"] and r["grade"]["state"] == "Supported"
    assert r["grade"]["confidence"] == "Med"        # abstract_only caps at Med
