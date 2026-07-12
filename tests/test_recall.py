"""Recall-layer tests. The network stages in recall.run() are not exercised; the
saturation rule, seed selection, and recall diff (the parts that decide coverage) are."""
from paper_evidence import recall


# --------------------------------------------------------------------------- #
# normalization + matching
# --------------------------------------------------------------------------- #
def test_paper_keys_strong_and_title():
    ks = recall.paper_keys({"doi": "https://doi.org/10.1/AbC", "arxiv_id": "2605.10817v2",
                            "title": "A Great EEG Model"})
    assert "doi:10.1/abc" in ks and "arxiv:2605.10817" in ks and "ti:a great eeg model" in ks


def test_is_survey():
    assert recall.is_survey("A Survey of EEG Foundation Models")
    assert recall.is_survey("EEG Decoding: A Systematic Review")
    assert not recall.is_survey("CLEF: an EEG Foundation Model")


def test_seed_id_prefers_arxiv_then_doi():
    assert recall.seed_id({"arxiv_id": "2605.10817", "doi": "10.1/x"}) == "arxiv:2605.10817"
    assert recall.seed_id({"doi": "10.1/x"}) == "doi:10.1/x"
    assert recall.seed_id({"title": "no ids"}) == ""


# --------------------------------------------------------------------------- #
# saturation stopping rule
# --------------------------------------------------------------------------- #
def test_saturated_rule():
    assert not recall.saturated([5, 0, 0], k=3)      # only 2 trailing zeros
    assert recall.saturated([5, 0, 0, 0], k=3)       # 3 trailing zeros
    assert not recall.saturated([0, 0, 1], k=3)      # last batch found one
    assert not recall.saturated([0, 0], k=3)         # not enough batches
    assert recall.new_count(10, 13) == 3 and recall.new_count(13, 13) == 0


# --------------------------------------------------------------------------- #
# seed selection
# --------------------------------------------------------------------------- #
def test_pick_seeds_surveys_first_then_cited():
    ledger = [
        {"title": "CLEF model", "arxiv_id": "2605.10817", "citations": 5},
        {"title": "A Survey of EEG FMs", "arxiv_id": "2601.00001", "citations": 2},
        {"title": "Highly cited method", "doi": "10.1/z", "citations": 900},
        {"title": "no id paper", "citations": 999},          # unresolvable -> skipped
    ]
    seeds = recall.pick_seeds(ledger, n=3)
    assert seeds[0] == "arxiv:2601.00001"                    # survey first
    assert "doi:10.1/z" in seeds and "arxiv:2605.10817" in seeds
    assert all(s for s in seeds)                             # no empties


def test_pick_survey_most_cited():
    ledger = [
        {"title": "A Survey of X", "arxiv_id": "2601.00001", "citations": 10},
        {"title": "Review of Y", "arxiv_id": "2601.00002", "citations": 40},
        {"title": "Ordinary paper", "arxiv_id": "2601.00003", "citations": 999},
    ]
    assert recall.pick_survey(ledger)["arxiv_id"] == "2601.00002"
    assert recall.pick_survey([{"title": "no survey here", "arxiv_id": "1"}]) is None


# --------------------------------------------------------------------------- #
# recall diff
# --------------------------------------------------------------------------- #
def test_recall_diff_counts_and_misses():
    ledger = [{"arxiv_id": "2605.10817"}, {"doi": "10.1/have"}]
    refs = [
        {"arxiv_id": "2605.10817", "title": "have A"},        # in ledger (arxiv)
        {"doi": "10.1/HAVE", "title": "have B"},              # in ledger (doi, case-insens)
        {"arxiv_id": "2605.10817", "title": "dup"},           # duplicate of first -> ignored
        {"doi": "10.1/missing", "title": "Missing Paper", "citations": 12},  # miss
        {"title": "Titled Only Miss"},                        # miss via title
    ]
    d = recall.recall_diff(refs, ledger)
    assert d["n_refs"] == 4 and d["n_found"] == 2 and d["recall"] == 0.5
    assert {m["title"] for m in d["misses"]} == {"Missing Paper", "Titled Only Miss"}
    assert d["misses"][0]["title"] == "Missing Paper"         # sorted by citations desc


def test_recall_diff_empty_refs():
    d = recall.recall_diff([], [{"arxiv_id": "2605.10817"}])
    assert d["n_refs"] == 0 and d["recall"] is None


# --------------------------------------------------------------------------- #
# LLM query expansion (fake client)
# --------------------------------------------------------------------------- #
class _FakeLLM:
    def __init__(self, queries):
        self.queries = queries

    def complete_json(self, system, prompt, **kw):
        return {"queries": self.queries}


def test_expand_queries_dedupes_and_caps():
    llm = _FakeLLM(["EEG representation learning", "eeg foundation model", "new variant 3",
                    "another variant 4", "variant 5"])
    out = recall.expand_queries(llm, "q?", tried=["EEG foundation model"], n=3)
    assert "eeg foundation model" not in [q.lower() for q in out]   # dropped as already tried
    assert len(out) == 3                                            # capped at n


def test_expand_queries_soft_fail():
    class Boom:
        def complete_json(self, *a, **k): raise RuntimeError("down")
    assert recall.expand_queries(Boom(), "q?", tried=[]) == []
    assert recall.expand_queries(None, "q?", tried=[]) == []
