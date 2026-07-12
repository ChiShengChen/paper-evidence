"""Evidence-corpus builder tests. The network/subprocess stages in evidence.run()
are not exercised here; the pure helpers and the LLM card step (with a fake client)
are, plus an end-to-end tie-in with the quote gate."""
import json

from paper_evidence import evidence, quote_gate

SRC = (
    "4.1 Results. Our frequency-domain encoder reached 0.912 macro-F1 on the sleep "
    "cohort, beating the time-domain baseline by 0.14. Ablating the wavelet stem "
    "dropped performance to 0.77."
)


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def complete_json(self, system, prompt, **kw):
        self.calls += 1
        return self.payload


def _write_ledger(tmp_path, rows):
    p = tmp_path / "ledger.jsonl"
    evidence.write_jsonl(p, rows)
    return p


# --------------------------------------------------------------------------- #
# mark_include: land the best OA rows
# --------------------------------------------------------------------------- #
def test_mark_include_prefers_oa_and_citations(tmp_path):
    led = _write_ledger(tmp_path, [
        {"paper_no": "P01", "arxiv_id": "2607.00001", "citations": 3},
        {"paper_no": "P02", "arxiv_id": "", "pdf_url": "", "pmcid": ""},  # no OA -> excluded
        {"paper_no": "P03", "arxiv_id": "2607.00003", "citations": 40},
    ])
    chosen = evidence.mark_include(led, max_papers=2)
    assert chosen == ["P01", "P03"]  # P02 has no landable source
    rows = {r["paper_no"]: r for r in evidence.read_jsonl(led)}
    assert rows["P03"]["status"] == "include" and rows["P01"]["status"] == "include"
    assert rows["P02"].get("status") in (None, "")


def test_mark_include_respects_max(tmp_path):
    led = _write_ledger(tmp_path, [
        {"paper_no": f"P{i:02d}", "arxiv_id": f"2607.000{i:02d}", "citations": i}
        for i in range(1, 6)])
    assert len(evidence.mark_include(led, max_papers=2)) == 2


# --------------------------------------------------------------------------- #
# parse_cards / extract_cards_llm
# --------------------------------------------------------------------------- #
def test_parse_cards_assigns_stable_ids_and_drops_empty():
    raw = {"cards": [
        {"claim": "a", "quote": "reached 0.912 macro-F1", "numbers": ["0.912"]},
        {"claim": "b", "quote": "   "},                       # empty quote -> dropped
        {"claim": "c", "quote": "beating the time-domain baseline", "numbers": "0.14"},
    ]}
    cards = evidence.parse_cards(raw, "P01")
    assert [c["card_id"] for c in cards] == ["P01-c1", "P01-c2"]
    assert all(c["paper"] == "P01" for c in cards)
    assert cards[1]["numbers"] == ["0.14"]  # scalar coerced to list


def test_extract_cards_llm_failure_is_soft(tmp_path):
    class Boom:
        def complete_json(self, *a, **k):
            raise RuntimeError("api down")
    assert evidence.extract_cards_llm(Boom(), "P01", "some snippet") == []


def test_extract_cards_llm_happy():
    llm = FakeLLM({"cards": [{"claim": "x", "quote": "reached 0.912 macro-F1",
                              "numbers": ["0.912"]}]})
    cards = evidence.extract_cards_llm(llm, "P07", "…reached 0.912 macro-F1…")
    assert llm.calls == 1 and cards[0]["card_id"] == "P07-c1"


class _CardRepairLLM:
    """First call returns cards; a REPAIR call (system starts 'You copy EXACT') returns fixes."""
    def __init__(self, cards_payload, repair_payload):
        self.cards_payload, self.repair_payload = cards_payload, repair_payload

    def complete_json(self, system, prompt, **kw):
        return self.repair_payload if system.startswith("You copy EXACT") else self.cards_payload


def test_extract_cards_self_repair(tmp_path):
    src = "Dopamine gates memory in the mushroom body. The APL neuron maintains sparseness."
    cards_payload = {"cards": [
        {"claim": "dopamine gates memory",
         "quote": "Dopamine gates memory in the mushroom body.", "numbers": []},   # verbatim -> kept
        {"claim": "apl role",
         "quote": "The APL neuron abolishes all memory forever.", "numbers": []},  # stitched -> repair
    ]}
    repair_payload = {"quotes": {"P01-c2": "The APL neuron maintains sparseness."}}  # verbatim fix
    out = evidence.extract_cards_llm(_CardRepairLLM(cards_payload, repair_payload), "P01",
                                     "…snippets…", source_text=src)
    by_id = {c["card_id"]: c for c in out}
    assert "P01-c1" in by_id                                        # verbatim quote kept
    assert by_id["P01-c2"]["quote"] == "The APL neuron maintains sparseness."   # repaired


def test_extract_cards_self_repair_drops_unrecoverable(tmp_path):
    src = "Dopamine gates memory in the mushroom body."
    cards_payload = {"cards": [
        {"claim": "made up", "quote": "Kenyon cells store visual maps.", "numbers": []}]}
    repair_payload = {"quotes": {"P01-c1": "still not in the source at all"}}
    out = evidence.extract_cards_llm(_CardRepairLLM(cards_payload, repair_payload), "P01",
                                     "…snippets…", source_text=src)
    assert out == []                                               # non-verbatim, unrepairable -> dropped


# --------------------------------------------------------------------------- #
# snippets + terms file
# --------------------------------------------------------------------------- #
def test_gather_snippets_falls_back_to_fulltext(tmp_path):
    (tmp_path / "papers").mkdir()
    (tmp_path / "papers" / "P01.txt").write_text(SRC, encoding="utf-8")
    blob = evidence.gather_snippets(tmp_path, "P01")
    assert "0.912 macro-F1" in blob


def test_make_terms_file_splits_query_into_keywords(tmp_path):
    p = evidence.make_terms_file(tmp_path / "terms.txt", [], ["EEG foundation model for sleep"])
    lines = p.read_text().split()
    assert "EEG" in lines and "foundation" in lines and "sleep" in lines
    assert "for" not in lines                          # stopword dropped
    # explicit --terms still passes through verbatim
    p2 = evidence.make_terms_file(tmp_path / "t2.txt", ["custom term"], ["ignored"])
    assert p2.read_text().strip() == "custom term"


def test_query_terms_quoted_phrase_and_syntax():
    # quoted phrase kept as one anchor; field prefix + boolean dropped
    terms = evidence.query_terms(['all:"mushroom body" AND dopamine'])
    assert "mushroom body" in terms and "dopamine" in terms
    assert "all" not in terms and "AND" not in terms and "and" not in terms
    # dedup across queries, case-insensitive
    assert evidence.query_terms(["Dopamine dopamine DOPAMINE"]) == ["Dopamine"]


def test_expand_sources_all_shortcut():
    s = evidence.expand_sources("all")
    assert s == evidence.ALL_SEARCH_SOURCES
    for src in ("arxiv", "pubmed", "dblp", "openreview", "core", "crossref"):
        assert src in s.split(",")
    assert "unpaywall" not in s.split(",")               # DOI-lookup, not a keyword source
    # a concrete list passes through unchanged (case-insensitive 'all' only)
    assert evidence.expand_sources("arxiv,pubmed") == "arxiv,pubmed"
    assert evidence.expand_sources("ALL") == evidence.ALL_SEARCH_SOURCES


# --------------------------------------------------------------------------- #
# install_corpus  +  end-to-end tie-in with the quote gate
# --------------------------------------------------------------------------- #
def test_install_then_gate_verifies(tmp_path):
    # simulate a skill workdir with one landed paper
    work = tmp_path / "work"
    (work / "papers").mkdir(parents=True)
    (work / "papers" / "P01.txt").write_text(SRC, encoding="utf-8")

    # an LLM produced one faithful card and one fabricated card
    cards = evidence.parse_cards({"cards": [
        {"claim": "F1 result", "quote": "reached 0.912 macro-F1", "numbers": ["0.912"]},
        {"claim": "made up", "quote": "achieved 0.999 on every benchmark ever"},
    ]}, "P01")

    summary = evidence.install_corpus(tmp_path, work, cards, ["P01"])
    assert summary["papers_copied"] == ["P01"] and summary["n_cards"] == 2
    assert (tmp_path / "data" / "evidence" / "papers" / "P01.txt").exists()

    # with no draft, the fabricated card is dropped (warning), the real one is kept —
    # an uncited bad card does not block
    gate = quote_gate.build(tmp_path, draft_text="")
    assert gate["passed"] and gate["n_cards"] == 2 and gate["n_cards_failed"] == 1
    verified = evidence.read_jsonl(tmp_path / "data" / "evidence" / "cards_verified.jsonl")
    assert [c["card_id"] for c in verified] == ["P01-c1"]

    # but the moment that fabricated quote reaches the prose, it is a blocker
    blocked = quote_gate.build(
        tmp_path,
        draft_text='We note it "achieved 0.999 accuracy on every benchmark ever created".')
    assert not blocked["passed"] and blocked["n_draft_failed"] == 1
