"""Tests for the features backported from NeuronLit: citation verify, mis-attribution
guard (focus_named), and graded evidence_state. Pure/offline parts only (citation network
resolvers are exercised via monkeypatched fetchers)."""
from paper_evidence import citation, evidence_state, quote_gate


# --------------------------------------------------------------------------- #
# mis-attribution guard (quote_gate.name_matches / focus_named)
# --------------------------------------------------------------------------- #
def test_name_matches_boundary_aware():
    assert quote_gate.name_matches("CLEF", "the CLEF model reached 0.74")
    assert not quote_gate.name_matches("P1", "as shown in P12 and P13")     # P1 not inside P12
    assert quote_gate.name_matches("MBON 1", "silencing MBON-1 impaired memory")
    assert not quote_gate.name_matches("MBON 1", "MBON 12 targets the lateral horn")
    assert quote_gate.focus_named(["CBraMod", "MTDP"], "our MTDP framework distils into CBraMod")
    assert not quote_gate.focus_named(["DINOv3"], "kenyon cells provide a sparse code")


def test_gate_flags_misattributed_card(tmp_path):
    ev = tmp_path / "data" / "evidence"
    (ev / "papers").mkdir(parents=True)
    (ev / "papers" / "P01.txt").write_text(
        "The baseline model reached 0.55 accuracy on the task.", encoding="utf-8")
    import json
    (ev / "cards.jsonl").write_text(json.dumps(
        {"card_id": "P01-c1", "paper": "P01", "subject": "CLEF",
         "claim": "CLEF reached 0.55", "quote": "The baseline model reached 0.55 accuracy",
         "numbers": ["0.55"]}) + "\n", encoding="utf-8")
    g = quote_gate.build(tmp_path)          # verbatim OK, but subject 'CLEF' not in the quote
    r = g["card_results"][0]
    assert r["status"] == "MISATTRIBUTED" and not r["ok"]
    assert g["n_cards_passed"] == 0


# --------------------------------------------------------------------------- #
# evidence_state grading (the three rules)
# --------------------------------------------------------------------------- #
def test_state_supported_and_confidence():
    s = evidence_state.classify(verbatim_ok=True, faithful=True, subject_named=True)
    assert s["state"] == "Supported" and s["confidence"] == "High"


def test_state_silence_is_unknown_never_conflict():
    s = evidence_state.classify(verbatim_ok=False)
    assert s["state"] == "Unknown" and s["overclaim_risk"] is False


def test_state_overclaim_only_positive():
    # quote present but doesn't support the claim -> Conflict + overclaim
    s = evidence_state.classify(verbatim_ok=True, faithful=False)
    assert s["state"] == "Conflict" and s["overclaim_risk"] and s["confidence"] == "Low"
    # subject not named -> Tension, still an overclaim (positive mis-attribution)
    t = evidence_state.classify(verbatim_ok=True, faithful=True, subject_named=False)
    assert t["state"] == "Tension" and t["overclaim_risk"]


def test_state_abstract_only_caps_confidence():
    s = evidence_state.classify(verbatim_ok=True, faithful=True, subject_named=True,
                                abstract_only=True)
    assert s["state"] == "Supported" and s["confidence"] == "Med"


def test_from_card_result_maps_signals():
    assert evidence_state.from_card_result({"status": "EXACT"})["state"] == "Supported"
    assert evidence_state.from_card_result({"status": "MISATTRIBUTED"})["state"] == "Tension"
    assert evidence_state.from_card_result({"status": "UNFAITHFUL"})["state"] == "Conflict"
    assert evidence_state.from_card_result({"status": "FAIL"})["state"] == "Unknown"


# --------------------------------------------------------------------------- #
# citation verify (monkeypatched resolvers — no network)
# --------------------------------------------------------------------------- #
def test_verify_citation_verified_and_title_match(monkeypatch):
    monkeypatch.setattr(citation, "fetch_crossref",
                        lambda doi: {"source": "crossref", "doi": doi,
                                     "title": "CLEF: an EEG Foundation Model", "retracted": False})
    monkeypatch.setattr(citation, "fetch_openalex", lambda doi: None)
    r = citation.verify_citation(doi="10.1/x", title="CLEF EEG foundation model")
    assert r["status"] == "verified" and r["ok"] and r["title_match"] is True


def test_verify_citation_unverified(monkeypatch):
    monkeypatch.setattr(citation, "fetch_crossref", lambda doi: None)
    monkeypatch.setattr(citation, "fetch_openalex", lambda doi: None)
    r = citation.verify_citation(doi="10.9/invented")
    assert r["status"] == "unverified" and not r["ok"]


def test_verify_citation_retracted(monkeypatch):
    monkeypatch.setattr(citation, "fetch_openalex",
                        lambda doi: {"source": "openalex", "doi": doi, "title": "X",
                                     "retracted": True})
    monkeypatch.setattr(citation, "fetch_crossref", lambda doi: None)
    r = citation.verify_citation(doi="10.1/bad")
    assert r["status"] == "RETRACTED" and r["retracted"] and not r["ok"]


def test_verify_batch_summary(monkeypatch):
    monkeypatch.setattr(citation, "fetch_crossref",
                        lambda doi: {"source": "crossref", "doi": doi, "title": "T",
                                     "retracted": False} if doi == "10.1/ok" else None)
    monkeypatch.setattr(citation, "fetch_openalex", lambda doi: None)
    out = citation.verify_batch([{"doi": "10.1/ok"}, {"doi": "10.2/missing"}])
    assert out["n"] == 2 and out["n_verified"] == 1 and out["n_unverified"] == 1
