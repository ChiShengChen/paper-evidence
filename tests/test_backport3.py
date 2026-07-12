"""Tests for the NeuronLit-derived backports: claim-modality caps in evidence_state, and
non-destructive provenance."""
from paper_evidence import evidence_state, provenance


# --------------------------------------------------------------------------- #
# claim modality -> confidence cap
# --------------------------------------------------------------------------- #
def test_claim_modality_detection():
    assert evidence_state.claim_modality("We show that X drives Y") == "demonstrated"
    assert evidence_state.claim_modality("X may modulate Y") == "hedged"
    assert evidence_state.claim_modality("we propose a putative role") == "hedged"
    assert evidence_state.claim_modality("X connects to Y") == "asserted"


def test_hedged_claim_caps_confidence():
    strong = evidence_state.classify(verbatim_ok=True, faithful=True, subject_named=True)
    assert strong["confidence"] == "High"
    hedged = evidence_state.classify(verbatim_ok=True, faithful=True, subject_named=True,
                                     modality="hedged")
    assert hedged["state"] == "Supported" and hedged["confidence"] == "Low"
    assert hedged["modality"] == "hedged"


def test_from_card_result_detects_modality():
    judged = {"judged": True, "supported": True}
    hedged = {"status": "EXACT", "faithful": judged, "card": {"claim": "we may speculate X regulates Y"}}
    g = evidence_state.from_card_result(hedged)
    assert g["modality"] == "hedged" and g["confidence"] == "Low"   # hedged caps a supported claim
    demo = {"status": "EXACT", "faithful": judged, "card": {"claim": "we demonstrate X drives Y"}}
    assert evidence_state.from_card_result(demo)["confidence"] == "High"   # demonstrated, judged -> High


# --------------------------------------------------------------------------- #
# non-destructive provenance
# --------------------------------------------------------------------------- #
def test_merge_provenance_protects_enrichment():
    enriched = {"model_pin": "gemini-2.5-flash", "llm_invoked": True, "extractor": "cards",
                "tokens": 1234, "skill": "v1"}
    pins_only = {"skill": "v1", "datasets": {"x": "1"}}
    m = provenance.merge_provenance(enriched, pins_only)
    assert m["model_pin"] == "gemini-2.5-flash" and m["llm_invoked"] is True
    assert m["tokens"] == 1234 and m["datasets"] == {"x": "1"}     # incoming still overlaid


def test_llm_invoked_true_is_sticky():
    enriched = {"llm_invoked": True, "model_pin": "m"}
    assert provenance.merge_provenance(enriched, {"llm_invoked": False})["llm_invoked"] is True
    # but an empty existing may be SET by incoming
    assert provenance.merge_provenance({}, enriched)["model_pin"] == "m"


def test_assert_provenance_retained():
    assert provenance.assert_provenance_retained({"model_pin": "m", "llm_invoked": True})
    import pytest
    with pytest.raises(AssertionError):
        provenance.assert_provenance_retained({"skill": "v1"})   # lost model_pin + llm_invoked
