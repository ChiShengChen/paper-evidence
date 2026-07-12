"""Grade a claim's evidence state — beyond the binary verified/failed of quote_gate.

Turns verification signals (is the quote verbatim? does it support the claim? is the claim's
subject named in it? is it only from an abstract?) into a state + a confidence tier, so a
hypothesis generator can say *how* supported a claim is, not just yes/no.

Ported from NeuronLit's evidence_state, domain-neutralized, keeping its three sharp rules:
  * overclaim = POSITIVE mis-attribution only — a quote that says the wrong thing. Never inferred
    from silence (a source simply not mentioning something is Unknown, not a conflict/overclaim).
  * asymmetry — a hit can upgrade; silence caps at Unknown, never Conflict.
  * evidence depth — abstract-only evidence (or any overclaim risk) caps confidence at Med.

States: Supported | Tension | Conflict | Unknown.  Confidence: High | Med | Low | Unknown.
"""
from __future__ import annotations

from typing import Any

_CONF = ["Unknown", "Low", "Med", "High"]

# Assertion-strength cues in a claim's own wording (from NeuronLit's crossing_gates):
# a source that HEDGES ("may modulate") is less certain than one that DEMONSTRATES
# ("we show X drives Y"), so a hedged claim's confidence is capped regardless of verbatim
# support — the evidence itself is tentative.
_HEDGE = ("may ", "might", "could", "likely", "possibly", "perhaps", "hypothes", "given that",
          "this may", "this might", "suggest", "we propose", "proposed", "putative", "presumably",
          "would ", "appears to", "appear to", "seems to", "potential", "is thought",
          "we speculate", "raise the possibility", "consistent with the idea")
_DEMONSTRATED = ("we found", "we show", "we showed", "demonstrate", "we observed", "revealed that",
                 "confirmed that", "established that", "results show", "we identified", "showed that",
                 "we quantified", "we demonstrate", "here we show", "data show")


def claim_modality(sentence: str) -> str:
    """Return 'demonstrated' | 'asserted' | 'hedged' from the claim's own wording."""
    s = " " + str(sentence or "").lower() + " "
    if any(h in s for h in _HEDGE):
        return "hedged"
    if any(d in s for d in _DEMONSTRATED):
        return "demonstrated"
    return "asserted"


def _cap(tier: str, ceiling: str) -> str:
    return ceiling if _CONF.index(tier) > _CONF.index(ceiling) else tier


def classify(verbatim_ok: bool, faithful: bool | None = None,
             subject_named: bool | None = None, abstract_only: bool = False,
             asserts: bool = True, modality: str | None = None) -> dict[str, Any]:
    """Grade one claim.

    verbatim_ok    : the supporting quote is verbatim in the source.
    faithful       : an independent judge says the quote supports the claim (None = unjudged).
    subject_named  : the claim's subject is named in the quote (None = not checked).
    abstract_only  : the only evidence is an abstract (caps confidence).
    asserts        : the claim makes a positive assertion (vs a hedge/question).
    modality       : 'demonstrated'|'asserted'|'hedged' — a hedged source caps confidence.
    """
    if not verbatim_ok:
        # silence: no verbatim evidence -> Unknown, never a conflict/overclaim
        return {"state": "Unknown", "confidence": "Low", "overclaim_risk": False,
                "modality": modality, "reasons": ["no verbatim quote in source (silence → Unknown)"]}

    reasons: list[str] = []
    overclaim = (faithful is False) or (subject_named is False)   # positive mis-attribution only
    if faithful is False:
        state = "Conflict" if asserts else "Unknown"
        reasons.append("quote does not support the claim")
    elif subject_named is False:
        state = "Tension" if asserts else "Unknown"
        reasons.append("claim subject not named in the quote")
    else:
        state = "Supported"

    conf = "High" if (faithful is True and subject_named is not False) else "Med"
    if faithful is None:
        conf = _cap(conf, "Med"); reasons.append("faithfulness unjudged")
    if abstract_only:
        conf = _cap(conf, "Med"); reasons.append("abstract-only evidence")
    if overclaim:
        conf = _cap(conf, "Med"); reasons.append("overclaim risk")
    # a hedged source is tentative — cap confidence even when verbatim-supported
    if modality == "hedged":
        conf = _cap(conf, "Low"); reasons.append("hedged claim (source expresses uncertainty)")
    if state in ("Conflict", "Tension"):
        conf = _cap(conf, "Low")
    return {"state": state, "confidence": conf, "overclaim_risk": overclaim,
            "modality": modality, "reasons": reasons}


def from_card_result(r: dict, abstract_only: bool = False, asserts: bool = True) -> dict[str, Any]:
    """Map a quote_gate card verification result to an evidence state."""
    status = r.get("status", "")
    verbatim_ok = status in ("EXACT", "NORMALIZED", "MISATTRIBUTED", "UNFAITHFUL")
    faithful = (False if status == "UNFAITHFUL"
                else (r.get("faithful") or {}).get("supported") if r.get("faithful") else None)
    subject_named = False if status == "MISATTRIBUTED" else None
    card = r.get("card") or {}
    modality = claim_modality(card.get("claim") or card.get("quote") or "")
    return classify(verbatim_ok=verbatim_ok, faithful=faithful, subject_named=subject_named,
                    abstract_only=abstract_only, asserts=asserts, modality=modality)
