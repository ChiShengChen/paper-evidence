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


def _cap(tier: str, ceiling: str) -> str:
    return ceiling if _CONF.index(tier) > _CONF.index(ceiling) else tier


def classify(verbatim_ok: bool, faithful: bool | None = None,
             subject_named: bool | None = None, abstract_only: bool = False,
             asserts: bool = True) -> dict[str, Any]:
    """Grade one claim.

    verbatim_ok    : the supporting quote is verbatim in the source.
    faithful       : an independent judge says the quote supports the claim (None = unjudged).
    subject_named  : the claim's subject is named in the quote (None = not checked).
    abstract_only  : the only evidence is an abstract (caps confidence).
    asserts        : the claim makes a positive assertion (vs a hedge/question).
    """
    if not verbatim_ok:
        # silence: no verbatim evidence -> Unknown, never a conflict/overclaim
        return {"state": "Unknown", "confidence": "Low", "overclaim_risk": False,
                "reasons": ["no verbatim quote in source (silence → Unknown)"]}

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
    if state in ("Conflict", "Tension"):
        conf = _cap(conf, "Low")
    return {"state": state, "confidence": conf, "overclaim_risk": overclaim, "reasons": reasons}


def from_card_result(r: dict, abstract_only: bool = False, asserts: bool = True) -> dict[str, Any]:
    """Map a quote_gate card verification result to an evidence state."""
    status = r.get("status", "")
    verbatim_ok = status in ("EXACT", "NORMALIZED", "MISATTRIBUTED", "UNFAITHFUL")
    faithful = (False if status == "UNFAITHFUL"
                else (r.get("faithful") or {}).get("supported") if r.get("faithful") else None)
    subject_named = False if status == "MISATTRIBUTED" else None
    return classify(verbatim_ok=verbatim_ok, faithful=faithful, subject_named=subject_named,
                    abstract_only=abstract_only, asserts=asserts)
