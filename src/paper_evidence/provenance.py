"""Non-destructive provenance — once truthfully set, a later pass can't drop or downgrade it.

Ported from NeuronLit's provenance guard (a round-10 audit found a later assembly pass had
clobbered a cell's `model_pin`/`llm_invoked` down to just skill/datasets). In any multi-pass
pipeline — extract → verify → grade → assemble — the record of *how* a claim was produced
(which model, whether an LLM was invoked, token cost) must survive later passes that only add
structural fields. This module makes the merge non-destructive and provides a gate that fails
assembly if a produced record loses its provenance.

Pure stdlib. `merge_provenance` / `assert_provenance_retained` are unit-tested.
"""
from __future__ import annotations

from typing import Any, Iterable

# Keys that a later overlay must never drop or downgrade once truthfully set.
PROTECTED = ("model_pin", "llm_invoked", "extractor", "tokens", "provider")


def merge_provenance(existing: dict, incoming: dict) -> dict[str, Any]:
    """Overlay `incoming` onto `existing` without ever dropping/downgrading the PROTECTED keys.

    - llm_invoked: True is sticky — a later `llm_invoked=False` cannot overwrite a real True.
    - the other PROTECTED keys: an existing truthy value survives unless `incoming` supplies its
      own non-empty value.
    - every other key takes the incoming value (a normal overlay).
    """
    existing = dict(existing or {})
    incoming = dict(incoming or {})
    out = {**existing, **incoming}
    if existing.get("llm_invoked") is True:
        out["llm_invoked"] = True
    for k in PROTECTED:
        if existing.get(k) and not incoming.get(k):
            out[k] = existing[k]
    return out


def assert_provenance_retained(provenance: dict,
                               required: Iterable[str] = ("model_pin", "llm_invoked")) -> bool:
    """Raise AssertionError if `provenance` is missing any `required` key (falsy counts as missing).

    Call it after an assembly/merge pass on any record that was LLM-enriched, so a pass that
    accidentally overwrote provenance with structural-only fields fails loudly instead of
    silently shipping an unattributable claim."""
    prov = provenance or {}
    missing = [k for k in required if not prov.get(k)]
    assert not missing, f"provenance lost required key(s): {missing}"
    return True
