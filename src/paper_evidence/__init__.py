"""paper-evidence — extract verifiable evidence from literature and ground LLM claims in it.

Two layers:
  * quote_gate — the long-text reasoning / anti-hallucination core (stdlib only). Verifies
    that every claim about a source is backed by a verbatim quote, that numbers sit next to
    their quote, and — with a cross-family judge — that paraphrased/cited claims are
    actually supported. Works on any (claim, quote, source); no network.
  * evidence + recall — drive the `paper-deep-search` skill to search literature to
    saturation, snowball citations, land full texts, and extract verbatim evidence cards.

See README for the fruit-fly-hypothesis-generator style usage: generate hypotheses with an
LLM, then run quote_gate to keep only the ones your source corpus actually supports.
"""
from . import evidence, hypothesis, quote_gate, recall

__all__ = ["quote_gate", "evidence", "recall", "hypothesis"]
__version__ = "0.1.0"
