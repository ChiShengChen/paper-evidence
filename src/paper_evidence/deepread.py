"""Single-paper deep read: PDF/URL/arXiv -> verified evidence cards + a grounded summary.

The skill's "single-paper deep-read mode" packaged and self-contained: no search, no
ledger, no skill scripts needed — just get the text, anchor snippets (semantic if a
question + key, else keyword), extract cards, verify them with quote_gate, and synthesize
a summary from ONLY the verified cards. See scripts/deep_read_paper.py for the CLI.
"""
from __future__ import annotations

import io
import re
import urllib.request
from pathlib import Path
from typing import Any

_ARXIV = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")


def _pdf_to_text(data: bytes) -> str:
    import fitz  # PyMuPDF
    doc = fitz.open(stream=data, filetype="pdf")
    return "".join(page.get_text("text") for page in doc)


def _fetch(pdf: str | None, url: str | None, arxiv: str | None,
           contact_email: str | None) -> tuple[bytes | None, str | None]:
    """Return (pdf_bytes, plain_text) for one paper — exactly one is non-None."""
    if pdf:
        p = Path(pdf)
        if p.suffix.lower() == ".pdf":
            return p.read_bytes(), None
        return None, p.read_text(encoding="utf-8", errors="replace")   # already-extracted .txt
    if arxiv:
        m = _ARXIV.search(arxiv)
        if not m:
            raise ValueError(f"not an arXiv id: {arxiv}")
        url = f"https://arxiv.org/pdf/{m.group(1)}"
    if not url:
        raise ValueError("provide one of pdf / url / arxiv")
    req = urllib.request.Request(url, headers={
        "User-Agent": f"paper-evidence/1.0 (mailto:{contact_email or 'anonymous@example.com'})"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    return (data, None) if data[:5].startswith(b"%PDF") else (None, data.decode("utf-8", "replace"))


def load_text(pdf: str | None = None, url: str | None = None, arxiv: str | None = None,
              contact_email: str | None = None) -> str:
    """Return the full text of one paper from a local PDF, a PDF URL, or an arXiv id."""
    data, text = _fetch(pdf, url, arxiv, contact_email)
    return _pdf_to_text(data) if data is not None else (text or "")


def load_structured(pdf: str | None = None, url: str | None = None, arxiv: str | None = None,
                    contact_email: str | None = None):
    """Return a section-tagged StructuredDoc (Marker-style): cleaner source + real section
    names for cards. Falls back to a single text block for an already-extracted .txt."""
    from .structure import Block, StructuredDoc, extract_structured
    data, text = _fetch(pdf, url, arxiv, contact_email)
    if data is not None:
        return extract_structured(data)
    return StructuredDoc([Block("text", "", 1, text or "")])


SYNTH_SYSTEM = (
    "You write a faithful, concise summary of ONE paper using ONLY the provided verified "
    "findings (each is a claim backed by a verbatim quote). Add no facts beyond them. End "
    "each sentence with the finding id(s) it rests on, e.g. [P01-c2]. If a question is "
    "given, orient the summary toward it, but never state anything the findings don't support."
)


def synthesize(llm: Any, cards: list[dict], question: str | None = None) -> str:
    """A grounded summary of a single paper from its verified cards (or a stub if none)."""
    if not cards:
        return "_No verified evidence cards — nothing to summarize._"
    findings = "\n".join(f"[{c['card_id']}] {c.get('claim','')} "
                         f"(quote: \"{c.get('quote','')}\")" for c in cards)
    q = f"Question to orient toward: {question}\n\n" if question else ""
    try:
        return llm.complete(SYNTH_SYSTEM, f"{q}Verified findings:\n{findings}\n\n"
                            "Write a 4-6 sentence summary, each sentence citing its finding id(s).")
    except Exception as e:  # noqa: BLE001
        return f"_Synthesis failed ({e}); see the verified cards below._"
