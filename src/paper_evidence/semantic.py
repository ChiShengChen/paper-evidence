"""Semantic anchoring for long documents: chunk a paper, rank chunks by a question's
embedding, and return the most relevant passages — so "reworded" questions still hit,
unlike literal keyword/regex matching.

The pure pieces (chunk_text, cosine, rank_chunks with an injected embed_fn, keyword_windows)
are stdlib + unit-tested. embed_texts() is the only network part (Gemini embeddings via
google-genai); it is optional — callers fall back to keyword windows when no key/SDK.
"""
from __future__ import annotations

import math
import re
from typing import Any, Callable

_SENT = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, size: int = 1200, overlap: int = 200) -> list[str]:
    """Whitespace-normalized, ~size-char chunks with overlap, broken on sentence ends
    where possible (so a chunk rarely splits mid-sentence)."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    sents = _SENT.split(text)
    chunks, cur = [], ""
    for s in sents:
        if cur and len(cur) + 1 + len(s) > size:
            chunks.append(cur)
            cur = (cur[-overlap:] + " " + s) if overlap else s   # carry overlap for context
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur)
    # a single monster sentence longer than size: hard-split it
    out: list[str] = []
    for c in chunks:
        if len(c) <= size * 2:
            out.append(c)
        else:
            for i in range(0, len(c), size - overlap):
                out.append(c[i:i + size])
    return out


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def rank_chunks(question: str, chunks: list[str], embed_fn: Callable[[list[str]], list[list[float]]],
                k: int = 12) -> list[tuple[int, float, str]]:
    """Return the top-k (index, score, chunk) most similar to `question`. embed_fn embeds a
    list of texts -> list of vectors; the question is embedded in the same batch."""
    if not chunks:
        return []
    vecs = embed_fn([question] + chunks)
    qv, cvs = vecs[0], vecs[1:]
    scored = sorted(((i, cosine(qv, cv), chunks[i]) for i, cv in enumerate(cvs)),
                    key=lambda t: t[1], reverse=True)
    return scored[:k]


def keyword_windows(text: str, terms: list[str], window: int = 450,
                    max_per_term: int = 30) -> list[str]:
    """Literal fallback: context windows around each term occurrence (like the skill's
    extract_snippets, but self-contained so a single-paper read needs no skill)."""
    out, seen = [], set()
    for term in terms:
        try:
            rx = re.compile(term[3:], re.I) if term.startswith("re:") else re.compile(re.escape(term), re.I)
        except re.error:
            continue
        for m in list(rx.finditer(text))[:max_per_term]:
            s, e = max(0, m.start() - window), min(len(text), m.end() + window)
            w = text[s:e].strip()
            key = w[:80]
            if key not in seen:
                seen.add(key)
                out.append(w)
    return out


# --------------------------------------------------------------------------- #
# embeddings (Gemini; the only network part)
# --------------------------------------------------------------------------- #
def embed_texts(texts: list[str], model: str | None = None) -> list[list[float]]:
    """Embed texts with Gemini (model from $PAPER_EVIDENCE_EMBED_MODEL, default
    gemini-embedding-001). Raises if google-genai or a key is missing (callers that want a
    soft fallback should catch and use keyword_windows instead)."""
    import os

    from google import genai

    from .llm import _resolve_key
    key = _resolve_key("gemini")
    if not key:
        raise RuntimeError("no Gemini key for embeddings (GEMINI_API_KEY / GOOGLE_API_KEY)")
    model = model or os.environ.get("PAPER_EVIDENCE_EMBED_MODEL", "gemini-embedding-001")
    client = genai.Client(api_key=key)
    out: list[list[float]] = []
    for i in range(0, len(texts), 100):            # Gemini caps a batch at 100 requests
        resp = client.models.embed_content(model=model, contents=texts[i:i + 100])
        out.extend(list(e.values) for e in resp.embeddings)
    return out


def semantic_windows(text: str, question: str, k: int = 12, size: int = 1200,
                     overlap: int = 200, embed_fn: Callable | None = None) -> list[str]:
    """Top-k passages of `text` most relevant to `question` (embedding similarity)."""
    chunks = chunk_text(text, size=size, overlap=overlap)
    if not chunks:
        return []
    ranked = rank_chunks(question, chunks, embed_fn or embed_texts, k=k)
    return [c for _, _, c in ranked]
