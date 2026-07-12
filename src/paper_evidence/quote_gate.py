"""Long-text reasoning gate: no unverifiable claim about a source survives.

The core anti-hallucination tool. Given evidence cards (a claim + a verbatim quote from
a source) and/or a draft that cites sources, it blocks anything an LLM might state about
a source that the source does not actually support — the failure mode of asking a model
to "summarize / compare / reason over" long documents: plausible sentences attributed to
papers that never said them, invented numbers, paraphrase drift.

What is a BLOCKER: a quoted span in the draft prose that traces to no source; and — via
an independent judge — a *cited sentence* whose source's verified cards don't support it,
even with no quote marks and no new number (this closes the paraphrase hole). An evidence
card that fails verification, or a citation to a source with no card yet, is a WARNING,
not a hard block. An uncited bad card never blocks.

This module is fully self-contained (stdlib only): it needs a source text and cards, not
a network. A quote passes only if it is found in the source as an EXACT or
whitespace/unicode/hyphenation-NORMALIZED substring; FUZZY matches pass only
when explicitly allowed. Two extra checks tighten "verbatim" into "supported":
  * numbers-in-context — a statistic the claim leans on must appear in a window *next
    to* the quoted sentence, not merely somewhere in the paper (NUM_FAIL otherwise);
  * claim faithfulness — an optional independent, cross-family LLM judge (make_judge)
    confirms the quote actually supports the claim; an unsupported card is UNFAITHFUL
    and dropped. This closes the hole where a verbatim quote is paired with an
    LLM-written paraphrase that overstates or misreads it.

Two inputs, both optional (gate is a no-op if neither exists, so the default pipeline
is unaffected):
  * evidence cards   data/evidence/cards.jsonl   -> verified card-by-card
  * landed sources   data/evidence/papers/*.txt  -> also used to check draft quotes

Outputs under data/evidence/: cards_verified.jsonl + quote_verification.md.
"""
from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

PASS_STATUSES = ("EXACT", "NORMALIZED")
# quoted spans in prose shorter than this are treated as scare-quotes/terms, not claims
MIN_QUOTE_CHARS = 40
_QUOTE_SPAN = re.compile(r'["“]([^"“”]{%d,})[”"]' % MIN_QUOTE_CHARS)


# --------------------------------------------------------------------------- #
# verification primitives (ported from paper-deep-search/verify_quotes.py)
# --------------------------------------------------------------------------- #
def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("­", "")                      # soft hyphen
    s = re.sub(r"-\s*\n\s*", "", s)                  # hyphenation across line breaks
    s = re.sub(r"[‘’ʼ]", "'", s)
    s = re.sub(r"[“”]", '"', s)
    s = re.sub(r"[‐-―−]", "-", s)
    s = re.sub(r"(?<=\w)-(?=\w)", "", s)             # intra-word hyphens
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def _fuzzy_best(hay_norm: str, q_norm: str) -> float:
    L = len(q_norm)
    if L == 0 or len(hay_norm) < 10:
        return 0.0
    step = max(20, L // 3)
    win = int(L * 1.4) + 20
    best = 0.0
    sm = SequenceMatcher(autojunk=False)
    sm.set_seq2(q_norm)
    for i in range(0, max(1, len(hay_norm) - L + 1), step):
        sm.set_seq1(hay_norm[i:i + win])
        if sm.real_quick_ratio() < best:
            continue
        r = sm.ratio()
        if r > best:
            best = r
            if best > 0.995:
                break
    return best


def _missing_numbers(numbers: list[Any] | None, hay_norm: str) -> list[str]:
    """Numbers that do not appear in the given (normalized) context window."""
    missing = []
    for num in numbers or []:
        n = str(num).strip()
        variants = {n, n.replace(",", ""), n.replace(",", " ")}
        if not any(v and v.lower() in hay_norm for v in variants):
            missing.append(n)
    return missing


def verify_quote(quote: str, source: str, numbers: list[Any] | None = None,
                 fuzzy: float = 0.95, allow_fuzzy: bool = False,
                 num_window: int | None = 200) -> dict[str, Any]:
    """Return {status, score, note, ok} for one quote against one source text.

    status: EXACT | NORMALIZED | FUZZY | FAIL | NUM_FAIL.
    ok is True only for a passing status (FUZZY passes only with allow_fuzzy).

    Numbers are checked in a ±num_window-char window around where the quote matched,
    not against the whole document — a statistic the claim leans on must appear *next
    to* the quoted sentence, so a value that merely occurs somewhere else in the paper
    no longer counts. Set num_window=None to check against the whole source.
    """
    quote = (quote or "").strip()
    if not quote:
        return {"status": "FAIL", "score": 0.0, "note": "empty quote", "ok": False}
    q_norm = normalize(quote)
    norm = normalize(source)
    if quote in source:
        status, score = "EXACT", 1.0
    elif q_norm and q_norm in norm:
        status, score = "NORMALIZED", 1.0
    else:
        score = _fuzzy_best(norm, q_norm)
        status = "FUZZY" if score >= fuzzy else "FAIL"

    note = ""
    if status != "FAIL":
        pos = norm.find(q_norm)
        if pos >= 0 and num_window is not None:
            lo, hi = max(0, pos - num_window), min(len(norm), pos + len(q_norm) + num_window)
            hay = norm[lo:hi]
        else:
            hay = norm
        miss = _missing_numbers(numbers, hay)
        if miss:
            status, note = "NUM_FAIL", "numbers not near quote: " + ", ".join(miss)

    ok = status in PASS_STATUSES or (status == "FUZZY" and allow_fuzzy)
    return {"status": status, "score": round(score, 4), "note": note, "ok": ok}


# --------------------------------------------------------------------------- #
# claim <-> quote faithfulness (independent, cross-family LLM judge)
# --------------------------------------------------------------------------- #
FAITHFUL_SYSTEM = (
    "You are a strict fact-checker. You are given a CLAIM about a paper and a verbatim "
    "QUOTE from that paper. Decide whether the quote SUPPORTS the claim: the claim is "
    "supported only if every assertion in it follows from the quote alone, adding no "
    "outside facts and no stronger wording than the quote warrants. Numbers must match. "
    'Reply JSON: {"supported": true|false, "reason": "<one sentence>"}.'
)
FAITHFUL_PROMPT = "CLAIM: {claim}\n\nQUOTE: {quote}\n\nDoes the quote support the claim?"


def judge_claim_support(judge: Any, claim: str, quote: str) -> dict[str, Any]:
    """Ask an independent LLM whether `quote` supports `claim`.

    Faithfulness is an *additive* check: if no judge is supplied or the call fails,
    the card is treated as unjudged (supported=True, judged=False) so infrastructure
    problems never silently drop evidence — they just fall back to verbatim-only.
    """
    claim = (claim or "").strip()
    quote = (quote or "").strip()
    if judge is None or not claim or not quote:
        return {"supported": True, "reason": "", "judged": False}
    try:
        r = judge.complete_json(FAITHFUL_SYSTEM,
                                FAITHFUL_PROMPT.format(claim=claim, quote=quote))
    except Exception as e:  # noqa: BLE001 — a flaky judge must not drop a good card
        return {"supported": True, "reason": f"judge unavailable: {e}", "judged": False}
    if not isinstance(r, dict):
        return {"supported": True, "reason": "judge returned non-dict", "judged": False}
    return {"supported": bool(r.get("supported", True)),
            "reason": str(r.get("reason", ""))[:300], "judged": True}


def make_judge(avoid: str = "gemini") -> Any:
    """A cross-family LLM client for faithfulness judging, or None if none available.

    Prefers a provider whose model family differs from `avoid` (the card extractor),
    so the judge is not the same model that wrote the claim. Returns None when only the
    extractor's family has a key — faithfulness judging is then skipped, not faked.
    """
    try:
        from .llm import api_key_available, get_client
    except Exception:  # noqa: BLE001
        return None
    for p in ("deepseek", "anthropic", "gemini"):
        if p != avoid and api_key_available(p):
            try:
                return get_client(provider=p)
            except Exception:  # noqa: BLE001
                continue
    return None


def citemap_from_ledger(root: Path) -> dict[str, str]:
    """Best-effort {bibtex_key: paper_no} from the evidence ledger, so a manuscript's
    \\cite{...} keys tie back to evidence cards. Mirrors related_work's key scheme
    ("arxiv" + alphanumerics of the arXiv id). Empty when no ledger is present."""
    led = Path(root) / "data" / "evidence" / "_work" / "ledger.jsonl"
    out: dict[str, str] = {}
    if not led.exists():
        return out
    try:
        for line in led.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            aid = str(r.get("arxiv_id", "")).strip()
            pno = r.get("paper_no", "")
            if aid and pno:
                out["arxiv" + re.sub(r"[^0-9a-zA-Z]", "", aid)] = pno
    except (json.JSONDecodeError, OSError):
        pass
    return out


# --------------------------------------------------------------------------- #
# gate
# --------------------------------------------------------------------------- #
def _load_sources(papers_dir: Path) -> dict[str, str]:
    if not papers_dir.exists():
        return {}
    return {p.stem: p.read_text(encoding="utf-8", errors="replace")
            for p in sorted(papers_dir.glob("*.txt"))}


def verify_cards(cards: list[dict], sources: dict[str, str],
                 allow_fuzzy: bool = False, judge: Any = None) -> list[dict]:
    """Verify each evidence card: verbatim quote first, then (optionally) faithfulness.

    A card enters the verified pool only if its quote is verbatim AND — when a `judge`
    is supplied — the judge confirms the claim is supported by that quote. A verbatim-OK
    but unsupported card is marked UNFAITHFUL (ok=False), so it drops out just like a
    quote that failed to match.
    """
    results = []
    for c in cards:
        pno = c.get("paper", "")
        cid = c.get("card_id", "?")
        src = sources.get(pno)
        if src is None:
            results.append({"card_id": cid, "paper": pno, "status": "FAIL",
                            "score": 0.0, "note": f"no source text for {pno}",
                            "ok": False, "card": c})
            continue
        v = verify_quote(c.get("quote", ""), src, c.get("numbers"),
                         allow_fuzzy=allow_fuzzy)
        r = {"card_id": cid, "paper": pno, **v, "card": c}
        if r["ok"] and judge is not None:
            fj = judge_claim_support(judge, c.get("claim", ""), c.get("quote", ""))
            r["faithful"] = fj
            if fj["judged"] and not fj["supported"]:
                r["status"], r["ok"] = "UNFAITHFUL", False
                r["note"] = "claim not supported by quote: " + fj["reason"]
        results.append(r)
    return results


def scan_draft_quotes(draft_text: str, sources: dict[str, str],
                      allow_fuzzy: bool = False) -> list[dict]:
    """Find long quoted spans in the manuscript and require each to match some source."""
    if not sources:
        return []
    results = []
    for m in _QUOTE_SPAN.finditer(draft_text or ""):
        span = m.group(1).strip()
        best = {"status": "FAIL", "score": 0.0, "ok": False, "note": "", "paper": ""}
        for pno, src in sources.items():
            v = verify_quote(span, src, allow_fuzzy=allow_fuzzy)
            if v["ok"] or v["score"] > best["score"]:
                best = {**v, "paper": pno}
                if v["ok"]:
                    break
        results.append({"quote": span[:120], **{k: best[k] for k in
                        ("status", "score", "note", "ok", "paper")}})
    return results


# --------------------------------------------------------------------------- #
# cited-claim check — unquoted sentences that assert something about a paper
# --------------------------------------------------------------------------- #
MIN_CLAIM_CHARS = 25
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CITE_PXX = re.compile(r"\[(P\d+(?:-c\d+)?(?:\s*,\s*P\d+(?:-c\d+)?)*)\]")
_CITE_TEX = re.compile(r"\\cite\{([^}]+)\}")

CLAIM_SUPPORT_SYSTEM = (
    "You are a strict fact-checker for a related-work section. Given a SENTENCE that "
    "asserts something about a specific paper, and that paper's EVIDENCE (verbatim "
    "quotes extracted from it), decide whether the evidence supports the sentence. It "
    "is supported only if every factual assertion in the sentence follows from the "
    "evidence, with no added facts and no stronger wording. General framing with no "
    'factual claim counts as supported. Reply JSON: {"supported": true|false, "reason": "<one sentence>"}.'
)
CLAIM_SUPPORT_PROMPT = "SENTENCE: {sentence}\n\nEVIDENCE:\n{evidence}\n\nDoes the evidence support the sentence?"


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text or "") if s.strip()]


def _cited_papers(sentence: str, citekey_to_paper: dict[str, str] | None) -> set[str]:
    papers: set[str] = set()
    for m in _CITE_PXX.finditer(sentence):
        papers.update(re.findall(r"P\d+", m.group(1)))
    for m in _CITE_TEX.finditer(sentence):
        for key in m.group(1).split(","):
            p = (citekey_to_paper or {}).get(key.strip())
            if p:
                papers.add(p)
    return papers


def _strip_citations(sentence: str) -> str:
    return _CITE_TEX.sub("", _CITE_PXX.sub("", sentence)).strip()


def judge_sentence_support(judge: Any, sentence: str, quotes: list[str]) -> dict[str, Any]:
    """Ask the judge whether a paper's verified quotes support a related-work sentence."""
    if judge is None or not sentence or not quotes:
        return {"supported": True, "reason": "", "judged": False}
    evidence = "\n".join(f"- {q}" for q in quotes if q)
    try:
        r = judge.complete_json(CLAIM_SUPPORT_SYSTEM,
                                CLAIM_SUPPORT_PROMPT.format(sentence=sentence, evidence=evidence))
    except Exception as e:  # noqa: BLE001
        return {"supported": True, "reason": f"judge unavailable: {e}", "judged": False}
    if not isinstance(r, dict):
        return {"supported": True, "reason": "judge returned non-dict", "judged": False}
    return {"supported": bool(r.get("supported", True)),
            "reason": str(r.get("reason", ""))[:300], "judged": True}


def scan_cited_claims(draft_text: str, verified_cards: list[dict],
                      citekey_to_paper: dict[str, str] | None = None,
                      judge: Any = None) -> list[dict]:
    """Check every sentence that cites a paper against that paper's verified cards.

    Closes the hole where an *unquoted* paraphrase about a paper (no quote marks, no new
    number) states something the evidence never supports. Requires a judge — paraphrase
    support can't be grepped; without one the check is skipped (returns []).

    Per cited paper in a factual sentence:
      * has verified quotes + judge says supported   -> SUPPORTED (ok)
      * has verified quotes + judge says unsupported -> UNSUPPORTED_CLAIM (blocker)
      * no verified card for the cited paper         -> NO_EVIDENCE (warning, non-blocking)
    """
    if judge is None:
        return []
    by_paper: dict[str, list[str]] = {}
    for c in verified_cards:
        by_paper.setdefault(c.get("paper", ""), []).append(c.get("quote", ""))
    results = []
    for sent in _sentences(draft_text):
        papers = _cited_papers(sent, citekey_to_paper)
        if not papers:
            continue
        claim = _strip_citations(sent)
        if len(claim) < MIN_CLAIM_CHARS:
            continue
        for p in sorted(papers):
            quotes = by_paper.get(p, [])
            if not quotes:
                results.append({"sentence": claim[:160], "paper": p, "status": "NO_EVIDENCE",
                                "ok": True, "note": f"no verified card for cited {p}"})
                continue
            fj = judge_sentence_support(judge, claim, quotes)
            supported = fj["supported"] or not fj["judged"]
            results.append({"sentence": claim[:160], "paper": p,
                            "status": "SUPPORTED" if supported else "UNSUPPORTED_CLAIM",
                            "ok": supported, "note": fj.get("reason", "")})
    return results


def build(root: Path, draft_text: str = "", allow_fuzzy: bool = False,
          judge: Any = None, citekey_to_paper: dict[str, str] | None = None) -> dict[str, Any]:
    """Run the gate. Returns a report dict; `passed` is False if any quote failed.

    No-op (skipped=True, passed=True) when neither cards nor source texts are present,
    so the default M6 pipeline is unaffected until a paper-deep-search corpus is wired in.
    Pass `judge` (see make_judge) to also drop cards whose claim the quote doesn't support.
    """
    ev = Path(root) / "data" / "evidence"
    cards_path = ev / "cards.jsonl"
    sources = _load_sources(ev / "papers")

    have_cards = cards_path.exists()
    if not have_cards and not sources:
        return {"skipped": True, "passed": True, "n_cards": 0, "n_failed": 0,
                "n_draft_quotes": 0, "card_results": [], "draft_results": []}

    cards = []
    if have_cards:
        for line in cards_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                cards.append(json.loads(line))

    card_results = verify_cards(cards, sources, allow_fuzzy=allow_fuzzy, judge=judge)
    passed_cards = [r["card"] for r in card_results if r["ok"]]
    # An unverifiable card is dropped (never enters cards_verified) and reported as a
    # WARNING — it is not yet cited, so it does not freeze the draft.
    card_warnings = [r for r in card_results if not r["ok"]]

    # Things that actually reached the manuscript prose and don't trace to evidence:
    #   * a quoted span with no verbatim source        -> BLOCKER
    #   * a cited sentence the paper's cards don't back -> BLOCKER
    #   * a cited sentence for a paper with no card yet -> WARNING (extract one / soften)
    draft_results = scan_draft_quotes(draft_text, sources, allow_fuzzy=allow_fuzzy)
    cited_results = scan_cited_claims(draft_text, passed_cards, citekey_to_paper, judge)
    quote_blockers = [r for r in draft_results if not r["ok"]]
    claim_blockers = [r for r in cited_results if r["status"] == "UNSUPPORTED_CLAIM"]
    cited_warnings = [r for r in cited_results if r["status"] == "NO_EVIDENCE"]
    blockers = quote_blockers + claim_blockers

    ev.mkdir(parents=True, exist_ok=True)
    with (ev / "cards_verified.jsonl").open("w", encoding="utf-8") as f:
        for c in passed_cards:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    _write_report(ev / "quote_verification.md", card_results, draft_results, cited_results)

    return {
        "skipped": False,
        "passed": len(blockers) == 0,
        "n_cards": len(card_results),
        "n_cards_passed": len(passed_cards),
        "n_cards_failed": len(card_warnings),
        "n_draft_quotes": len(draft_results),
        "n_draft_failed": len(quote_blockers),
        "n_cited_claims": len(cited_results),
        "n_cited_failed": len(claim_blockers),
        "n_failed": len(blockers),          # total blocking count (prose only)
        "card_warnings": card_warnings,     # dropped cards — reported, non-blocking
        "cited_warnings": cited_warnings,   # cited papers lacking a card — warning
        "blockers": blockers,               # unverifiable quotes/claims in prose — freeze
        "failed": blockers,                 # back-compat alias for `blockers`
        "card_results": card_results,
        "draft_results": draft_results,
        "cited_results": cited_results,
        "report": str(ev / "quote_verification.md"),
    }


def _write_report(path: Path, card_results: list[dict], draft_results: list[dict],
                  cited_results: list[dict] | None = None) -> None:
    n_ok = sum(r["ok"] for r in card_results)
    lines = [f"# Quote verification — {n_ok}/{len(card_results)} evidence cards passed", ""]
    if card_results:
        lines += ["## Evidence cards", "", "| card | paper | status | score | note |",
                  "|---|---|---|---|---|"]
        lines += [f"| {r['card_id']} | {r['paper']} | {r['status']} | "
                  f"{r['score']:.3f} | {r['note']} |" for r in card_results]
        lines.append("")
    if draft_results:
        lines += ["## Quoted spans in the draft", "",
                  "| quote (truncated) | matched paper | status | score |",
                  "|---|---|---|---|"]
        lines += [f"| {r['quote']} | {r['paper'] or '—'} | {r['status']} | "
                  f"{r['score']:.3f} |" for r in draft_results]
        lines.append("")
    if cited_results:
        lines += ["## Cited claims in the draft", "",
                  "| sentence (truncated) | paper | status | note |", "|---|---|---|---|"]
        lines += [f"| {r['sentence']} | {r['paper']} | {r['status']} | {r['note']} |"
                  for r in cited_results]
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
