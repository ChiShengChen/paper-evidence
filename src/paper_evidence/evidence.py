"""Build the evidence corpus the quote gate consumes, by driving the
`paper-deep-search` skill and closing its one un-scripted step with an LLM.

Skill pipeline (each stage is one of the skill's stdlib scripts, shelled out):
    search_papers.py  -> results/qNN.jsonl
    dedupe.py         -> ledger.jsonl/csv
    (triage)          -> mark_include(): pick papers to land            [here]
    fetch_fulltext.py -> papers/Pxx.txt          (legal OA only)
    extract_snippets.py -> snippets/Pxx.jsonl    (term-anchored windows)
    (card extraction) -> extract_cards_llm(): snippets -> evidence cards [here, LLM]
    install_corpus()  -> data/evidence/{papers/Pxx.txt, cards.jsonl}    [here]

The card step is the skill's only stage with no script (normally a reading model).
We run it with this package's LLM client so the whole thing is one command; the cards'
quotes are then re-greped against the source by `quote_gate` — extraction and
verification stay separate, so a hallucinated quote is caught, not trusted.

Network/LLM stages live in `run()`; the pure helpers below are unit-tested.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .quote_gate import verify_quote

DEFAULT_SKILL_DIR = Path.home() / ".claude" / "skills" / "paper-deep-search" / "scripts"

# Every keyword-searchable source. `unpaywall` is intentionally absent: it is a DOI
# lookup, not a keyword search (it is already used as an OA-PDF fallback in
# fetch_fulltext). `--sources all` expands to this via expand_sources().
ALL_SEARCH_SOURCES = ("arxiv,pubmed,europepmc,semanticscholar,openalex,crossref,"
                      "dblp,openreview,core")


def expand_sources(sources: str) -> str:
    """Map the 'all' shortcut to the full keyword-searchable source list; else passthrough."""
    return ALL_SEARCH_SOURCES if (sources or "").strip().lower() == "all" else sources

CARD_SYSTEM = (
    "You extract verbatim evidence cards from academic-paper snippets. Every 'quote' "
    "MUST be copied character-for-character from the provided snippet text — never "
    "paraphrase inside a quote, never invent a number. One finding per card. If a "
    "snippet supports no clear factual claim, emit no card for it."
)
CARD_PROMPT = """From the snippets below (all from paper {pno}), extract up to {max_cards} \
evidence cards as JSON: {{"cards": [{{"claim": "...", "quote": "...", "section": "...", \
"numbers": ["..."]}}]}}.

- claim: a one-sentence paraphrase in your own words.
- quote: <= 300 chars, copied EXACTLY from a snippet window below.
- section: best guess at the section/heading, or "" if unknown.
- numbers: every statistic the claim depends on, copied exactly (e.g. ["0.87", "12,450"]).

SNIPPETS:
{snippets}
"""


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
def read_jsonl(path: Path) -> list[dict]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _landable_rank(p: dict) -> tuple:
    """Sort key: prefer sources that reliably yield OA full text, THEN citations.

    arXiv (3) always lands; a PMC id (2) usually lands via Europe PMC; a bare pdf_url (1)
    is often a paywalled publisher link that 403s. Ranking OA-reliability above citations
    matters because in many fields the most-cited papers are the paywalled ones — sorting
    by citations alone fills the triage list with rows that never land."""
    oa_tier = 3 if p.get("arxiv_id") else 2 if p.get("pmcid") else 1 if p.get("pdf_url") else 0
    try:
        cites = int(p.get("citations") or 0)
    except (TypeError, ValueError):
        cites = 0
    return (oa_tier, cites)


def mark_include(ledger_path: Path, max_papers: int, require_oa: bool = True) -> list[str]:
    """Set status='include' on the best `max_papers` rows so fetch_fulltext lands them.

    Rows already 'fulltext'/'PAYWALLED' keep their status. Returns chosen paper_nos.
    """
    papers = read_jsonl(ledger_path)
    candidates = [p for p in papers if p.get("status") in (None, "", "new", "include")]
    if require_oa:
        candidates = [p for p in candidates
                      if p.get("arxiv_id") or p.get("pmcid") or p.get("pdf_url")]
    candidates.sort(key=_landable_rank, reverse=True)
    chosen = {p["paper_no"] for p in candidates[:max_papers]}
    for p in papers:
        if p["paper_no"] in chosen and p.get("status") not in ("fulltext", "PAYWALLED"):
            p["status"] = "include"
    write_jsonl(ledger_path, papers)
    return sorted(chosen)


# tokens that carry no signal as a snippet anchor: stopwords, booleans, and the field
# prefixes of native search syntax (arXiv all:/ti:/abs:, PubMed [tiab]/[ab]/[mesh], ...)
_TERM_STOP = {
    "the", "a", "an", "of", "in", "on", "for", "and", "or", "not", "to", "with", "by",
    "using", "via", "from", "is", "are", "that", "this", "we", "our", "new", "based",
    "all", "ti", "abs", "au", "cat", "jr", "rn", "co", "tiab", "ab", "mesh", "so", "dp",
}
_TERM_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")
_TERM_PHRASE = re.compile(r'"([^"]{3,})"')


def query_terms(queries: list[str]) -> list[str]:
    """Snippet anchors derived from queries: quoted phrases first (they tend to appear
    verbatim), then individual keywords — minus stopwords / booleans / field prefixes.
    Splitting matters: a whole query phrase rarely occurs verbatim in a paper, so without
    this the snippet stage finds nothing and extraction falls back to the document head."""
    seen: set[str] = set()
    terms: list[str] = []

    def add(t: str) -> None:
        t = t.strip()
        k = t.lower()
        if t and k not in seen and k not in _TERM_STOP:
            seen.add(k)
            terms.append(t)

    for q in queries:
        for ph in _TERM_PHRASE.findall(q):
            add(ph)
        for m in _TERM_WORD.finditer(q):
            add(m.group(0))
    return terms


def make_terms_file(path: Path, terms: list[str], queries: list[str]) -> Path:
    """Write terms.txt for extract_snippets: explicit --terms, else keywords from queries."""
    lines = terms or query_terms(queries) or [q for q in queries if q.strip()]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return Path(path)


def gather_snippets(workdir: Path, pno: str, max_windows: int = 24,
                    max_chars: int = 12000) -> str:
    """Concatenate a paper's snippet windows (fallback: head of the full text)."""
    snip = Path(workdir) / "snippets" / f"{pno}.jsonl"
    windows: list[str] = []
    if snip.exists():
        seen: set[str] = set()
        for row in read_jsonl(snip):
            w = (row.get("window") or "").strip()
            key = w[:80]
            if w and key not in seen:
                seen.add(key)
                windows.append(w)
            if len(windows) >= max_windows:
                break
    if not windows:
        txt = Path(workdir) / "papers" / f"{pno}.txt"
        if txt.exists():
            windows = [txt.read_text(encoding="utf-8", errors="replace")[:max_chars]]
    blob = "\n---\n".join(windows)
    return blob[:max_chars]


def parse_cards(raw: Any, pno: str, start_index: int = 1) -> list[dict]:
    """Normalize an LLM card payload into gate-ready card dicts with stable ids."""
    if isinstance(raw, dict):
        raw = raw.get("cards", raw.get("evidence", []))
    if not isinstance(raw, list):
        return []
    cards, i = [], start_index
    for item in raw:
        if not isinstance(item, dict):
            continue
        quote = (item.get("quote") or "").strip()
        if not quote:
            continue
        nums = item.get("numbers") or []
        if not isinstance(nums, list):
            nums = [str(nums)]
        cards.append({
            "card_id": f"{pno}-c{i}",
            "paper": pno,
            "claim": (item.get("claim") or "").strip(),
            "quote": quote[:300],
            "section": (item.get("section") or "").strip(),
            "numbers": [str(n).strip() for n in nums if str(n).strip()],
        })
        i += 1
    return cards


REPAIR_SYSTEM = (
    "You copy EXACT verbatim quotes. For each claim, return the single sentence from the "
    "SNIPPETS that supports it, copied character-for-character with no edits, paraphrase, "
    "or stitching across sentences. If no single sentence supports the claim, return \"\"."
)


def _repair_quotes(llm: Any, pno: str, bad: list[dict], snippets: str,
                   source_text: str) -> list[dict]:
    """One re-prompt that asks the model to copy a verbatim sentence for each failed card.

    Returns the subset whose repaired quote now verifies. This is what turns a stitched /
    hallucinated quote (reads like the paper but is not in it) into either a real verbatim
    quote or a dropped card — instead of leaning on the downstream gate to reject them all.
    """
    claims = "\n".join(f"- {c['card_id']}: {c['claim']}" for c in bad)
    prompt = (f"CLAIMS:\n{claims}\n\nSNIPPETS:\n{snippets}\n\n"
              'Return JSON {"quotes": {"<card_id>": "<exact verbatim sentence or empty>"}}.')
    try:
        r = llm.complete_json(REPAIR_SYSTEM, prompt)
    except Exception:  # noqa: BLE001
        return []
    fixed = r.get("quotes", {}) if isinstance(r, dict) else {}
    out = []
    for c in bad:
        q = str(fixed.get(c["card_id"], "")).strip()
        if q and verify_quote(q, source_text, c.get("numbers"))["ok"]:
            out.append({**c, "quote": q[:300]})
    return out


def extract_cards_llm(llm: Any, pno: str, snippets: str, max_cards: int = 6,
                      source_text: str | None = None) -> list[dict]:
    """One LLM call -> cards for one paper. If `source_text` is given, quotes are checked
    verbatim on the spot and non-verbatim ones get one repair re-prompt (kept only if the
    repaired quote verifies). Never raises; failures -> []."""
    if not snippets.strip():
        return []
    prompt = CARD_PROMPT.format(pno=pno, max_cards=max_cards, snippets=snippets)
    try:
        raw = llm.complete_json(CARD_SYSTEM, prompt)
    except Exception as e:  # noqa: BLE001 — a flaky paper must not sink the batch
        print(f"  [cards] {pno}: LLM extraction failed ({e}); 0 cards", file=sys.stderr)
        return []
    cards = parse_cards(raw, pno)
    if source_text is None:
        return cards
    good = [c for c in cards if verify_quote(c["quote"], source_text, c.get("numbers"))["ok"]]
    bad = [c for c in cards if c not in good]
    if bad:
        recovered = _repair_quotes(llm, pno, bad, snippets, source_text)
        print(f"  [cards] {pno}: {len(good)} verbatim, {len(bad)} non-verbatim -> "
              f"{len(recovered)} repaired", file=sys.stderr)
        good += recovered
    return good


def install_corpus(root: Path, workdir: Path, cards: list[dict],
                   paper_nos: list[str] | None = None) -> dict[str, Any]:
    """Copy landed texts + write cards under data/evidence/ for the quote gate."""
    ev = Path(root) / "data" / "evidence"
    (ev / "papers").mkdir(parents=True, exist_ok=True)
    src_dir = Path(workdir) / "papers"
    copied = []
    want = set(paper_nos) if paper_nos else None
    for txt in sorted(src_dir.glob("*.txt")):
        if want is None or txt.stem in want:
            shutil.copy2(txt, ev / "papers" / txt.name)
            copied.append(txt.stem)
    write_jsonl(ev / "cards.jsonl", cards)
    return {"evidence_dir": str(ev), "papers_copied": copied, "n_cards": len(cards)}


# --------------------------------------------------------------------------- #
# orchestration (network + LLM)
# --------------------------------------------------------------------------- #
def _run_script(skill_dir: Path, name: str, args: list[str], env_email: str | None) -> None:
    import os

    cmd = [sys.executable, str(Path(skill_dir) / name), *args]
    env = dict(os.environ)
    if env_email:
        env.setdefault("CONTACT_EMAIL", env_email)
    print(f"  $ {name} {' '.join(args)}", file=sys.stderr)
    subprocess.run(cmd, check=True, env=env)


def _resolve_skill_dir(skill_dir: Path | None) -> Path:
    skill_dir = Path(skill_dir or DEFAULT_SKILL_DIR)
    if not (skill_dir / "search_papers.py").exists():
        raise FileNotFoundError(f"paper-deep-search scripts not found at {skill_dir} "
                                "(install the skill or pass --skill-dir).")
    return skill_dir


def _snippets_for(workdir: Path, pno: str, src: str, question: str | None,
                  semantic: bool) -> str:
    """Snippet blob for extraction: semantic (question-ranked chunks) when requested and
    possible, else the keyword-anchored windows from the skill's extract_snippets."""
    if semantic and question:
        try:
            from .semantic import semantic_windows
            wins = semantic_windows(src, question, k=20)
            if wins:
                return "\n---\n".join(wins)[:14000]
        except Exception as e:  # noqa: BLE001 — embeddings optional; fall back to keywords
            print(f"  [semantic] {pno}: {e}; falling back to keyword windows", file=sys.stderr)
    return gather_snippets(workdir, pno)


def land_and_card(root: Path, queries: list[str], max_papers: int = 8,
                  terms: list[str] | None = None, skill_dir: Path | None = None,
                  use_llm: bool = True, max_cards: int = 6, contact_email: str | None = None,
                  workdir: Path | None = None, question: str | None = None,
                  semantic: bool = False) -> dict[str, Any]:
    """Triage an existing ledger, land full texts, extract + install verbatim cards.

    Assumes <workdir>/ledger.jsonl already exists (from evidence.run's search or from
    recall.run). This is the half after search — so a coverage-first loop can grow the
    ledger with recall.run and then land the winners here without searching again.
    """
    from .llm import api_key_available, get_client

    root = Path(root)
    skill_dir = _resolve_skill_dir(skill_dir)
    workdir = Path(workdir or root / "data" / "evidence" / "_work")

    chosen = mark_include(workdir / "ledger.jsonl", max_papers)
    print(f"[evidence] triage: {len(chosen)} paper(s) marked include", file=sys.stderr)

    _run_script(skill_dir, "fetch_fulltext.py", ["--workdir", str(workdir)], contact_email)

    landed = sorted(p.stem for p in (workdir / "papers").glob("*.txt"))
    if landed and not (semantic and question):   # semantic path doesn't need the skill's snippets
        terms_path = make_terms_file(workdir / "terms.txt", terms or [], queries)
        _run_script(skill_dir, "extract_snippets.py",
                    ["--workdir", str(workdir), "--terms", str(terms_path)], contact_email)

    cards: list[dict] = []
    if use_llm and landed and api_key_available():
        llm = get_client()
        mode = "semantic" if (semantic and question) else "keyword"
        print(f"[evidence] extracting cards from {len(landed)} paper(s) via LLM ({mode})…",
              file=sys.stderr)
        for pno in landed:
            src = (workdir / "papers" / f"{pno}.txt").read_text(encoding="utf-8", errors="replace")
            snips = _snippets_for(workdir, pno, src, question, semantic)
            cards.extend(extract_cards_llm(llm, pno, snips, max_cards=max_cards, source_text=src))
    elif use_llm and landed:
        print("[evidence] no LLM key — skipping card extraction "
              "(draft-quote scanning still works).", file=sys.stderr)

    summary = install_corpus(root, workdir, cards, landed)
    summary.update(landed=len(landed))
    print(f"[evidence] corpus ready: {len(landed)} paper(s), {len(cards)} card(s) "
          f"-> {summary['evidence_dir']}", file=sys.stderr)
    return summary


def run(root: Path, queries: list[str],
        sources: str = "arxiv,openalex,semanticscholar,pubmed,europepmc,crossref",
        max_per_source: int = 25, max_papers: int = 8, terms: list[str] | None = None,
        skill_dir: Path | None = None, use_llm: bool = True, max_cards: int = 6,
        contact_email: str | None = None, workdir: Path | None = None,
        question: str | None = None, semantic: bool = False) -> dict[str, Any]:
    """Full search -> land -> snippet -> card -> install pipeline. Returns a summary."""
    root = Path(root)
    skill_dir = _resolve_skill_dir(skill_dir)
    workdir = Path(workdir or root / "data" / "evidence" / "_work")
    (workdir / "results").mkdir(parents=True, exist_ok=True)
    sources = expand_sources(sources)

    # Stage 1 — search
    for i, q in enumerate(queries, 1):
        out = workdir / "results" / f"q{i:02d}.jsonl"
        _run_script(skill_dir, "search_papers.py",
                    ["--source", sources, "--query", q, "--max", str(max_per_source),
                     "--out", str(out)], contact_email)

    # Stage 2 — dedupe
    _run_script(skill_dir, "dedupe.py",
                ["--inputs", str(workdir / "results" / "*.jsonl"),
                 "--workdir", str(workdir)], contact_email)

    # Stages triage -> land -> snippet -> card -> install
    summary = land_and_card(root, queries=queries, max_papers=max_papers, terms=terms,
                            skill_dir=skill_dir, use_llm=use_llm, max_cards=max_cards,
                            contact_email=contact_email, workdir=workdir,
                            question=question, semantic=semantic)
    summary.update(queries=len(queries), sources=sources)
    return summary
