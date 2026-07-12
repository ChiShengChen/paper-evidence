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
    """Sort key: prefer rows we can actually fetch (arXiv/PMC/OA pdf), then citations."""
    has_oa = bool(p.get("arxiv_id") or p.get("pmcid") or p.get("pdf_url"))
    try:
        cites = int(p.get("citations") or 0)
    except (TypeError, ValueError):
        cites = 0
    return (has_oa, cites)


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


def extract_cards_llm(llm: Any, pno: str, snippets: str, max_cards: int = 6) -> list[dict]:
    """One LLM call -> verbatim cards for one paper. Never raises; failures -> []."""
    if not snippets.strip():
        return []
    prompt = CARD_PROMPT.format(pno=pno, max_cards=max_cards, snippets=snippets)
    try:
        raw = llm.complete_json(CARD_SYSTEM, prompt)
    except Exception as e:  # noqa: BLE001 — a flaky paper must not sink the batch
        print(f"  [cards] {pno}: LLM extraction failed ({e}); 0 cards", file=sys.stderr)
        return []
    return parse_cards(raw, pno)


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


def run(root: Path, queries: list[str],
        sources: str = "arxiv,openalex,semanticscholar,pubmed,europepmc,crossref",
        max_per_source: int = 25, max_papers: int = 8, terms: list[str] | None = None,
        skill_dir: Path | None = None, use_llm: bool = True, max_cards: int = 6,
        contact_email: str | None = None, workdir: Path | None = None) -> dict[str, Any]:
    """Full search -> land -> snippet -> card -> install pipeline. Returns a summary."""
    from .llm import api_key_available, get_client

    root = Path(root)
    skill_dir = Path(skill_dir or DEFAULT_SKILL_DIR)
    if not (skill_dir / "search_papers.py").exists():
        raise FileNotFoundError(f"paper-deep-search scripts not found at {skill_dir} "
                                "(install the skill or pass --skill-dir).")
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

    # Triage — pick what to land
    chosen = mark_include(workdir / "ledger.jsonl", max_papers)
    print(f"[evidence] triage: {len(chosen)} paper(s) marked include", file=sys.stderr)

    # Stage 4 — land full texts
    _run_script(skill_dir, "fetch_fulltext.py", ["--workdir", str(workdir)], contact_email)

    # Stage 5a — snippets
    terms_path = make_terms_file(workdir / "terms.txt", terms or [], queries)
    landed = sorted(p.stem for p in (workdir / "papers").glob("*.txt"))
    if landed:
        _run_script(skill_dir, "extract_snippets.py",
                    ["--workdir", str(workdir), "--terms", str(terms_path)], contact_email)

    # Stage 5b — cards (our LLM)
    cards: list[dict] = []
    if use_llm and landed and api_key_available():
        llm = get_client()
        print(f"[evidence] extracting cards from {len(landed)} paper(s) via LLM…",
              file=sys.stderr)
        for pno in landed:
            cards.extend(extract_cards_llm(llm, pno, gather_snippets(workdir, pno),
                                           max_cards=max_cards))
    elif use_llm and landed:
        print("[evidence] no LLM key — skipping card extraction "
              "(draft-quote scanning still works).", file=sys.stderr)

    # Install
    summary = install_corpus(root, workdir, cards, landed)
    summary.update(queries=len(queries), landed=len(landed), sources=sources)
    print(f"[evidence] corpus ready: {len(landed)} paper(s), {len(cards)} card(s) "
          f"-> {summary['evidence_dir']}", file=sys.stderr)
    return summary
