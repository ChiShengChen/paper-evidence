"""Recall layer for the paper-deep-search corpus: make coverage measurable.

Three jobs the evidence orchestrator (evidence.py) does not do — it lands and cards
whatever it is given, but says nothing about whether the *right* papers were found:

  1. Saturation-driven search — run query variants, dedupe after each, and STOP only
     when >=k consecutive batches add 0 new unique papers (the skill's stopping rule),
     instead of after a fixed number of queries. Optional LLM query expansion keeps
     formulating variants until saturation is actually reached.
  2. Snowball — seed from the most-cited hits + any survey, chase references+citations
     (snowball.py), re-dedupe. Catches papers whose authors used vocabulary no query
     guessed.
  3. Recall audit — take a survey from the ledger, pull its reference list, and diff it
     against the ledger. |refs ∩ ledger| / |refs| is a concrete recall number, and the
     misses are a to-chase list.

Network stages live in run(); the scoring helpers below are pure and unit-tested. The
skill scripts (search_papers.py, dedupe.py, snowball.py) are shelled out, sharing the
evidence workdir so the two orchestrators build one ledger.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .evidence import DEFAULT_SKILL_DIR, _run_script, expand_sources, read_jsonl, write_jsonl

_SURVEY_RE = re.compile(r"\b(survey|review|overview|systematic|tutorial|taxonomy)\b", re.I)
_ARXIV_RE = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?")


# --------------------------------------------------------------------------- #
# id / title normalization + matching (self-contained; mirrors the skill's keys)
# --------------------------------------------------------------------------- #
def norm_doi(d: Any) -> str:
    if not d:
        return ""
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", str(d).strip(), flags=re.I)
    d = d.strip().strip(".").lower()
    return d if d.startswith("10.") else ""


def norm_arxiv(s: Any) -> str:
    if not s:
        return ""
    m = _ARXIV_RE.search(str(s))
    return m.group(1) if m else ""


def norm_title(t: Any) -> str:
    if not t:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(t).lower())).strip()


def paper_keys(r: dict) -> set[str]:
    """Strong keys (doi/arxiv) plus a normalized-title fallback."""
    ks: set[str] = set()
    if (d := norm_doi(r.get("doi"))):
        ks.add("doi:" + d)
    if (a := norm_arxiv(r.get("arxiv_id"))):
        ks.add("arxiv:" + a)
    if (nt := norm_title(r.get("title"))) and len(nt) > 8:
        ks.add("ti:" + nt)
    return ks


def is_survey(title: str) -> bool:
    return bool(_SURVEY_RE.search(title or ""))


def seed_id(r: dict) -> str:
    """A snowball seed string for a ledger row, preferring arXiv then DOI. '' if neither."""
    if (a := norm_arxiv(r.get("arxiv_id"))):
        return f"arxiv:{a}"
    if (d := norm_doi(r.get("doi"))):
        return f"doi:{d}"
    return ""


def _citations(r: dict) -> int:
    try:
        return int(r.get("citations") or 0)
    except (TypeError, ValueError):
        return 0


# --------------------------------------------------------------------------- #
# saturation
# --------------------------------------------------------------------------- #
def new_count(before: int, after: int) -> int:
    return max(0, after - before)


def saturated(new_counts: list[int], k: int = 3) -> bool:
    """True once the last k batches each contributed 0 new unique papers."""
    return len(new_counts) >= k and all(n == 0 for n in new_counts[-k:])


# --------------------------------------------------------------------------- #
# seed selection + recall diff
# --------------------------------------------------------------------------- #
def pick_seeds(ledger: list[dict], n: int = 5) -> list[str]:
    """Surveys first, then most-cited papers; return up to n resolvable seed ids."""
    surveys = [r for r in ledger if is_survey(r.get("title", ""))]
    rest = sorted((r for r in ledger if not is_survey(r.get("title", ""))),
                  key=_citations, reverse=True)
    seeds: list[str] = []
    for r in surveys + rest:
        sid = seed_id(r)
        if sid and sid not in seeds:
            seeds.append(sid)
        if len(seeds) >= n:
            break
    return seeds


def pick_survey(ledger: list[dict]) -> dict | None:
    """The most-cited survey/review in the ledger that we can seed from, or None."""
    surveys = [r for r in ledger if is_survey(r.get("title", "")) and seed_id(r)]
    return max(surveys, key=_citations) if surveys else None


def recall_diff(refs: list[dict], ledger: list[dict]) -> dict[str, Any]:
    """Fraction of a survey's (deduped) references already in the ledger, plus the misses."""
    led_keys: set[str] = set()
    for r in ledger:
        led_keys |= paper_keys(r)
    seen: set[str] = set()
    found, misses = 0, []
    n = 0
    for r in refs:
        ks = paper_keys(r)
        if not ks or (ks & seen):
            continue
        seen |= ks
        n += 1
        if ks & led_keys:
            found += 1
        else:
            misses.append({"title": r.get("title", ""), "year": r.get("year", ""),
                           "doi": norm_doi(r.get("doi")), "arxiv_id": norm_arxiv(r.get("arxiv_id")),
                           "citations": r.get("citations", "")})
    return {"n_refs": n, "n_found": found,
            "recall": (round(found / n, 3) if n else None),
            "misses": sorted(misses, key=lambda m: _citations(m), reverse=True)}


# --------------------------------------------------------------------------- #
# optional LLM query expansion (to actually reach saturation)
# --------------------------------------------------------------------------- #
EXPAND_SYSTEM = (
    "You expand a literature-search query into diverse alternative formulations that a "
    "high-recall search would need: synonyms, abbreviation<->full form, cross-domain "
    "wording, and spelling/hyphenation variants. Return only new formulations."
)
EXPAND_PROMPT = ('Research question: {question}\n\nQueries already tried:\n{tried}\n\n'
                 'Return JSON {{"queries": ["...", ...]}} with up to {n} NEW search '
                 'queries not equivalent to any already tried.')


def expand_queries(llm: Any, question: str, tried: list[str], n: int = 4) -> list[str]:
    """LLM-propose new query variants; [] on any failure (never raises)."""
    if llm is None or not question:
        return []
    try:
        r = llm.complete_json(EXPAND_SYSTEM, EXPAND_PROMPT.format(
            question=question, tried="\n".join(f"- {q}" for q in tried), n=n))
    except Exception:  # noqa: BLE001
        return []
    qs = r.get("queries", []) if isinstance(r, dict) else []
    lowered = {q.lower() for q in tried}
    return [q.strip() for q in qs if isinstance(q, str) and q.strip()
            and q.strip().lower() not in lowered][:n]


# --------------------------------------------------------------------------- #
# orchestration (network)
# --------------------------------------------------------------------------- #
def _ledger_size(workdir: Path) -> int:
    p = Path(workdir) / "ledger.jsonl"
    return len(read_jsonl(p)) if p.exists() else 0


def _dedupe(skill_dir: Path, workdir: Path, email: str | None) -> None:
    _run_script(skill_dir, "dedupe.py",
                ["--inputs", str(workdir / "results" / "*.jsonl"), "--workdir", str(workdir)],
                email)


def run(root: Path, queries: list[str],
        sources: str = "arxiv,openalex,semanticscholar,pubmed,europepmc,crossref",
        max_per_source: int = 25, sat_k: int = 3, seed_n: int = 5, snowball_max: int = 150,
        skill_dir: Path | None = None, contact_email: str | None = None,
        workdir: Path | None = None, research_question: str | None = None,
        expand_llm: Any = None, max_expand: int = 8) -> dict[str, Any]:
    """Saturation search -> snowball -> recall audit. Returns a summary dict."""
    root = Path(root)
    skill_dir = Path(skill_dir or DEFAULT_SKILL_DIR)
    if not (skill_dir / "search_papers.py").exists():
        raise FileNotFoundError(f"paper-deep-search scripts not found at {skill_dir}")
    workdir = Path(workdir or root / "data" / "evidence" / "_work")
    (workdir / "results").mkdir(parents=True, exist_ok=True)
    sources = expand_sources(sources)

    # --- 1. saturation-driven search ---
    tried: list[str] = []
    new_counts: list[int] = []
    q_idx = 0
    q_list = list(queries)
    while q_idx < len(q_list):
        q = q_list[q_idx]
        q_idx += 1
        tried.append(q)
        before = _ledger_size(workdir)
        out = workdir / "results" / f"q{len(tried):02d}.jsonl"
        _run_script(skill_dir, "search_papers.py",
                    ["--source", sources, "--query", q, "--max", str(max_per_source),
                     "--out", str(out)], contact_email)
        _dedupe(skill_dir, workdir, contact_email)
        nc = new_count(before, _ledger_size(workdir))
        new_counts.append(nc)
        print(f"[recall] batch {len(tried)} '{q[:48]}' -> {nc} new "
              f"(ledger {_ledger_size(workdir)})")
        if saturated(new_counts, sat_k):
            print(f"[recall] saturated: {sat_k} consecutive zero-yield batches.")
            break
        # exhausted the provided variants but not saturated -> let the LLM propose more
        if q_idx >= len(q_list) and expand_llm is not None and len(tried) < len(queries) + max_expand:
            more = expand_queries(expand_llm, research_question or (queries[0] if queries else ""),
                                  tried, n=min(4, len(queries) + max_expand - len(tried)))
            if more:
                q_list.extend(more)
                print(f"[recall] +{len(more)} LLM query variant(s) toward saturation")

    # --- 2. snowball ---
    ledger = read_jsonl(workdir / "ledger.jsonl") if _ledger_size(workdir) else []
    seeds = pick_seeds(ledger, n=seed_n)
    snow_new = 0
    if seeds:
        before = _ledger_size(workdir)
        args = []
        for s in seeds:
            args += ["--id", s]
        args += ["--direction", "both", "--max", str(snowball_max),
                 "--out", str(workdir / "results" / "snow.jsonl")]
        _run_script(skill_dir, "snowball.py", args, contact_email)
        _dedupe(skill_dir, workdir, contact_email)
        snow_new = new_count(before, _ledger_size(workdir))
        print(f"[recall] snowball on {len(seeds)} seed(s) -> {snow_new} new "
              f"(ledger {_ledger_size(workdir)})")

    # --- 3. recall audit against a survey's references ---
    ledger = read_jsonl(workdir / "ledger.jsonl") if _ledger_size(workdir) else []
    survey = pick_survey(ledger)
    audit = None
    if survey:
        refs_out = workdir / "results" / "survey_refs.jsonl"
        _run_script(skill_dir, "snowball.py",
                    ["--id", seed_id(survey), "--direction", "refs",
                     "--max", str(snowball_max), "--out", str(refs_out)], contact_email)
        refs = read_jsonl(refs_out) if refs_out.exists() else []
        audit = recall_diff(refs, ledger)
        audit["survey_title"] = survey.get("title", "")
        print(f"[recall] audit vs '{survey.get('title','')[:50]}': "
              f"recall {audit['recall']} ({audit['n_found']}/{audit['n_refs']}), "
              f"{len(audit['misses'])} miss(es)")

    summary = {
        "ledger_size": _ledger_size(workdir),
        "batches": len(tried),
        "new_per_batch": new_counts,
        "saturated": saturated(new_counts, sat_k),
        "snowball_seeds": seeds,
        "snowball_new": snow_new,
        "audit": audit,
    }
    _write_recall_report(workdir / "recall_report.md", summary, tried)
    return summary


def _write_recall_report(path: Path, s: dict[str, Any], tried: list[str]) -> None:
    L = [f"# Recall report — ledger {s['ledger_size']} papers", ""]
    L += [f"- batches run: {s['batches']} (saturated: {s['saturated']})",
          f"- new per batch: {s['new_per_batch']}",
          f"- snowball: {s['snowball_new']} new from {len(s['snowball_seeds'])} seed(s)", ""]
    L += ["## Queries tried", ""] + [f"{i+1}. {q}" for i, q in enumerate(tried)] + [""]
    a = s.get("audit")
    if a:
        L += [f"## Recall audit vs survey", "",
              f"- survey: {a.get('survey_title','')}",
              f"- recall: **{a['recall']}** ({a['n_found']}/{a['n_refs']} references in ledger)", ""]
        if a["misses"]:
            L += ["### Missed references (chase these)", "",
                  "| title | year | citations | id |", "|---|---|---|---|"]
            L += [f"| {m['title'][:70]} | {m['year']} | {m['citations']} | "
                  f"{m['arxiv_id'] or m['doi']} |" for m in a["misses"][:40]]
    else:
        L += ["## Recall audit", "", "_no survey/review in the ledger to audit against._"]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
