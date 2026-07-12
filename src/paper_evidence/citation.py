"""Verify that a cited identifier is real — resolvable, title-consistent, and not retracted.

The other half of "never invent": quote_gate proves a *quote* is in a source; this proves the
*citation* points at a paper that actually exists. Given a DOI / PMID / arXiv id (and optionally
an expected title), it resolves the id against Crossref, OpenAlex, and PubMed and reports:

  verified    at least one service returned a record
  RETRACTED   a service flags the work as retracted (OpenAlex is_retracted / a retraction title /
              Crossref is-retracted-by) — do not build on it
  unverified  no service resolved the id (likely invented or malformed)

Plus `title_match` when an expected title is given. Network calls are best-effort (urllib, no
extra deps) and never raise — a failed lookup degrades to unverified, not a crash. Inspired by
NeuronLit's citation_verify, reduced to a domain-neutral core.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Any

_UA = "paper-evidence-citation/1.0 (mailto:anonymous@example.com)"
_ARXIV = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?")


def _norm(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def _get_json(url: str, timeout: float = 15.0) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except Exception:  # noqa: BLE001 — a failed lookup is "unverified", never a crash
        return None


# --------------------------------------------------------------------------- #
# per-service resolvers -> {source, title, doi, retracted}
# --------------------------------------------------------------------------- #
def fetch_crossref(doi: str) -> dict | None:
    js = _get_json(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}")
    msg = (js or {}).get("message")
    if not msg:
        return None
    retract = bool((msg.get("relation") or {}).get("is-retracted-by")) or \
        any("retract" in str(u.get("type", "")).lower() for u in (msg.get("update-to") or []))
    return {"source": "crossref", "doi": msg.get("DOI") or doi,
            "title": (msg.get("title") or [None])[0], "retracted": retract}


def fetch_openalex(doi: str) -> dict | None:
    js = _get_json("https://api.openalex.org/works/doi:" + urllib.parse.quote(doi))
    if not js or not js.get("id"):
        return None
    return {"source": "openalex", "doi": doi, "title": js.get("title"),
            "retracted": bool(js.get("is_retracted"))}


def fetch_pubmed(pmid: str) -> dict | None:
    js = _get_json("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?"
                   + urllib.parse.urlencode({"db": "pubmed", "id": pmid, "retmode": "json"}))
    rec = ((js or {}).get("result") or {}).get(str(pmid))
    if not rec or rec.get("error"):
        return None
    pubtypes = " ".join(str(t) for t in (rec.get("pubtype") or []))
    return {"source": "pubmed", "pmid": str(pmid), "title": rec.get("title"),
            "retracted": "retract" in pubtypes.lower()}


def fetch_arxiv(arxiv_id: str) -> dict | None:
    """Resolve an arXiv id against the arXiv API (authoritative — works for pre-2022 papers
    that have no arXiv-minted DOI, unlike an OpenAlex DOI lookup)."""
    m = _ARXIV.search(arxiv_id)
    if not m:
        return None
    aid = m.group(1)
    url = ("https://export.arxiv.org/api/query?"
           + urllib.parse.urlencode({"id_list": aid, "max_results": 1}))
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            xml = r.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return None
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return None
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entry = root.find("a:entry", ns)
    if entry is None:
        return None
    title = (entry.findtext("a:title", "", ns) or "").strip()
    eid = (entry.findtext("a:id", "", ns) or "")
    if not title or title.lower() == "error" or aid not in eid:   # arXiv returns an "Error" entry for bad ids
        return None
    return {"source": "arxiv", "title": re.sub(r"\s+", " ", title), "retracted": False}


# --------------------------------------------------------------------------- #
def is_retracted(records: list[dict]) -> bool:
    if any(r.get("retracted") for r in records):
        return True
    titles = " ".join(_norm(r.get("title")) for r in records)
    return "retracted" in titles or "retraction of" in titles


def title_match(expected: str, resolved: str, threshold: float = 0.6) -> bool | None:
    """Token-overlap match of an expected title against the resolved one (None if either missing)."""
    if not expected or not resolved:
        return None
    a, b = set(_norm(expected).split()), set(_norm(resolved).split())
    if not a or not b:
        return None
    return len(a & b) / len(a | b) >= threshold


def verify_citation(doi: str | None = None, pmid: str | None = None,
                    arxiv: str | None = None, title: str | None = None) -> dict[str, Any]:
    """Resolve one citation across services. Returns status + resolved title + retraction flag."""
    records: list[dict] = []
    if pmid and (r := fetch_pubmed(str(pmid))):
        records.append(r)
    doi = doi or next((r.get("doi") for r in records if r.get("doi")), None)
    if doi:
        for fetch in (fetch_crossref, fetch_openalex):
            if (r := fetch(str(doi))):
                records.append(r)
    if arxiv and not records and (r := fetch_arxiv(str(arxiv))):
        records.append(r)

    resolved = next((r.get("title") for r in records if r.get("title")), None)
    if not records:
        status = "unverified"
    elif is_retracted(records):
        status = "RETRACTED"
    else:
        status = "verified"
    return {"status": status, "resolved_title": resolved,
            "sources": [r.get("source") for r in records if r.get("source")],
            "retracted": status == "RETRACTED",
            "title_match": title_match(title, resolved) if title else None,
            "doi": doi, "pmid": str(pmid) if pmid else None, "arxiv": arxiv,
            "ok": status == "verified" and (title_match(title, resolved) is not False)}


def verify_batch(items: list[dict]) -> dict[str, Any]:
    """Verify many citations (each {doi?/pmid?/arxiv?/title?}). Returns per-item + a summary."""
    results = [{**it, **verify_citation(it.get("doi"), it.get("pmid"),
                                        it.get("arxiv") or it.get("arxiv_id"), it.get("title"))}
               for it in items]
    return {"results": results, "n": len(results),
            "n_verified": sum(1 for r in results if r["status"] == "verified"),
            "n_unverified": sum(1 for r in results if r["status"] == "unverified"),
            "n_retracted": sum(1 for r in results if r["status"] == "RETRACTED")}
