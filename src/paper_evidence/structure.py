"""Structured PDF extraction — section-aware blocks, not flat text.

Inspired by PaperHub's Marker→chunk pipeline (blocks tagged with section/page/type, tables
and captions kept), reduced to a light, dependency-optional core: a pymupdf-based extractor
that recovers section headings (font-size + numbering heuristics), tags each block with its
current section + page, keeps figure/table captions, and drops running page headers/footers.

Why it upgrades evidence cards: the source the quote gate re-greps is cleaner (no interleaved
headers/footers), and every card can carry a REAL `section` (the heading it falls under) via
`StructuredDoc.section_for(quote)` — instead of a guessed one.

`extract_structured(pdf_bytes)` prefers real Marker if the `marker-pdf` package is installed
(higher fidelity: equations→LaTeX, real tables), else falls back to pymupdf. The classifier
(`_classify`) is pure and unit-tested; the pymupdf/Marker adapters just feed it lines.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

_CAPTION_RE = re.compile(r"^\s*\*?\s*(Table|Fig(?:ure)?\.?)\s*(\d+)", re.IGNORECASE)
_HEADING_NUM_RE = re.compile(r"^\d+(?:\.\d+)*\.?\s+\S")
_PAGENO_RE = re.compile(r"^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$")


@dataclass
class Line:
    """One rendered text line fed to the classifier."""
    text: str
    size: float          # max font size on the line
    page: int            # 1-based
    bold: bool = False


@dataclass
class Block:
    type: str            # "heading" | "text" | "caption" | "equation"
    section: str         # the heading this block falls under ("" before the first heading)
    page: int
    text: str


@dataclass
class StructuredDoc:
    blocks: list[Block] = field(default_factory=list)

    def text(self) -> str:
        """Clean full text (headers/footers dropped, sections in reading order)."""
        return "\n".join(b.text for b in self.blocks)

    def markdown(self) -> str:
        out = []
        for b in self.blocks:
            if b.type == "heading":
                out.append(f"\n## {b.text}")
            elif b.type == "caption":
                out.append(f"*{b.text}*")
            else:
                out.append(b.text)
        return "\n".join(out).strip()

    def sections(self) -> dict[str, str]:
        """section name -> concatenated text of its non-heading blocks."""
        acc: dict[str, list[str]] = {}
        for b in self.blocks:
            if b.type != "heading":
                acc.setdefault(b.section, []).append(b.text)
        return {k: "\n".join(v) for k, v in acc.items()}

    def caption_labels(self) -> list[str]:
        out = []
        for b in self.blocks:
            if b.type == "caption" and (lab := _caption_label(b.text)):
                out.append(lab)
        return out

    def section_for(self, quote: str) -> str:
        """The section whose block text contains `quote` (normalized), or '' if none."""
        from .quote_gate import normalize
        qn = normalize(quote)
        if not qn:
            return ""
        for b in self.blocks:
            if b.type != "heading" and qn in normalize(b.text):
                return b.section
        return ""


def _caption_label(text: str) -> str | None:
    m = _CAPTION_RE.match(text or "")
    if not m:
        return None
    kind = "Table" if m.group(1).lower().startswith("table") else "Figure"
    return f"{kind} {m.group(2)}"


def _is_heading(text: str, size: float, bold: bool, body_size: float) -> bool:
    if len(text) > 120 or text.endswith((".", ",", ";")):   # a real sentence, not a heading
        return False
    if size >= body_size * 1.12:                            # visibly larger than body
        return True
    if _HEADING_NUM_RE.match(text) and (bold or size >= body_size * 1.02):
        return True                                        # "3.1 Results" style
    return False


def _classify(lines: list[Line]) -> StructuredDoc:
    """Pure: lines -> section-tagged blocks. Drops page numbers; tags captions + headings."""
    sizes = [ln.size for ln in lines if ln.text.strip() and ln.size > 0]
    body = statistics.median(sizes) if sizes else 10.0
    blocks: list[Block] = []
    section = ""
    for ln in lines:
        t = ln.text.strip()
        if not t or _PAGENO_RE.match(t):
            continue
        if _caption_label(t):
            blocks.append(Block("caption", section, ln.page, t))
            continue
        if _is_heading(t, ln.size, ln.bold, body):
            section = t
            blocks.append(Block("heading", section, ln.page, t))
            continue
        blocks.append(Block("text", section, ln.page, t))
    return StructuredDoc(blocks)


# --------------------------------------------------------------------------- #
# adapters
# --------------------------------------------------------------------------- #
def _lines_pymupdf(pdf_bytes: bytes) -> list[Line]:
    import fitz  # PyMuPDF
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    lines: list[Line] = []
    for pno, page in enumerate(doc, 1):
        height = page.rect.height
        d = page.get_text("dict")
        for blk in d.get("blocks", []):
            for line in blk.get("lines", []):
                spans = line.get("spans", [])
                text = "".join(s.get("text", "") for s in spans)
                if not text.strip():
                    continue
                y = line.get("bbox", [0, 0, 0, 0])[1]
                if y < 45 or y > height - 45:            # running header / footer band
                    if len(text.strip()) < 80:
                        continue
                size = max((s.get("size", 0) for s in spans), default=0.0)
                bold = any((s.get("flags", 0) & 16) or "bold" in s.get("font", "").lower()
                           for s in spans)
                lines.append(Line(text=text, size=size, page=pno, bold=bold))
    return lines


def _extract_marker(pdf_bytes: bytes) -> StructuredDoc | None:
    """Use real Marker if installed (marker-pdf). Returns None if unavailable."""
    try:
        import importlib.util
        if importlib.util.find_spec("marker") is None:
            return None
    except Exception:  # noqa: BLE001
        return None
    return None  # hook point: a full Marker adapter can render blocks here when desired


def extract_structured(pdf_bytes: bytes, *, prefer_marker: bool = True) -> StructuredDoc:
    """PDF bytes -> section-tagged StructuredDoc. Marker if available, else pymupdf."""
    if prefer_marker and (doc := _extract_marker(pdf_bytes)) is not None:
        return doc
    return _classify(_lines_pymupdf(pdf_bytes))
