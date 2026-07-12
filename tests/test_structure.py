"""Structured-extraction tests: the pure classifier (headings / captions / page-number drop /
section tagging). The pymupdf adapter is exercised live in a separate smoke, not here."""
from paper_evidence import structure
from paper_evidence.structure import Line


def _doc():
    return structure._classify([
        Line("Abstract", size=13, page=1, bold=True),
        Line("We study heading-direction encoding in the fly.", size=10, page=1),
        Line("3 Results", size=13, page=2, bold=True),
        Line("The model reached an accuracy of 0.912 on the cohort.", size=10, page=2),
        Line("Figure 1: the ring attractor network.", size=9, page=2),
        Line("12", size=10, page=2),                     # page number -> dropped
        Line("running header text", size=8, page=2),     # small -> stays as text here
    ])


def test_headings_sections_and_captions():
    d = _doc()
    kinds = [(b.type, b.section) for b in d.blocks]
    assert ("heading", "Abstract") in kinds
    assert ("heading", "3 Results") in kinds
    # body text is tagged with the section it falls under
    body = next(b for b in d.blocks if b.text.startswith("The model reached"))
    assert body.type == "text" and body.section == "3 Results"
    # caption typed + labelled
    cap = next(b for b in d.blocks if b.type == "caption")
    assert cap.section == "3 Results"
    assert d.caption_labels() == ["Figure 1"]
    # bare page number dropped
    assert all(b.text != "12" for b in d.blocks)


def test_section_for_quote():
    d = _doc()
    assert d.section_for("accuracy of 0.912") == "3 Results"     # normalized substring match
    assert d.section_for("heading-direction encoding") == "Abstract"
    assert d.section_for("nowhere in the paper") == ""


def test_sections_and_markdown():
    d = _doc()
    secs = d.sections()
    assert "3 Results" in secs and "0.912" in secs["3 Results"]
    md = d.markdown()
    assert "## 3 Results" in md and "*Figure 1:" in md


def test_heading_and_caption_helpers():
    assert structure._caption_label("Table 2: results") == "Table 2"
    assert structure._caption_label("Fig. 3 shows") == "Figure 3"
    assert structure._caption_label("Ordinary sentence") is None
    # a numbered heading is detected even at body size when bold
    assert structure._is_heading("4.1 Ablations", size=10, bold=True, body_size=10)
    # a full sentence is never a heading
    assert not structure._is_heading("We show that the model works.", 13, True, 10)
