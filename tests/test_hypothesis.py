"""Hypothesis generation + grounding tests (fake LLM/judge, toy corpus, no network)."""
import json

from paper_evidence import hypothesis, quote_gate

CARDS = [
    {"card_id": "P01-c1", "paper": "P01", "claim": "Dopamine gates memory in the mushroom body."},
    {"card_id": "P02-c1", "paper": "P02", "claim": "APL inhibition keeps Kenyon-cell coding sparse."},
]


class _GenLLM:
    def __init__(self, hyps):
        self.hyps = hyps

    def complete_json(self, system, prompt, **kw):
        return {"hypotheses": self.hyps}


class _ContentJudge:
    """Rejects support iff a trigger substring appears in the prompt (else supports)."""
    def __init__(self, reject):
        self.reject = reject.lower()

    def complete_json(self, system, prompt, **kw):
        return {"supported": self.reject not in prompt.lower(), "reason": "test"}


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #
def test_generate_parses_and_drops_empty():
    llm = _GenLLM([
        {"premise": "Dopamine gates memory in the mushroom body [P01].", "prediction": "boost it"},
        {"premise": "   ", "prediction": "x"},                       # empty premise -> dropped
        {"prediction": "no premise key"},                            # dropped
    ])
    out = hypothesis.generate(llm, CARDS, n=3)
    assert len(out) == 1 and out[0]["premise"].startswith("Dopamine gates memory")


def test_generate_empty_without_cards():
    assert hypothesis.generate(_GenLLM([{"premise": "x [P01]"}]), [], n=3) == []


def test_generate_soft_fail():
    class Boom:
        def complete_json(self, *a, **k): raise RuntimeError("down")
    assert hypothesis.generate(Boom(), CARDS, n=3) == []


# --------------------------------------------------------------------------- #
# ground (premise verified against a toy corpus)
# --------------------------------------------------------------------------- #
def _corpus(tmp_path):
    ev = tmp_path / "data" / "evidence"
    (ev / "papers").mkdir(parents=True)
    (ev / "papers" / "P01.txt").write_text(
        "Dopamine gates memory formation in the mushroom body of the fly.", encoding="utf-8")
    (ev / "cards.jsonl").write_text(json.dumps(
        {"card_id": "P01-c1", "paper": "P01",
         "claim": "Dopamine gates memory in the mushroom body.",
         "quote": "Dopamine gates memory formation in the mushroom body of the fly.",
         "numbers": []}) + "\n", encoding="utf-8")
    # build the verified pool
    quote_gate.build(tmp_path, judge=None)
    return tmp_path


def test_ground_keeps_faithful_blocks_unsupported(tmp_path):
    root = _corpus(tmp_path)
    hyps = [
        {"premise": "Dopamine gates memory in the mushroom body [P01].",   # supported
         "prediction": "so boosting dopamine strengthens memory."},
        {"premise": "The mushroom body stores visual place maps for navigation [P01].",  # not supported
         "prediction": "so lesions abolish path integration."},
    ]
    results = hypothesis.ground(root, hyps, judge=_ContentJudge("visual place maps"))
    assert results[0]["grounded"] and results[0]["papers"] == ["P01"]
    assert not results[1]["grounded"]
    assert results[1]["blockers"] and results[1]["blockers"][0]["status"] == "UNSUPPORTED_CLAIM"


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def test_export_markdown_and_jsonl(tmp_path):
    results = [
        {"premise": "P [P01].", "prediction": "pred", "papers": ["P01"], "grounded": True,
         "blockers": []},
        {"premise": "bad [P02].", "prediction": "y", "papers": ["P02"], "grounded": False,
         "blockers": [{"status": "UNSUPPORTED_CLAIM", "reason": "nope", "text": "bad"}]},
    ]
    md = hypothesis.export_markdown(tmp_path / "h.md", results, question="Q?")
    text = md.read_text()
    assert "1 grounded / 2 generated" in text and "**Question:** Q?" in text
    assert "H1  (P01)" in text and "~~bad [P02].~~ — nope" in text

    jl = hypothesis.export_jsonl(tmp_path / "h.jsonl", [r for r in results if r["grounded"]])
    rows = [json.loads(l) for l in jl.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["premise"] == "P [P01]."
