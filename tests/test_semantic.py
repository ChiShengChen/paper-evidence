"""Semantic anchoring + deep-read tests (pure helpers with a fake embedder; no network)."""
from paper_evidence import deepread, semantic


def test_chunk_text_sizes_and_overlap():
    text = ". ".join(f"sentence number {i} about flies" for i in range(40)) + "."
    chunks = semantic.chunk_text(text, size=120, overlap=30)
    assert len(chunks) > 1
    assert all(len(c) <= 240 for c in chunks)          # no runaway chunks
    assert semantic.chunk_text("") == []


def test_cosine():
    assert semantic.cosine([1, 0], [1, 0]) == 1.0
    assert semantic.cosine([1, 0], [0, 1]) == 0.0
    assert semantic.cosine([], [1]) == 0.0


def test_rank_chunks_orders_by_similarity():
    # fake embedder: 2-D vectors keyed by a marker word so we control similarity
    def fake_embed(texts):
        out = []
        for t in texts:
            if "smell" in t or "odor" in t:
                out.append([1.0, 0.0])
            elif "visual" in t:
                out.append([0.0, 1.0])
            else:
                out.append([0.5, 0.5])
        return out
    chunks = ["odor coding in the mushroom body", "visual processing in the optic lobe",
              "general anatomy of the brain"]
    ranked = semantic.rank_chunks("how are smells encoded?", chunks, fake_embed, k=2)
    assert ranked[0][2] == "odor coding in the mushroom body"     # smell-aligned chunk first
    assert len(ranked) == 2


def test_semantic_windows_uses_injected_embed_fn():
    def fake_embed(texts):
        hot = ("dopamine", "punish", "valence", "negative")
        return [[1.0, 0.0] if any(h in t.lower() for h in hot) else [0.0, 1.0] for t in texts]
    text = ("The retina processes visual input in the optic lobe. "
            "Dopaminergic neurons convey punishment signals during training.")
    wins = semantic.semantic_windows(text, "negative valence in learning", k=1,
                                     size=60, overlap=10, embed_fn=fake_embed)
    assert wins and "punishment" in wins[0]


def test_keyword_windows():
    text = "A. The dopamine neuron fires. B. The kenyon cell is sparse. C. End."
    wins = semantic.keyword_windows(text, ["dopamine", "kenyon"], window=10)
    assert any("dopamine" in w for w in wins) and any("kenyon" in w for w in wins)
    assert semantic.keyword_windows(text, ["nonexistent"]) == []


# --------------------------------------------------------------------------- #
# deep read
# --------------------------------------------------------------------------- #
def test_load_text_from_txt(tmp_path):
    p = tmp_path / "paper.txt"
    p.write_text("already extracted text", encoding="utf-8")
    assert deepread.load_text(pdf=str(p)) == "already extracted text"


def test_load_text_requires_a_source():
    import pytest
    with pytest.raises(ValueError):
        deepread.load_text()
    with pytest.raises(ValueError):
        deepread.load_text(arxiv="not-an-id")


class _SynthLLM:
    def complete(self, system, prompt, **kw):
        return "The paper shows X [P01-c1]."


def test_synthesize_uses_cards_or_stubs():
    assert "No verified" in deepread.synthesize(_SynthLLM(), [])
    out = deepread.synthesize(_SynthLLM(), [{"card_id": "P01-c1", "claim": "x", "quote": "q"}])
    assert out == "The paper shows X [P01-c1]."
