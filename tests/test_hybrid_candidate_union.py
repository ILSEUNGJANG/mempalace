"""Tests for ``candidate_strategy="union"`` in ``search_memories``.

The default ``"vector"`` strategy gathers candidates from the vector index
only. Docs with strong BM25 signal but vector embeddings far from the query
get skipped — terminology guides looked up by narrative-shaped queries are
the canonical case.

The ``"union"`` strategy also pulls top-K BM25-only candidates from sqlite
FTS5 and merges them into the rerank pool. Both signal sources contribute
candidates; the hybrid rerank picks the best from a richer pool.

Default behavior is unchanged ("vector") — these tests exercise opt-in
"union" mode.
"""

from mempalace.palace import get_collection
from mempalace.searcher import search_memories


def _seed_drawers(palace_path):
    """Seed a corpus where the right doc for one query is BM25-strong but
    vector-distant.

    D1-D3 are short narrative tickets that semantically cluster around
    "customer support / order / shipped" vocabulary. D4 is a meta-document
    of bullet rules ("brand voice") that contains rare keywords like
    "Absolutely" and "apologize" the query repeats verbatim — strong BM25
    signal but stylistically far from the narrative tickets.
    """
    col = get_collection(palace_path, create=True)
    col.upsert(
        ids=["D1", "D2", "D3", "D4"],
        documents=[
            "Customer wrote in asking why their order shipped without "
            "the promo sticker. Standard reply explaining the threshold.",
            "Order delivery delayed three days; customer requested a "
            "refund. Support agent processed return via ticket queue.",
            "Customer asked about the missing freebie; the reply "
            "explained the campaign mechanics and shipped status.",
            "Brand voice rules: dry, sturdy, never effusive. "
            "Never 'Absolutely!' Never apologize for policy — explain it. "
            "Avoid premium / curated / elevated vocabulary.",
        ],
        metadatas=[
            {"wing": "shop", "room": "support", "source_file": "ticket_D1.md"},
            {"wing": "shop", "room": "support", "source_file": "ticket_D2.md"},
            {"wing": "shop", "room": "support", "source_file": "ticket_D3.md"},
            {"wing": "shop", "room": "guides", "source_file": "brand_voice_D4.md"},
        ],
    )


_NARRATIVE_QUERY = (
    "A support agent is drafting a reply to a customer asking why their "
    "order shipped without a free sticker. Draft the reply, but never say "
    "'Absolutely!' and do not apologize for policy."
)


class TestCandidateUnion:
    def test_default_vector_strategy_unchanged(self, tmp_path):
        """Default behavior must be identical to omitting the parameter."""
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        without = search_memories(_NARRATIVE_QUERY, palace, n_results=5)
        with_default = search_memories(
            _NARRATIVE_QUERY, palace, n_results=5, candidate_strategy="vector"
        )
        ids_a = [h["source_file"] for h in without["results"]]
        ids_b = [h["source_file"] for h in with_default["results"]]
        assert ids_a == ids_b, "explicit candidate_strategy='vector' must match default"

    def test_union_surfaces_bm25_strong_vector_distant_doc(self, tmp_path):
        """The brand-voice doc has strong BM25 signal for the query but is
        stylistically far from the narrative tickets. Union mode must
        retrieve it; vector-only mode is allowed to miss it."""
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        result = search_memories(_NARRATIVE_QUERY, palace, n_results=5, candidate_strategy="union")
        ids = [h["source_file"] for h in result["results"]]
        assert "brand_voice_D4.md" in ids, (
            "union mode must surface BM25-strong docs even when vector signal "
            f"is weak; got {ids}"
        )

    def test_union_preserves_vector_hits(self, tmp_path):
        """Union mode must not drop docs that vector-only mode finds —
        the rerank pool grows, it doesn't shrink."""
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        vector = search_memories(_NARRATIVE_QUERY, palace, n_results=5, candidate_strategy="vector")
        union = search_memories(_NARRATIVE_QUERY, palace, n_results=5, candidate_strategy="union")
        vec_ids = {h["source_file"] for h in vector["results"]}
        union_ids = {h["source_file"] for h in union["results"]}
        # In a 4-doc corpus with n_results=5, both should return all 4.
        # The invariant is: union should not lose anything vector found.
        missing = vec_ids - union_ids
        assert not missing, f"union dropped docs that vector found: {missing}"

    def test_union_handles_empty_palace(self, tmp_path):
        """No drawers — union mode should return empty results, not crash."""
        palace = str(tmp_path / "palace")
        get_collection(palace, create=True)  # create empty collection
        result = search_memories("anything", palace, n_results=5, candidate_strategy="union")
        assert result.get("results", []) == []

    def test_invalid_candidate_strategy_raises(self, tmp_path):
        """Bad arg should raise rather than silently fall back."""
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        import pytest

        with pytest.raises(ValueError, match="candidate_strategy"):
            search_memories("anything", palace, n_results=5, candidate_strategy="bogus")

    def test_invalid_strategy_raises_even_when_vector_disabled(self, tmp_path):
        """Validation must happen before the ``vector_disabled`` early return —
        invalid values must fail consistently regardless of routing."""
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        import pytest

        with pytest.raises(ValueError, match="candidate_strategy"):
            search_memories(
                "anything",
                palace,
                n_results=5,
                vector_disabled=True,
                candidate_strategy="bogus",
            )

    def test_union_respects_n_results_limit(self, tmp_path):
        """When the merged candidate set is larger than ``n_results``, the
        result must be trimmed back to the requested size — the MCP
        ``limit`` contract depends on this invariant."""
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        # 4-doc corpus, n_results=2 → union pool can grow to ~8 candidates,
        # rerank reorders them, but final list must respect the cap.
        result = search_memories(_NARRATIVE_QUERY, palace, n_results=2, candidate_strategy="union")
        assert (
            len(result["results"]) <= 2
        ), f"union must trim to n_results=2; got {len(result['results'])} results"

    def test_union_lexical_channel_survives_max_distance(self, tmp_path):
        """``max_distance`` bounds the *vector* channel only. BM25-only
        candidates carry ``distance=None`` — their relevance evidence is
        the literal term match itself — so a strict vector threshold must
        not silence the lexical channel. (Regression: with the serve
        default cutoff of 0.50, name queries like "Inoue" returned zero
        results because union mode dropped all BM25 candidates.)"""
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        # Sanity: without max_distance, union surfaces the BM25-strong doc.
        unfiltered = search_memories(
            _NARRATIVE_QUERY, palace, n_results=5, candidate_strategy="union"
        )
        assert "brand_voice_D4.md" in {h["source_file"] for h in unfiltered["results"]}

        # With a tight max_distance the BM25-strong doc must STILL be
        # reachable, while every hit that does carry a vector distance
        # keeps honoring the threshold.
        filtered = search_memories(
            _NARRATIVE_QUERY,
            palace,
            n_results=5,
            candidate_strategy="union",
            max_distance=0.5,
        )
        sources = {h["source_file"] for h in filtered["results"]}
        assert "brand_voice_D4.md" in sources, (
            "max_distance must bound the vector channel only; the lexical "
            f"(BM25) channel must survive it — got {sources}"
        )
        for h in filtered["results"]:
            if h.get("distance") is not None:
                assert (
                    h["distance"] <= 0.5
                ), f"vector hit violates max_distance=0.5: distance={h['distance']}"

    def test_union_merger_gates_zero_bm25_candidates(self, monkeypatch):
        """Union must inject only candidates with actual term-match
        evidence (``bm25_score > 0``). Recency-fallback rows — produced
        when no query token survives the trigram tokenizer — score 0.0
        and must never pad the pool."""
        from mempalace import searcher

        hits = [
            {
                "text": "vector hit",
                "distance": 0.3,
                "source_file": "a.md",
                "_source_file_full": "docs/a.md",
                "_chunk_index": 0,
            }
        ]
        fake = {
            "results": [
                {
                    "text": "recency junk — no query term at all",
                    "bm25_score": 0.0,
                    "source_file": "b.md",
                    "_source_file_full": "docs/b.md",
                    "_chunk_index": 0,
                },
                {
                    "text": "real lexical match",
                    "bm25_score": 1.234,
                    "source_file": "c.md",
                    "_source_file_full": "docs/c.md",
                    "_chunk_index": 0,
                },
            ]
        }
        monkeypatch.setattr(
            searcher, "_bm25_only_via_sqlite", lambda *a, **k: fake
        )
        searcher._merge_bm25_union_candidates(
            hits, "query", "/nonexistent", None, None, 5, max_distance=0.5
        )
        texts = [h["text"] for h in hits]
        assert "real lexical match" in texts, "positive-BM25 candidate must be injected"
        assert not any("junk" in t for t in texts), "zero-BM25 candidate must be gated out"

    def test_union_injects_drawers_with_empty_source_file(self, monkeypatch):
        """KB-filed drawers can carry an empty ``source_file``, which
        collapses the fallback dedup key to ``"?"``. They must still be
        injected (deduped by text), not silently discarded — discarding
        them dropped all 164 LaPy12 drawers from 'whirls and curls'
        (2026-08-13)."""
        from mempalace import searcher

        hits = [
            {
                "text": "vector hit",
                "distance": 0.3,
                "source_file": "a.md",
                "_source_file_full": "docs/a.md",
                "_chunk_index": 0,
            }
        ]
        kb = {
            "text": "# LaPy12:Thm 5.7 — whirls and curls",
            "bm25_score": 0.145,
            "source_file": "?",
            "_source_file_full": "",
            "_chunk_index": 0,
        }
        dup_of_vector = {
            "text": "vector hit",  # same text as an existing hit — must dedup
            "bm25_score": 0.5,
            "source_file": "?",
            "_source_file_full": "",
            "_chunk_index": 0,
        }
        fake = {"results": [kb, dup_of_vector]}
        monkeypatch.setattr(
            searcher, "_bm25_only_via_sqlite", lambda *a, **k: fake
        )
        searcher._merge_bm25_union_candidates(
            hits, "whirls and curls", "/nonexistent", None, None, 5, max_distance=0.5
        )
        texts = [h["text"] for h in hits]
        assert "# LaPy12:Thm 5.7 — whirls and curls" in texts, (
            "empty-source_file drawer must be injected via text dedup"
        )
        assert texts.count("vector hit") == 1, "text dedup must drop the duplicate"

    def test_union_dedup_is_chunk_precise_not_basename(self, tmp_path):
        """Two files with the same basename in different directories must
        not collide — union must dedup on full path (or chunk-level key),
        not on basename alone. Otherwise a BM25-strong README from one
        directory silently shadows a BM25-strong README from another.
        """
        palace = str(tmp_path / "palace")
        col = get_collection(palace, create=True)
        col.upsert(
            ids=["A_README", "B_README", "narrative"],
            documents=[
                # Both README files share the basename README.md but live
                # in different directories. Each contains distinctive
                # terminology a query might surface via BM25.
                "PROJECT ALPHA: configuration for the Frobnitz subsystem. "
                "Set FROBNITZ_TIMEOUT=30 to enable widget rotation.",
                "PROJECT BETA: configuration for the Wibble subsystem. "
                "Set WIBBLE_THRESHOLD=0.5 to enable signal smoothing.",
                "Engineers occasionally chat about how the legacy "
                "subsystems all need their config knobs tweaked.",
            ],
            metadatas=[
                {"wing": "code", "room": "docs", "source_file": "alpha/README.md"},
                {"wing": "code", "room": "docs", "source_file": "beta/README.md"},
                {"wing": "code", "room": "docs", "source_file": "chat.md"},
            ],
        )
        # Query that hits BM25 for BOTH READMEs (distinct vocab from each).
        # Vector-only might pick the chat doc as semantically "closest";
        # union must surface both READMEs without basename collision.
        result = search_memories(
            "FROBNITZ_TIMEOUT WIBBLE_THRESHOLD configuration",
            palace,
            n_results=5,
            candidate_strategy="union",
        )
        sources = [h["source_file"] for h in result["results"]]
        readme_count = sum(1 for s in sources if s == "README.md")
        assert readme_count >= 2, (
            f"union must surface both README.md files from different dirs "
            f"(basename collision would drop one); got sources={sources}"
        )


class TestTrigramFloodResistance:
    """Trigram FTS matches substrings, so a stopword fragment ("and")
    matches nearly every document. Candidate selection must be AND-first
    so the LIMIT-bounded window is never flooded by a common fragment —
    the "whirls and curls" regression (2026-08-13)."""

    def test_stopword_fragment_does_not_evict_phrase_doc(self, tmp_path):
        from mempalace.searcher import _bm25_only_via_sqlite

        palace = str(tmp_path / "palace")
        col = get_collection(palace, create=True)
        # 6 flood docs (low rowids, all contain "and") seeded BEFORE the
        # one document that actually contains both rare terms.
        ids, docs, metas = [], [], []
        for i in range(6):
            ids.append(f"flood{i}")
            docs.append(f"Filler number {i}: sand bandage candidate operand and more.")
            metas.append({"wing": "w", "room": "r", "source_file": f"flood{i}.md"})
        ids.append("target")
        docs.append("Whirls and curls generate the totally nonnegative loop group.")
        metas.append({"wing": "w", "room": "r", "source_file": "target.md"})
        col.upsert(ids=ids, documents=docs, metadatas=metas)

        # max_candidates=3: an OR-joined MATCH fills the window with flood
        # rows (lower rowids) and the target never enters the pool.
        res = _bm25_only_via_sqlite(
            "whirls and curls", palace, n_results=3, max_candidates=3
        )
        sources = [h["source_file"] for h in res["results"]]
        assert "target.md" in sources, (
            f"AND-first candidate selection must survive the 'and' flood; got {sources}"
        )


class TestHybridRankTolerantOfMissingDistance:
    """``_hybrid_rank`` accepts ``distance=None`` — required for BM25-only
    candidates injected by union mode."""

    def test_distance_none_scored_as_zero_vector_sim(self):
        from mempalace.searcher import _hybrid_rank

        results = [
            {"text": "alpha beta gamma", "distance": 0.2},  # close vector match
            {"text": "alpha alpha alpha", "distance": None},  # BM25-only — heavy term repetition
        ]
        # Query matches "alpha" heavily; the BM25-only candidate with no
        # vector signal should still rank competitively on BM25 alone.
        ranked = _hybrid_rank(results, "alpha")
        assert all("bm25_score" in r for r in ranked), "rerank should add bm25_score"
        # Both must survive — neither should crash on distance=None.
        assert len(ranked) == 2
