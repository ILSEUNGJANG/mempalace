#!/usr/bin/env python3
"""
searcher.py — Find anything. Exact words.

Hybrid search: BM25 keyword matching + vector semantic similarity. The
drawer query is the floor — always runs — and closet hits add a rank-based
boost when they agree. Closets are a ranking *signal*, never a gate, so
weak closets (regex extraction on narrative content) can only help, never
hide drawers the direct path would have found.
"""

import logging
import math
import os
import re
import sqlite3
from pathlib import Path

from .palace import get_closets_collection, get_collection

# Closet pointer line format: "topic|entities|→drawer_id_a,drawer_id_b"
# Multiple lines may join with newlines inside one closet document.
_CLOSET_DRAWER_REF_RE = re.compile(r"→([\w,]+)")

logger = logging.getLogger("mempalace_mcp")
_MAX_ROOM_SHARDS_FOR_WING_VECTOR_FALLBACK = 64


class SearchError(Exception):
    """Raised when search cannot proceed (e.g. no palace found)."""


_TOKEN_RE = re.compile(r"\w{2,}", re.UNICODE)


def _first_or_empty(results, key: str) -> list:
    """Return the first inner list of a query result field, or [].

    Accepts both the typed :class:`QueryResult` (attribute access) and the
    pre-typed chroma dict shape; this polymorphism is retained so test mocks
    still work and callers mid-migration do not crash. Preserves the empty-
    collection semantics from issue #195: when no queries returned hits, the
    outer list may be empty and indexing ``[0]`` would raise.
    """
    outer = getattr(results, key, None) if not isinstance(results, dict) else results.get(key)
    if not outer:
        return []
    return outer[0] or []


def _tokenize(text: str) -> list:
    """Lowercase + strip to alphanumeric tokens of length ≥ 2.

    Tolerates ``None`` documents — Chroma can return ``None`` in the
    ``documents`` field for drawers without text content, which would
    otherwise raise ``AttributeError`` mid-rerank.
    """
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def _bm25_scores(
    query: str,
    documents: list,
    k1: float = 1.5,
    b: float = 0.75,
) -> list:
    """Compute Okapi-BM25 scores for ``query`` against each document.

    IDF is computed over the *provided corpus* using the Lucene/BM25+
    smoothed formula ``log((N - df + 0.5) / (df + 0.5) + 1)``, which is
    always non-negative. This is well-defined for re-ranking a small
    candidate set returned by vector retrieval — IDF then reflects how
    discriminative each query term is *within the candidates*, exactly
    what's needed to reorder them.

    Parameters mirror Okapi-BM25 conventions:
        k1 — term-frequency saturation (1.2-2.0 typical, 1.5 default)
        b  — length normalization (0.0 = none, 1.0 = full, 0.75 default)

    Returns a list of scores in the same order as ``documents``.
    """
    n_docs = len(documents)
    query_terms = set(_tokenize(query))
    if not query_terms or n_docs == 0:
        return [0.0] * n_docs

    tokenized = [_tokenize(d) for d in documents]
    doc_lens = [len(toks) for toks in tokenized]
    if not any(doc_lens):
        return [0.0] * n_docs
    avgdl = sum(doc_lens) / n_docs or 1.0

    # Document frequency: how many docs contain each query term?
    df = {term: 0 for term in query_terms}
    for toks in tokenized:
        seen = set(toks) & query_terms
        for term in seen:
            df[term] += 1

    idf = {term: math.log((n_docs - df[term] + 0.5) / (df[term] + 0.5) + 1) for term in query_terms}

    scores = []
    for toks, dl in zip(tokenized, doc_lens):
        if dl == 0:
            scores.append(0.0)
            continue
        tf: dict = {}
        for t in toks:
            if t in query_terms:
                tf[t] = tf.get(t, 0) + 1
        score = 0.0
        for term, freq in tf.items():
            num = freq * (k1 + 1)
            den = freq + k1 * (1 - b + b * dl / avgdl)
            score += idf[term] * num / den
        scores.append(score)
    return scores


def _hybrid_rank(
    results: list,
    query: str,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> list:
    """Re-rank ``results`` by a convex combination of vector similarity and BM25.

    * Vector similarity uses absolute cosine sim ``max(0, 1 - distance)`` —
      ChromaDB's hnsw cosine distance lives in ``[0, 2]`` (0 = identical).
      Absolute (not relative-to-max) means adding/removing a candidate
      can't reshuffle the others.
    * BM25 is real Okapi-BM25 with corpus-relative IDF over the candidates
      themselves. Since the absolute scale is unbounded, BM25 is min-max
      normalized within the candidate set so weights are commensurable.

    Candidates with ``distance=None`` are treated as vector-unknown
    (no vector signal available) and scored on BM25 contribution alone.
    Used by candidate-union mode to merge BM25-only candidates that the
    vector index didn't surface.

    Mutates each result dict to add ``bm25_score`` and reorders the list
    in place. Returns the same list for convenience.
    """
    if not results:
        return results

    docs = [r.get("text", "") for r in results]
    bm25_raw = _bm25_scores(query, docs)
    max_bm25 = max(bm25_raw) if bm25_raw else 0.0
    bm25_norm = [s / max_bm25 for s in bm25_raw] if max_bm25 > 0 else [0.0] * len(bm25_raw)

    scored = []
    for r, raw, norm in zip(results, bm25_raw, bm25_norm):
        distance = r.get("distance")
        if distance is None:
            vec_sim = 0.0
        else:
            vec_sim = max(0.0, 1.0 - distance)
        r["bm25_score"] = round(raw, 3)
        scored.append((vector_weight * vec_sim + bm25_weight * norm, r))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    results[:] = [r for _, r in scored]
    return results


def build_where_filter(wing: str = None, room: str = None) -> dict:
    """Build ChromaDB where filter for wing/room filtering."""
    if wing and room:
        return {"$and": [{"wing": wing}, {"room": room}]}
    elif wing:
        return {"wing": wing}
    elif room:
        return {"room": room}
    return {}


def _is_chroma_id_lookup_error(exc: Exception) -> bool:
    """Return True for Chroma's transient/stale HNSW id lookup failure."""
    msg = str(exc)
    return "Error finding id" in msg or "Internal error" in msg


def _rooms_for_wing_via_sqlite(
    palace_path: str,
    wing: str,
    collection_name: str = None,
    max_rooms: int = _MAX_ROOM_SHARDS_FOR_WING_VECTOR_FALLBACK,
) -> list | None:
    """Return distinct rooms for ``wing`` from Chroma's SQLite metadata.

    Chroma vector query can fail on broad ``where={"wing": ...}`` filters
    while narrower ``wing+room`` filters still work. Room-sharding preserves
    vector search quality for textbook-style wings with a small room count.
    Returning ``None`` tells callers to use the SQLite BM25 fallback instead
    of issuing an unbounded number of vector queries for very large wings.
    """
    db_path = os.path.join(palace_path, "chroma.sqlite3")
    if not wing or not os.path.isfile(db_path):
        return None
    if collection_name is None:
        from .config import get_configured_collection_name

        collection_name = get_configured_collection_name()

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                """
                SELECT DISTINCT room_m.string_value
                FROM embedding_metadata wing_m
                JOIN embedding_metadata room_m
                  ON room_m.id = wing_m.id AND room_m.key = 'room'
                JOIN embeddings e ON e.id = wing_m.id
                JOIN segments s ON e.segment_id = s.id
                JOIN collections c ON s.collection = c.id
                WHERE wing_m.key = 'wing'
                  AND wing_m.string_value = ?
                  AND c.name = ?
                  AND room_m.string_value IS NOT NULL
                ORDER BY room_m.string_value
                LIMIT ?
                """,
                (wing, collection_name, max_rooms + 1),
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        logger.debug("room discovery for wing=%s failed", wing, exc_info=True)
        return None

    rooms = [row[0] for row in rows if row[0]]
    if len(rooms) > max_rooms:
        logger.debug(
            "wing=%s has more than %d rooms; using BM25 fallback instead of room shards",
            wing,
            max_rooms,
        )
        return None
    return rooms


def _query_wing_by_room_shards(
    drawers_col,
    query: str,
    palace_path: str,
    wing: str,
    n_results: int,
    collection_name: str = None,
) -> dict | None:
    """Run vector search for a wing by querying each room separately.

    This is a targeted recovery path for Chroma's broad wing-filter id lookup
    failures. If room discovery or any shard query fails, return ``None`` so
    the caller can fall back to the SQLite BM25 path with complete scoped
    coverage rather than returning partial vector results.
    """
    rooms = _rooms_for_wing_via_sqlite(palace_path, wing, collection_name=collection_name)
    if rooms is None:
        return None

    docs: list = []
    metas: list = []
    dists: list = []
    failed_rooms: list = []
    for room_name in rooms:
        try:
            shard_results = drawers_col.query(
                query_texts=[query],
                n_results=n_results * 3,
                include=["documents", "metadatas", "distances"],
                where={"$and": [{"wing": wing}, {"room": room_name}]},
            )
        except Exception:
            logger.debug(
                "room-sharded vector search failed for wing=%s room=%s",
                wing,
                room_name,
                exc_info=True,
            )
            failed_rooms.append(room_name)
            continue
        docs.extend(_first_or_empty(shard_results, "documents"))
        metas.extend(_first_or_empty(shard_results, "metadatas"))
        dists.extend(_first_or_empty(shard_results, "distances"))

    if failed_rooms and not docs:
        return None

    return {
        "documents": [docs],
        "metadatas": [metas],
        "distances": [dists],
        "_failed_rooms": failed_rooms,
    }


def _extract_drawer_ids_from_closet(closet_doc: str) -> list:
    """Parse all `→drawer_id_a,drawer_id_b` pointers out of a closet document.

    Preserves order and dedupes.
    """
    seen: dict = {}
    for match in _CLOSET_DRAWER_REF_RE.findall(closet_doc):
        for did in match.split(","):
            did = did.strip()
            if did and did not in seen:
                seen[did] = None
    return list(seen.keys())


def _expand_with_neighbors(drawers_col, matched_doc: str, matched_meta: dict, radius: int = 1):
    """Expand a matched drawer with its ±radius sibling chunks in the same source file.

    Motivation — "drawer-grep context" feature: a closet hit returns one
    drawer, but the chunk boundary may clip mid-thought (e.g., the matched
    chunk says "here's a breakdown:" and the actual breakdown lives in the
    next chunk). Fetching the small neighborhood around the match gives
    callers enough context without forcing a follow-up ``get_drawer`` call.

    Returns a dict with:
        ``text``            combined chunks in chunk_index order
        ``drawer_index``    the matched chunk's index in the source file
        ``total_drawers``   total drawer count for the source file (or None)

    On any ChromaDB failure or missing metadata, falls back to returning the
    matched drawer alone so search never breaks because neighbor expansion
    failed.
    """
    src = matched_meta.get("source_file")
    chunk_idx = matched_meta.get("chunk_index")
    if not src or not isinstance(chunk_idx, int):
        return {"text": matched_doc, "drawer_index": chunk_idx, "total_drawers": None}

    target_indexes = [chunk_idx + offset for offset in range(-radius, radius + 1)]
    try:
        neighbors = drawers_col.get(
            where={
                "$and": [
                    {"source_file": src},
                    {"chunk_index": {"$in": target_indexes}},
                ]
            },
            include=["documents", "metadatas"],
        )
    except Exception:
        return {"text": matched_doc, "drawer_index": chunk_idx, "total_drawers": None}

    indexed_docs = []
    for doc, meta in zip(neighbors.documents, neighbors.metadatas):
        ci = meta.get("chunk_index")
        if isinstance(ci, int):
            indexed_docs.append((ci, doc))
    indexed_docs.sort(key=lambda pair: pair[0])

    if not indexed_docs:
        combined_text = matched_doc
    else:
        combined_text = "\n\n".join(doc for _, doc in indexed_docs)

    # Cheap total_drawers lookup: metadata-only scan of the source file.
    total_drawers = None
    try:
        all_meta = drawers_col.get(where={"source_file": src}, include=["metadatas"])
        total_drawers = len(all_meta.ids) if all_meta.ids else None
    except Exception:
        logger.debug("total_drawers lookup failed for %s", src, exc_info=True)

    return {
        "text": combined_text,
        "drawer_index": chunk_idx,
        "total_drawers": total_drawers,
    }


def _warn_if_legacy_metric(col) -> None:
    """Print a one-line notice if the palace was created without
    ``hnsw:space=cosine``.

    ChromaDB's default is L2 (Euclidean), under which cosine-based
    similarity interpretation falls apart — distances routinely exceed
    1.0 and the display ``max(0, 1 - dist)`` floors every result to 0.
    Legacy palaces (mined before this metadata was consistently set)
    need ``mempalace repair`` to rebuild with the correct metric.

    The warning fires only for palaces that clearly have the wrong
    metric; palaces with no metadata table at all (empty dict) also
    fall under this check since that is the signal of a pre-metadata
    palace.
    """
    try:
        meta = getattr(col, "metadata", None)
    except Exception:
        return
    if not isinstance(meta, dict):
        return
    space = meta.get("hnsw:space")
    if space == "cosine":
        return
    # Either missing or set to something else — both are suspect.
    import sys as _sys

    detail = f"hnsw:space={space!r}" if space else "no hnsw:space metadata"
    print(
        f"\n  NOTICE: this palace was created without cosine distance ({detail}).\n"
        "          Semantic similarity scores will not be meaningful.\n"
        "          Run `mempalace repair` to rebuild the index with the correct metric.",
        file=_sys.stderr,
    )


def search(query: str, palace_path: str, wing: str = None, room: str = None, n_results: int = 5):
    """
    Search the palace. Returns verbatim drawer content.
    Optionally filter by wing (project) or room (aspect).
    """
    try:
        col = get_collection(palace_path, create=False)
    except Exception as e:
        print(f"\n  No palace found at {palace_path}")
        print("  Run: mempalace init <dir> then mempalace mine <dir>")
        raise SearchError(f"No palace found at {palace_path}") from e

    # Alert the user if this palace predates hnsw:space=cosine being set on
    # creation — their similarity scores will be junk until they run repair.
    _warn_if_legacy_metric(col)

    where = build_where_filter(wing, room)

    try:
        kwargs = {
            "query_texts": [query],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = col.query(**kwargs)

    except Exception as e:
        print(f"\n  Search error: {e}")
        raise SearchError(f"Search error: {e}") from e

    docs = _first_or_empty(results, "documents")
    metas = _first_or_empty(results, "metadatas")
    dists = _first_or_empty(results, "distances")

    if not docs:
        print(f'\n  No results found for: "{query}"')
        return

    # Pure-cosine retrieval on the CLI path was missing lexical matches:
    # a drawer whose text contains every query term can still score distance
    # >= 1.0 against the natural-language query when the drawer is a
    # mechanical artifact (directory listing, diff, log fragment) that
    # embeds as file-tree noise rather than as prose about its subject.
    # The MCP tool path already hybridizes BM25 with vector sim via
    # `_hybrid_rank`; do the same here so CLI results match what agents
    # see via `mempalace_search`.
    hits = [
        {"text": doc or "", "distance": float(dist), "metadata": meta or {}}
        for doc, meta, dist in zip(docs, metas, dists)
    ]
    hits = _hybrid_rank(hits, query)

    print(f"\n{'=' * 60}")
    print(f'  Results for: "{query}"')
    if wing:
        print(f"  Wing: {wing}")
    if room:
        print(f"  Room: {room}")
    print(f"{'=' * 60}\n")

    for i, hit in enumerate(hits, 1):
        vec_sim = round(max(0.0, 1 - hit["distance"]), 3)
        bm25 = hit.get("bm25_score", 0.0)
        meta = hit["metadata"]
        source = Path(meta.get("source_file", "?")).name
        wing_name = meta.get("wing", "?")
        room_name = meta.get("room", "?")

        print(f"  [{i}] {wing_name} / {room_name}")
        print(f"      Source: {source}")
        print(f"      Match:  cosine={vec_sim}  bm25={bm25}")
        print()
        # Print the verbatim text, indented
        for line in hit["text"].strip().split("\n"):
            print(f"      {line}")
        print()
        print(f"  {'─' * 56}")

    print()


def _bm25_only_via_sqlite(
    query: str,
    palace_path: str,
    wing: str = None,
    room: str = None,
    n_results: int = 5,
    max_candidates: int = 500,
    _include_internal: bool = False,
    collection_name: str = None,
) -> dict:
    """BM25-only search reading drawers directly from chroma.sqlite3.

    Used when HNSW is diverged or unloadable (#1222). Bypasses chromadb's
    Python client entirely so a corrupt vector segment can't segfault the
    MCP server. Routes through chromadb's own FTS5 trigram index
    (``embedding_fulltext_search``) for candidate selection, then re-ranks
    with the same Okapi-BM25 used in :func:`_hybrid_rank` so the result
    shape matches the vector path.

    The query is split into ≥3-char trigram-tokens — chromadb writes the
    index with ``tokenize='trigram'``, so shorter tokens never match and
    every token matches as a SUBSTRING. Candidate selection is AND-first
    (all tokens required; common fragments like "and" then only narrow),
    widening per-token from the longest token down when AND is too strict.
    When no usable token survives (e.g. "is a"), candidate selection falls
    back to the most-recent ``max_candidates`` rows so we still return
    *something* rather than nothing.
    """
    db_path = os.path.join(palace_path, "chroma.sqlite3")
    if not os.path.isfile(db_path):
        return {
            "error": "No palace found",
            "hint": "Run: mempalace init <dir> && mempalace mine <dir>",
        }
    if collection_name is None:
        from .config import get_configured_collection_name

        collection_name = get_configured_collection_name()

    def _metadata_filter_sql(row_id_expr: str) -> tuple[str, list[str]]:
        clauses = []
        params = []
        for key, value in (("wing", wing), ("room", room)):
            if not value:
                continue
            clauses.append(
                f"""
                AND EXISTS (
                    SELECT 1
                    FROM embedding_metadata mf
                    WHERE mf.id = {row_id_expr}
                      AND mf.key = ?
                      AND COALESCE(
                        mf.string_value,
                        CAST(mf.int_value AS TEXT),
                        CAST(mf.float_value AS TEXT),
                        CAST(mf.bool_value AS TEXT)
                      ) = ?
                )
                """
            )
            params.extend([key, value])
        return "".join(clauses), params

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error as e:
        return {"error": f"sqlite open failed: {e}"}

    try:
        # FTS5 MATCH expects whitespace-separated tokens. Drop tokens
        # shorter than 3 chars (trigram tokenizer can't match them).
        tokens = [t for t in _tokenize(query) if len(t) >= 3]
        candidate_ids: list[int] = []
        use_recency_fallback = not tokens
        if tokens:
            filter_sql, filter_params = _metadata_filter_sql("embedding_fulltext_search.rowid")

            def _fts_candidates(match_expr: str, limit: int) -> list[int]:
                rows = conn.execute(
                    f"""
                    SELECT embedding_fulltext_search.rowid
                    FROM embedding_fulltext_search
                    JOIN embeddings e ON e.id = embedding_fulltext_search.rowid
                    JOIN segments s ON e.segment_id = s.id
                    JOIN collections c ON s.collection = c.id
                    WHERE embedding_fulltext_search MATCH ?
                      AND c.name = ?
                    {filter_sql}
                    LIMIT ?
                    """,
                    (match_expr, collection_name, *filter_params, limit),
                ).fetchall()
                return [r[0] for r in rows]

            quoted = ['"' + t.replace('"', '""') + '"' for t in tokens]
            try:
                # AND-first: the trigram tokenizer matches SUBSTRINGS, so an
                # OR over all tokens lets a common fragment ("and", "the")
                # flood the LIMIT-bounded candidate window with unrelated
                # rows selected in arbitrary rowid order — the 2026-08-13
                # "whirls and curls" regression, where the flood evicted
                # every document containing the actual phrase. Requiring
                # all tokens (FTS5 implicit AND) makes stopword fragments
                # harmless: they only ever narrow the result.
                candidate_ids = _fts_candidates(" ".join(quoted), max_candidates)
                if len(candidate_ids) < n_results * 3 and len(tokens) > 1:
                    # AND too strict (e.g. a rare term plus a typo) — widen
                    # per-token, most-selective-first (length as the proxy),
                    # skipping short tokens that would reintroduce the flood.
                    seen = set(candidate_ids)
                    per_token = max(n_results * 3, max_candidates // len(tokens))
                    for tok in sorted(tokens, key=len, reverse=True):
                        if len(tok) < 4 or len(candidate_ids) >= max_candidates:
                            continue
                        for rid in _fts_candidates(
                            '"' + tok.replace('"', '""') + '"', per_token
                        ):
                            if rid not in seen:
                                seen.add(rid)
                                candidate_ids.append(rid)
                            if len(candidate_ids) >= max_candidates:
                                break
            except sqlite3.Error:
                # FTS5 tokenizer mismatch or syntax error — fall through
                # to the recency-window selector below.
                logger.debug("FTS5 MATCH failed; using recency fallback", exc_info=True)
                use_recency_fallback = True

        if not candidate_ids and use_recency_fallback:
            # No usable FTS tokens, or FTS itself failed — pull the most
            # recent rows for the drawers segment so we can BM25-rank
            # something rather than return empty-handed. A clean FTS miss
            # must stay empty, especially after wing/room filtering, because
            # recency fallback would return unrelated scoped drawers.
            # Wrapped in try/except because the schema may differ on legacy
            # palaces (older chromadb without ``created_at``, missing
            # ``segments`` rows after partial restore, etc.); on schema
            # mismatch we fall back to ordering by primary-key id and finally
            # to an empty result rather than letting search raise.
            try:
                filter_sql, filter_params = _metadata_filter_sql("e.id")
                rows = conn.execute(
                    f"""
                    SELECT e.id
                    FROM embeddings e
                    JOIN segments s ON e.segment_id = s.id
                    JOIN collections c ON s.collection = c.id
                    WHERE c.name = ?
                    {filter_sql}
                    ORDER BY e.created_at DESC
                    LIMIT ?
                    """,
                    (collection_name, *filter_params, max_candidates),
                ).fetchall()
                candidate_ids = [r[0] for r in rows]
            except sqlite3.Error:
                logger.debug(
                    "recency-window query failed; trying id-ordered fallback",
                    exc_info=True,
                )
                try:
                    filter_sql, filter_params = _metadata_filter_sql("e.id")
                    rows = conn.execute(
                        f"""
                        SELECT e.id
                        FROM embeddings e
                        JOIN segments s ON e.segment_id = s.id
                        JOIN collections c ON s.collection = c.id
                        WHERE c.name = ?
                        {filter_sql}
                        ORDER BY e.id DESC
                        LIMIT ?
                        """,
                        (collection_name, *filter_params, max_candidates),
                    ).fetchall()
                    candidate_ids = [r[0] for r in rows]
                except sqlite3.Error:
                    logger.debug("id-ordered fallback also failed", exc_info=True)
                    candidate_ids = []

        if not candidate_ids:
            return {
                "query": query,
                "filters": {"wing": wing, "room": room},
                "total_before_filter": 0,
                "results": [],
                "fallback": "bm25_only_via_sqlite",
            }

        placeholders = ",".join(["?"] * len(candidate_ids))
        meta_rows = conn.execute(
            f"""
            SELECT id, key, string_value, int_value
            FROM embedding_metadata
            WHERE id IN ({placeholders})
            """,
            candidate_ids,
        ).fetchall()
    finally:
        conn.close()

    # Group metadata rows into per-drawer dicts.
    drawers: dict[int, dict] = {}
    for emb_id, key, sval, ival in meta_rows:
        d = drawers.setdefault(emb_id, {"_id": emb_id, "metadata": {}, "text": ""})
        if key == "chroma:document":
            d["text"] = sval or ""
        else:
            d["metadata"][key] = sval if sval is not None else ival

    # Apply wing/room filters in Python (FTS5 candidates may include
    # entries from other wings).
    candidates = []
    for d in drawers.values():
        meta = d["metadata"]
        if wing and meta.get("wing") != wing:
            continue
        if room and meta.get("room") != room:
            continue
        full_source = meta.get("source_file", "") or ""
        candidates.append(
            {
                "text": d["text"],
                "wing": meta.get("wing", "unknown"),
                "room": meta.get("room", "unknown"),
                "source_file": Path(full_source).name if full_source else "?",
                "created_at": meta.get("filed_at", "unknown"),
                # No vector distance available in BM25-only mode.
                "similarity": None,
                "distance": None,
                "matched_via": "bm25_sqlite",
                # Internal: full path + chunk_index let callers (notably
                # candidate_strategy="union") dedupe at chunk granularity
                # rather than basename — two files in different directories
                # may share a basename, and one source_file is split across
                # multiple chunks. Stripped before this helper returns.
                "_source_file_full": full_source,
                "_chunk_index": meta.get("chunk_index"),
            }
        )

    # Local BM25 over the candidate set.
    docs = [c["text"] for c in candidates]
    bm25_raw = _bm25_scores(query, docs)
    max_bm25 = max(bm25_raw) if bm25_raw else 0.0
    for c, raw in zip(candidates, bm25_raw):
        c["bm25_score"] = round(raw, 3)
        c["_score"] = (raw / max_bm25) if max_bm25 > 0 else 0.0
    candidates.sort(key=lambda c: c["_score"], reverse=True)
    hits = candidates[:n_results]
    for h in hits:
        h.pop("_score", None)
        # Strip internal fields by default so the public BM25-only fallback
        # response stays clean. Callers that need chunk-precise dedup
        # (notably the union-merge path) opt in via _include_internal.
        if not _include_internal:
            h.pop("_source_file_full", None)
            h.pop("_chunk_index", None)

    return {
        "query": query,
        "filters": {"wing": wing, "room": room},
        "total_before_filter": len(candidates),
        "results": hits,
        "fallback": "bm25_only_via_sqlite",
        "fallback_reason": "vector_search_disabled",
    }


def _merge_bm25_union_candidates(
    hits: list,
    query: str,
    palace_path: str,
    wing: str,
    room: str,
    n_results: int,
    max_distance: float = 0.0,
) -> None:
    """Append top-K BM25-only candidates from sqlite into ``hits`` in place.

    Used by ``search_memories(..., candidate_strategy="union")`` to widen
    the rerank pool's *source* (not just its size) — vector-only candidate
    selection skips docs whose embeddings are far from the query even when
    BM25 signal is strong.

    Dedup is chunk-precise: the key is ``(_source_file_full, _chunk_index)``
    so two files sharing a basename in different directories don't collide,
    and a vector hit on chunk N of a file doesn't block BM25 from
    contributing chunk M of the same file. Falls back to ``source_file``
    only when full-path/chunk metadata is absent.

    BM25-only additions carry ``distance=None`` so ``_hybrid_rank`` scores
    them on BM25 contribution alone.

    ``max_distance`` bounds the *vector* channel only (filtering happens
    in the vector-hit loop upstream). BM25-only candidates carry no vector
    distance — their relevance evidence is the literal term match itself —
    so they are injected regardless of the threshold, gated instead on
    ``bm25_score > 0`` so recency-fallback rows (no term-match evidence,
    score 0.0) never pad the pool. Historical note: until 2026-08-13 this
    helper skipped ALL BM25 candidates whenever ``max_distance > 0.0``,
    which silenced the lexical channel under the serve default cutoff and
    made short/name queries ("Inoue") return nothing.
    """
    try:
        bm25_extra = _bm25_only_via_sqlite(
            query,
            palace_path,
            wing=wing,
            room=room,
            n_results=n_results * 3,
            _include_internal=True,
        ).get("results", [])
    except Exception:
        logger.debug("candidate_strategy=union: BM25 fetch failed", exc_info=True)
        return

    def _dedup_key(entry: dict):
        full = entry.get("_source_file_full")
        ci = entry.get("_chunk_index")
        if full and ci is not None:
            return (full, ci)
        # Fall back to basename only when richer metadata is missing —
        # avoids silently dropping candidates on legacy data while still
        # giving chunk-precise dedup whenever the metadata is present.
        return entry.get("source_file")

    seen = {_dedup_key(h) for h in hits}
    # KB-filed drawers can carry an EMPTY source_file in chroma metadata,
    # collapsing their fallback dedup key to "?". Skipping those (the old
    # behavior) silently discarded every such drawer from the lexical
    # channel — the palace's mathpapers wing lost all 164 LaPy12 hits for
    # "whirls and curls" while Vault chunks (real source_file) survived.
    # Dedup them by document text instead, mirroring
    # ``_merge_bm25_fallback_candidates``.
    seen_texts = {h.get("text") for h in hits}
    for bh in bm25_extra:
        key = _dedup_key(bh)
        if not key or key == "?":
            text = bh.get("text")
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)
        elif key in seen:
            continue
        # Term-match gate: only candidates with real BM25 evidence enter
        # the pool. (bm25_score is rounded to 3 decimals upstream; a match
        # weak enough to round to 0.000 carries no usable signal anyway.)
        if bh.get("bm25_score", 0.0) <= 0.0:
            continue
        bh["distance"] = None
        bh["effective_distance"] = None
        bh["closet_boost"] = 0.0
        hits.append(bh)
        seen.add(key)


def _merge_bm25_fallback_candidates(
    hits: list,
    query: str,
    palace_path: str,
    wing: str,
    room: str,
    n_results: int,
    collection_name: str = None,
) -> None:
    """Append SQLite BM25 candidates after a partial vector-filter failure.

    Unlike candidate_strategy="union", this path deliberately ignores
    ``max_distance`` because Chroma already failed to provide complete vector
    coverage for the requested scope. Results are marked ``bm25_sqlite`` and
    keep ``distance=None``.
    """
    try:
        bm25_extra = _bm25_only_via_sqlite(
            query,
            palace_path,
            wing=wing,
            room=room,
            n_results=n_results * 3,
            collection_name=collection_name,
        ).get("results", [])
    except Exception:
        logger.debug("partial vector fallback: BM25 fetch failed", exc_info=True)
        return

    seen_texts = {h.get("text", "") for h in hits}
    for bh in bm25_extra:
        text = bh.get("text", "")
        if not text or text in seen_texts:
            continue
        bh["effective_distance"] = None
        bh["closet_boost"] = 0.0
        hits.append(bh)
        seen_texts.add(text)


# Strategy dispatch — keeps search_memories' branch count under the
# project's complexity ceiling (C901 max-complexity=25). New strategies
# register here.
_CANDIDATE_MERGERS = {
    "vector": None,  # default no-op
    "union": _merge_bm25_union_candidates,
}


def _validate_candidate_strategy(strategy: str) -> None:
    """Raise ``ValueError`` for unknown strategies.

    Called eagerly at the top of ``search_memories`` so invalid values
    fail consistently regardless of whether the call routes through the
    vector path, the BM25-only fallback, or returns an early error dict.
    """
    if strategy not in _CANDIDATE_MERGERS:
        raise ValueError(
            f"candidate_strategy must be one of {tuple(_CANDIDATE_MERGERS)}, got {strategy!r}"
        )


def _apply_candidate_strategy(
    strategy: str,
    hits: list,
    query: str,
    palace_path: str,
    wing: str,
    room: str,
    n_results: int,
    max_distance: float = 0.0,
) -> None:
    """Dispatch to the registered merger for ``strategy``.

    Strategy validity is assumed (``_validate_candidate_strategy`` runs
    earlier); ``"vector"`` is a no-op.
    """
    merger = _CANDIDATE_MERGERS[strategy]
    if merger is not None:
        merger(hits, query, palace_path, wing, room, n_results, max_distance=max_distance)


def search_memories(
    query: str,
    palace_path: str,
    wing: str = None,
    room: str = None,
    n_results: int = 5,
    max_distance: float = 0.0,
    vector_disabled: bool = False,
    candidate_strategy: str = "vector",
    collection_name: str = None,
) -> dict:
    """Programmatic search — returns a dict instead of printing.

    Used by the MCP server and other callers that need data.

    Args:
        query: Natural language search query.
        palace_path: Path to the ChromaDB palace directory.
        wing: Optional wing filter.
        room: Optional room filter.
        n_results: Max results to return.
        max_distance: Max cosine distance threshold. The palace collection uses
            cosine distance (hnsw:space=cosine) — 0 = identical, 2 = opposite.
            Results with distance > this value are filtered out. A value of
            0.0 disables filtering. Typical useful range: 0.3–1.0.
        vector_disabled: When True, route to the sqlite-only BM25 fallback
            (#1222). Set by the MCP server when the HNSW capacity probe
            detects a divergence that would segfault chromadb on segment
            load.
        candidate_strategy: How candidates for the hybrid re-rank are gathered.

            * ``"vector"`` (default) — preserves historical behavior: top
              ``n_results * 3`` rows from the vector index are the rerank pool.
              Cheap; works well when query and target docs agree in the
              embedding space.
            * ``"union"`` — also pull top ``n_results * 3`` BM25 candidates
              from the sqlite FTS5 index and merge them into the rerank pool
              (deduped by source_file). Catches docs with strong BM25 signal
              that are vector-distant from the query (e.g. terminology guides
              looked up by narrative-shaped queries; policy clauses surfaced
              by scenario descriptions). Adds one sqlite open + FTS5 MATCH
              per query; perf cost is small but unmeasured at corpus scale.
              Opt in until the cost is characterized.

              ``max_distance`` bounds the vector channel only: vector hits
              beyond the threshold are filtered as usual, while BM25-only
              candidates (``distance=None``, ``bm25_score > 0``) are still
              merged — a literal term match is its own relevance evidence.
    """
    # Validate the strategy eagerly so invalid values fail the same way
    # regardless of whether the call routes through the vector path or
    # the BM25-only fallback below.
    _validate_candidate_strategy(candidate_strategy)

    if vector_disabled:
        return _bm25_only_via_sqlite(
            query,
            palace_path,
            wing=wing,
            room=room,
            n_results=n_results,
            collection_name=collection_name,
        )

    try:
        drawers_col = get_collection(palace_path, collection_name=collection_name, create=False)
    except Exception as e:
        logger.error("No palace found at %s: %s", palace_path, e)
        return {
            "error": "No palace found",
            "hint": "Run: mempalace init <dir> && mempalace mine <dir>",
        }

    where = build_where_filter(wing, room)

    # Hybrid retrieval: always query drawers directly (the floor), then use
    # closet hits to boost rankings. Closets are a ranking SIGNAL, never a
    # GATE — direct drawer search is always the baseline.
    #
    # This avoids the "weak-closets regression" where narrative content
    # produces low-signal closets (regex extraction matches few topics)
    # and closet-first routing hides drawers that direct search would find.
    drawer_query_fallback = None
    drawer_query_failed_rooms: list = []
    try:
        dkwargs = {
            "query_texts": [query],
            "n_results": n_results * 3,  # over-fetch for re-ranking
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            dkwargs["where"] = where
        drawer_results = drawers_col.query(**dkwargs)
    except Exception as e:
        if not _is_chroma_id_lookup_error(e):
            return {"error": f"Search error: {e}"}

        if wing and not room:
            drawer_results = _query_wing_by_room_shards(
                drawers_col,
                query,
                palace_path,
                wing,
                n_results,
                collection_name=collection_name,
            )
            if drawer_results is not None:
                drawer_query_fallback = "room_sharded_vector"
                drawer_query_failed_rooms = drawer_results.get("_failed_rooms", [])
            else:
                drawer_results = None
        else:
            drawer_results = None

        if drawer_results is None:
            fallback = _bm25_only_via_sqlite(
                query,
                palace_path,
                wing=wing,
                room=room,
                n_results=n_results,
                collection_name=collection_name,
            )
            if "error" not in fallback:
                fallback["fallback_reason"] = "vector_filter_id_lookup_error"
                fallback["vector_error"] = str(e)
            return fallback

    # Gather closet hits (best-per-source) to build a boost lookup.
    closet_boost_by_source: dict = {}  # source_file -> (rank, closet_dist, preview)
    try:
        closets_col = get_closets_collection(palace_path, create=False)
        ckwargs = {
            "query_texts": [query],
            "n_results": n_results * 2,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            ckwargs["where"] = where
        closet_results = closets_col.query(**ckwargs)
        for rank, (cdoc, cmeta, cdist) in enumerate(
            zip(
                _first_or_empty(closet_results, "documents"),
                _first_or_empty(closet_results, "metadatas"),
                _first_or_empty(closet_results, "distances"),
            )
        ):
            cmeta = cmeta or {}
            source = cmeta.get("source_file", "")
            if source and source not in closet_boost_by_source:
                closet_boost_by_source[source] = (rank, cdist, cdoc[:200])
    except Exception:
        # No closets yet — hybrid degrades to pure drawer search.
        logger.debug("Closet collection unavailable; using drawer-only search", exc_info=True)

    # Rank-based boost. The ordinal signal ("which closet matched best") is
    # more reliable than absolute distance on narrative content, where
    # closet distances cluster in 1.2-1.5 range regardless of match quality.
    CLOSET_RANK_BOOSTS = [0.40, 0.25, 0.15, 0.08, 0.04]
    CLOSET_DISTANCE_CAP = 1.5  # cosine dist > 1.5 = too weak to use as signal

    scored: list = []
    for doc, meta, dist in zip(
        _first_or_empty(drawer_results, "documents"),
        _first_or_empty(drawer_results, "metadatas"),
        _first_or_empty(drawer_results, "distances"),
    ):
        meta = meta or {}
        doc = doc or ""
        # Filter on raw distance before rounding to avoid precision loss.
        if max_distance > 0.0 and dist > max_distance:
            continue

        meta = meta or {}
        source = meta.get("source_file", "") or ""
        boost = 0.0
        matched_via = "drawer"
        closet_preview = None
        if source in closet_boost_by_source:
            c_rank, c_dist, c_preview = closet_boost_by_source[source]
            if c_dist <= CLOSET_DISTANCE_CAP and c_rank < len(CLOSET_RANK_BOOSTS):
                boost = CLOSET_RANK_BOOSTS[c_rank]
                matched_via = "drawer+closet"
                closet_preview = c_preview

        # Clamp to the valid cosine-distance range [0, 2]. When a strong
        # closet boost (up to 0.40) exceeds the raw distance, the subtraction
        # can go negative — which (a) yields ``similarity > 1.0`` downstream
        # and (b) makes the sort key land *below* ordinary positive distances,
        # inverting the ranking so the best hybrid matches sort last.
        effective_dist = max(0.0, min(2.0, dist - boost))
        entry = {
            "text": doc,
            "wing": meta.get("wing", "unknown"),
            "room": meta.get("room", "unknown"),
            "source_file": Path(source).name if source else "?",
            "created_at": meta.get("filed_at", "unknown"),
            "similarity": round(max(0.0, 1 - effective_dist), 3),
            "distance": round(dist, 4),
            "effective_distance": round(effective_dist, 4),
            "closet_boost": round(boost, 3),
            "matched_via": matched_via,
            # Internal: retain the full source_file path + chunk_index so the
            # enrichment step below doesn't have to reverse-lookup via
            # basename-suffix matching (which silently collides when two
            # files share a basename across different directories).
            "_sort_key": effective_dist,
            "_source_file_full": source,
            "_chunk_index": meta.get("chunk_index"),
        }
        if closet_preview:
            entry["closet_preview"] = closet_preview
        scored.append(entry)

    scored.sort(key=lambda h: h["_sort_key"])
    hits = scored[:n_results]

    # Drawer-grep enrichment: for closet-boosted hits whose source has
    # multiple drawers, return the keyword-best chunk + its immediate
    # neighbors instead of just the drawer vector search landed on. The
    # closet said "this source is relevant"; vector may have picked the
    # wrong chunk within it; grep picks the right one.
    MAX_HYDRATION_CHARS = 10000
    for h in hits:
        if h["matched_via"] == "drawer":
            continue
        full_source = h.get("_source_file_full") or ""
        if not full_source:
            continue
        try:
            source_drawers = drawers_col.get(
                where={"source_file": full_source},
                include=["documents", "metadatas"],
            )
        except Exception:
            logger.debug("Neighbor fetch failed for %s", full_source, exc_info=True)
            continue
        docs = source_drawers.documents
        metas_ = source_drawers.metadatas
        if len(docs) <= 1:
            continue

        # Sort by chunk_index so best_idx + neighbors are positional.
        indexed = []
        for idx, (d, m) in enumerate(zip(docs, metas_)):
            ci = m.get("chunk_index", idx) if isinstance(m, dict) else idx
            if not isinstance(ci, int):
                ci = idx
            indexed.append((ci, d))
        indexed.sort(key=lambda p: p[0])
        ordered_docs = [d for _, d in indexed]

        query_terms = set(_tokenize(query))
        best_idx, best_score = 0, -1
        for idx, d in enumerate(ordered_docs):
            d_lower = d.lower()
            s = sum(1 for t in query_terms if t in d_lower)
            if s > best_score:
                best_score, best_idx = s, idx

        start = max(0, best_idx - 1)
        end = min(len(ordered_docs), best_idx + 2)
        expanded = "\n\n".join(ordered_docs[start:end])
        if len(expanded) > MAX_HYDRATION_CHARS:
            expanded = (
                expanded[:MAX_HYDRATION_CHARS]
                + f"\n\n[...truncated. {len(ordered_docs)} total drawers. "
                "Use mempalace_get_drawer for full content.]"
            )
        h["text"] = expanded
        h["drawer_index"] = best_idx
        h["total_drawers"] = len(ordered_docs)

    if drawer_query_failed_rooms:
        _merge_bm25_fallback_candidates(
            hits,
            query,
            palace_path,
            wing,
            room,
            n_results,
            collection_name=collection_name,
        )
        drawer_query_fallback = "room_sharded_vector+bm25_sqlite"

    # Candidate strategy hook: optionally widen the rerank pool's *source*
    # before ranking. Default ("vector") is a no-op; "union" merges top-K
    # BM25 candidates from sqlite. See `_apply_candidate_strategy`.
    # ``max_distance`` is forwarded for signature stability; it bounds the
    # vector channel only (applied in the vector-hit loop above), while
    # union injects BM25-only candidates gated on ``bm25_score > 0``.
    _apply_candidate_strategy(
        candidate_strategy,
        hits,
        query,
        palace_path,
        wing,
        room,
        n_results,
        max_distance=max_distance,
    )

    # BM25 hybrid re-rank within the final candidate set, then trim back
    # to the requested size. Without the trim, ``candidate_strategy="union"``
    # would return up to 4× ``n_results`` (vector hits + BM25 union pool),
    # breaking the existing ``search_memories`` size contract that the MCP
    # ``limit`` parameter is built on.
    hits = _hybrid_rank(hits, query)[:n_results]
    for h in hits:
        h.pop("_sort_key", None)
        h.pop("_source_file_full", None)
        h.pop("_chunk_index", None)

    response = {
        "query": query,
        "filters": {"wing": wing, "room": room},
        "total_before_filter": len(_first_or_empty(drawer_results, "documents")),
        "results": hits,
    }
    if drawer_query_fallback:
        response["fallback"] = drawer_query_fallback
        response["fallback_reason"] = "vector_wing_filter_id_lookup_error"
    if drawer_query_failed_rooms:
        response["vector_failed_rooms"] = drawer_query_failed_rooms
    return response
