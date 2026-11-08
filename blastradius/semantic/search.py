# Copyright 2026 David Scheiderman
# Licensed under the Apache License, Version 2.0
"""Hybrid query engine: semantic KNN + FTS5 keyword + graph expansion, fused with RRF."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from blastradius.semantic.provider import EmbeddingProvider
    from blastradius.store.db import Store

_RRF_K = 60  # standard RRF constant

import re as _re


def _query_stems(query: str) -> list[str]:
    """Return lowercase 4-char stems of each word in the query."""
    return [
        w[:4].lower() for w in _re.sub(r"[^\w\s]", " ", query).split() if len(w) >= 3
    ]


def _path_boost(file_path: str, stems: list[str]) -> float:
    """1.5x if the file path contains any query stem; 1.0 otherwise."""
    path_lower = file_path.lower()
    return 1.5 if any(s in path_lower for s in stems) else 1.0


def _rrf_fuse(ranked_lists: list[list[int]]) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion over lists of symbol IDs.

    score(d) = Σ 1 / (RRF_K + rank_i) across all lists that contain d.
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, sid in enumerate(ranked, start=1):
            scores[sid] = scores.get(sid, 0.0) + 1.0 / (_RRF_K + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _semantic_candidates(
    store: Store,
    query: str,
    limit: int,
    provider: EmbeddingProvider | None,
) -> list[int]:
    """Retrieve KNN hits when configured; warn and fall back on retrieval failure."""
    if provider is not None and store._has_vec:
        try:
            vec = provider.embed([query])[0]
            return store.semantic_search(vec, limit)
        except Exception as exc:  # noqa: BLE001
            print(f"[blastradius] semantic search skipped: {exc}", file=sys.stderr)
    elif provider is not None and not store._has_vec:
        print(
            "[blastradius] sqlite-vec not available — falling back to FTS + graph search.",
            file=sys.stderr,
        )
    return []


def _keyword_candidates(store: Store, query: str, limit: int) -> list[int]:
    """Retrieve keyword hits; warn and fall back on retrieval failure."""
    try:
        return store.fts_search(query, limit)
    except Exception as exc:  # noqa: BLE001
        print(f"[blastradius] FTS search skipped: {exc}", file=sys.stderr)
        return []


def _collect_candidates(
    store: Store, query: str, k: int, provider: EmbeddingProvider | None
) -> tuple[list[list[int]], dict[int, list[str]]]:
    """Collect rankings and provenance in semantic, keyword, then graph order.

    Graph expansion supplements sparse primary signals, counting overlapping hits
    separately and using at most five seeds per primary signal.
    """
    ranked_lists: list[list[int]] = []
    provenance: dict[int, list[str]] = {}
    sem_ids = _semantic_candidates(store, query, k * 2, provider)
    fts_ids = _keyword_candidates(store, query, k * 2)
    for signal, ids in (("semantic", sem_ids), ("fts", fts_ids)):
        for sid in ids:
            provenance.setdefault(sid, []).append(signal)
        if ids:
            ranked_lists.append(ids)

    # Skip if FTS (or semantic) already returned enough results — graph neighbours
    # add structural noise that dilutes good keyword/semantic hits.
    primary_hit_count = sum(len(lst) for lst in ranked_lists)
    if ranked_lists and primary_hit_count < k:
        top_ids: list[int] = []
        for lst in ranked_lists:
            top_ids.extend(lst[:5])
        # deduplicate while preserving order
        seen: set[int] = set()
        deduped: list[int] = []
        for sid in top_ids:
            if sid not in seen:
                seen.add(sid)
                deduped.append(sid)
        graph_ids = store.graph_expand(deduped, k - primary_hit_count)
        for sid in graph_ids:
            if sid not in seen:
                provenance.setdefault(sid, []).append("graph")
        if graph_ids:
            ranked_lists.append(graph_ids)
    return ranked_lists, provenance


def _hydrate_and_rerank(
    store: Store,
    query: str,
    k: int,
    fused: list[tuple[int, float]],
    provenance: dict[int, list[str]],
) -> list[dict]:
    """Hydrate up to 3k visible hits, boost paths/exports, then keep the best k.

    Missing symbols are skipped without replenishment. Sorting preserves fused
    order for ties in the rounded, boosted scores.
    """
    stems = _query_stems(query)
    hydrated: list[tuple[float, dict]] = []
    for sid, score in fused[: k * 3]:  # oversample so rerank can promote better hits
        sym = store.get_symbol(sid)
        if sym is None:
            continue
        boost = _path_boost(sym["file"], stems)
        if sym.get("exported"):
            boost *= 1.2
        sym["rrf_score"] = round(score * boost, 6)
        sym["signals"] = provenance.get(sid, [])
        hydrated.append((sym["rrf_score"], sym))

    hydrated.sort(key=lambda x: x[0], reverse=True)
    return [sym for _, sym in hydrated[:k]]


def hybrid_search(
    store: Store,
    query: str,
    k: int = 10,
    as_of_reachable: set | None = None,
    provider: EmbeddingProvider | None = None,
) -> list[dict]:
    """Fuse semantic, FTS, and graph signals with Reciprocal Rank Fusion.

    Returns up to *k* result dicts, each with symbol metadata, rrf_score, and signals.
    Degrades gracefully: if provider is None or sqlite-vec absent, uses FTS + graph only.
    If as_of_reachable is provided, restricts to symbols visible at that historical point.
    """
    ranked_lists, provenance = _collect_candidates(store, query, k, provider)
    if not ranked_lists:
        return []

    fused = _rrf_fuse(ranked_lists)

    # Filter before hydration's oversampling limit, without fetching replacements.
    if as_of_reachable is not None:
        fused = [
            (sid, score)
            for sid, score in fused
            if store.symbol_visible_at(sid, as_of_reachable)
        ]

    return _hydrate_and_rerank(store, query, k, fused, provenance)
