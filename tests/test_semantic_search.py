# Copyright 2026 Christopher Wilson
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

import pytest

from blastradius.semantic.search import hybrid_search


class FakeStore:
    """Supply ranked database hits without requiring SQLite or sqlite-vec."""

    def __init__(self, *, semantic=(), fts=(), graph=(), has_vec=True):
        self._has_vec = has_vec
        self.semantic = list(semantic)
        self.fts = list(fts)
        self.graph = list(graph)
        self.symbols = {
            sid: {
                "id": sid,
                "name": f"symbol_{sid}",
                "file": f"src/module_{sid}.py",
                "exported": False,
            }
            for sid in {*semantic, *fts, *graph}
        }
        self.visible = set(self.symbols)
        self.calls = []
        self.fail_at = None
        self.failure = RuntimeError("database failed")

    def _record(self, method, *args):
        self.calls.append((method, *args))
        if self.fail_at == method:
            raise self.failure

    def semantic_search(self, vector, k):
        self._record("semantic", vector, k)
        return self.semantic

    def fts_search(self, query, k):
        self._record("fts", query, k)
        return self.fts

    def graph_expand(self, seeds, k):
        self._record("graph", seeds, k)
        return self.graph

    def symbol_visible_at(self, sid, reachable):
        self._record("visible", sid, reachable)
        return sid in self.visible

    def get_symbol(self, sid):
        self._record("hydrate", sid)
        sym = self.symbols.get(sid)
        return sym.copy() if sym is not None else None


class FakeProvider:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def embed(self, texts):
        self.calls.append(texts)
        if self.error is not None:
            raise self.error
        return [[0.25, 0.75]]


@pytest.mark.parametrize(
    "semantic,fts,expected",
    [
        ([1], [], [(1, 0.016393, ["semantic"])]),
        ([], [1], [(1, 0.016393, ["fts"])]),
        (
            [1, 2],
            [2, 3],
            [
                (2, 0.032522, ["semantic", "fts"]),
                (1, 0.016393, ["semantic"]),
                (3, 0.016129, ["fts"]),
            ],
        ),
        ([], [], []),
    ],
)
def test_primary_signals_preserve_metadata_scores_and_provenance(
    semantic, fts, expected, capsys
):
    store = FakeStore(semantic=semantic, fts=fts)
    provider = FakeProvider()

    results = hybrid_search(store, "query", k=3, provider=provider)

    assert results == [
        {
            **store.symbols[sid],
            "rrf_score": score,
            "signals": signals,
        }
        for sid, score, signals in expected
    ]
    assert provider.calls == [["query"]]
    assert store.calls[:2] == [
        ("semantic", [0.25, 0.75], 6),
        ("fts", "query", 6),
    ]
    assert capsys.readouterr().err == ""
    if not expected:
        assert len(store.calls) == 2  # No graph expansion without primary hits.


def test_without_provider_uses_keyword_and_graph_only(capsys):
    store = FakeStore(semantic=[9], fts=[1], graph=[2])

    results = hybrid_search(store, "query", k=2)

    assert [(r["id"], r["signals"]) for r in results] == [
        (1, ["fts"]),
        (2, ["graph"]),
    ]
    assert store.calls[:2] == [("fts", "query", 4), ("graph", [1], 1)]
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    "failure,warning,expected_signal",
    [
        (
            "vec",
            "[blastradius] sqlite-vec not available — falling back to FTS + graph search.\n",
            "fts",
        ),
        (
            "embed",
            "[blastradius] semantic search skipped: endpoint failed\n",
            "fts",
        ),
        (
            "semantic",
            "[blastradius] semantic search skipped: database failed\n",
            "fts",
        ),
        ("fts", "[blastradius] FTS search skipped: database failed\n", "semantic"),
    ],
)
def test_primary_failure_retains_other_signal(
    failure, warning, expected_signal, capsys
):
    store = FakeStore(semantic=[1], fts=[1], has_vec=failure != "vec")
    store.fail_at = failure
    provider = FakeProvider(
        RuntimeError("endpoint failed") if failure == "embed" else None
    )

    results = hybrid_search(store, "query", k=1, provider=provider)

    assert results == [
        {**store.symbols[1], "rrf_score": 0.016393, "signals": [expected_signal]}
    ]
    assert capsys.readouterr().err == warning
    if failure == "vec":
        assert provider.calls == []
        assert all(call[0] != "semantic" for call in store.calls)


def test_all_primary_failures_return_empty(capsys):
    store = FakeStore()
    store.fail_at = "fts"

    results = hybrid_search(
        store, "query", provider=FakeProvider(RuntimeError("endpoint failed"))
    )

    assert results == []
    assert store.calls == [("fts", "query", 20)]
    assert capsys.readouterr().err == (
        "[blastradius] semantic search skipped: endpoint failed\n"
        "[blastradius] FTS search skipped: database failed\n"
    )


def test_sparse_primary_hits_expand_graph_with_remaining_budget():
    store = FakeStore(semantic=[1, 2], fts=[2], graph=[3, 4])

    results = hybrid_search(store, "query", k=5, provider=FakeProvider())

    assert ("graph", [1, 2], 2) in store.calls
    assert [(r["id"], r["rrf_score"], r["signals"]) for r in results] == [
        (2, 0.032522, ["semantic", "fts"]),
        (1, 0.016393, ["semantic"]),
        (3, 0.016393, ["graph"]),
        (4, 0.016129, ["graph"]),
    ]


@pytest.mark.parametrize("fts", [[1], [2], [2, 3]])
def test_graph_is_suppressed_at_or_above_primary_hit_threshold(fts):
    store = FakeStore(semantic=[1], fts=fts, graph=[4])

    results = hybrid_search(store, "query", k=2, provider=FakeProvider())

    # Overlap still counts twice: one distinct hit can reach the threshold.
    assert all(call[0] != "graph" for call in store.calls)
    assert all("graph" not in result["signals"] for result in results)


def test_graph_seeds_take_five_per_signal_and_deduplicate_in_signal_order():
    store = FakeStore(
        semantic=[1, 2, 3, 4, 5, 6], fts=[3, 2, 7, 8, 9, 10], graph=[6, 11]
    )

    results = hybrid_search(store, "query", k=15, provider=FakeProvider())

    assert ("graph", [1, 2, 3, 4, 5, 7, 8, 9], 3) in store.calls
    by_id = {r["id"]: r for r in results}
    assert by_id[6]["signals"] == ["semantic", "graph"]
    assert by_id[11]["signals"] == ["graph"]


@pytest.mark.parametrize("visible", [set(), {4}])
def test_historical_filter_precedes_hydration_without_replenishing(visible):
    store = FakeStore(semantic=[1, 2], fts=[3, 4])
    store.visible = visible
    reachable = set()

    results = hybrid_search(
        store, "query", k=1, as_of_reachable=reachable, provider=FakeProvider()
    )

    assert [r["id"] for r in results] == ([4] if visible else [])
    assert [call for call in store.calls if call[0] == "visible"] == [
        ("visible", 1, reachable),
        ("visible", 3, reachable),
        ("visible", 2, reachable),
        ("visible", 4, reachable),
    ]
    assert [call for call in store.calls if call[0] == "hydrate"] == (
        [("hydrate", 4)] if visible else []
    )
    assert sum(call[0] == "fts" for call in store.calls) == 1
    assert all(call[0] != "graph" for call in store.calls)


def test_path_and_export_boosts_rerank_before_final_limit():
    store = FakeStore(fts=[1, 2, 3])
    store.symbols[2].update(file="src/AUTH.ts", exported=True)
    store.symbols[3]["exported"] = True

    results = hybrid_search(store, "AUTHentication!", k=2)

    assert [(r["id"], r["rrf_score"]) for r in results] == [
        (2, 0.029032),
        (3, 0.019048),
    ]
    assert all(r["signals"] == ["fts"] for r in results)


def test_equal_scores_preserve_fused_signal_order():
    store = FakeStore(semantic=[20], fts=[10])

    results = hybrid_search(store, "query", k=2, provider=FakeProvider())

    assert [(r["id"], r["rrf_score"]) for r in results] == [
        (20, 0.016393),
        (10, 0.016393),
    ]


def test_missing_symbols_do_not_extend_hydration_past_three_times_k():
    store = FakeStore(semantic=[1, 2, 3, 4], fts=[5, 6, 7, 8])
    for sid in [1, 5, 2, 6, 3]:
        del store.symbols[sid]
    store.symbols[4].update(file="src/query.py", exported=True)

    results = hybrid_search(store, "query", k=2, provider=FakeProvider())

    assert [r["id"] for r in results] == [7]
    assert [call for call in store.calls if call[0] == "hydrate"] == [
        ("hydrate", sid) for sid in [1, 5, 2, 6, 3, 7]
    ]


@pytest.mark.parametrize("stage", ["graph", "visible", "hydrate"])
def test_non_retrieval_failures_propagate_unchanged(stage, capsys):
    store = FakeStore(fts=[1])
    store.fail_at = stage

    with pytest.raises(RuntimeError) as raised:
        hybrid_search(
            store,
            "query",
            k=2,
            as_of_reachable={"commit"} if stage == "visible" else None,
        )

    assert raised.value is store.failure
    assert capsys.readouterr().err == ""
