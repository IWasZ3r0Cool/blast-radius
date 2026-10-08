#!/usr/bin/env python3
# Copyright 2026 David Scheiderman
# Licensed under the Apache License, Version 2.0
"""Viz loading benchmark: synthetic graphs + static HTML checks.

Usage:
  uv run python benchmark/test_viz_loading.py [--sizes 1000,5000,20000]

Measures, for each synthetic graph size:
  - JSON payload bytes + parse time (mirrors fetch + parse stage)
  - Old-style lookup cost (linear NODES.find per link) vs indexed
    Map/adjacency cost (mirrors applyLayerFilter / getConnections /
    connCount hot paths in explorer.html)
  - Static checks that explorer.html contains the staged progress
    shell (progress bar stages, paint yields, sim budget, deferred
    cycles, CDN banner)

Exit non-zero if any static check fails.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PASS = "ok"
FAIL = "FAIL"


def make_graph(n_nodes: int) -> dict:
    nodes = [
        {
            "id": f"mod{i // 10}/file{i}.py",
            "type": "module",
            "size": 100 + (i % 400),
            "group": i % 8,
            "imports": 2,
            "loc": 100 + (i % 400),
            "language": "python",
        }
        for i in range(n_nodes)
    ]
    links = [
        {"source": nodes[i % n_nodes]["id"], "target": nodes[(i * 7 + 3) % n_nodes]["id"], "weight": 1 + (i % 3)}
        for i in range(n_nodes * 2)
    ]
    return {"nodes": nodes, "links": links, "meta": {"root": "synthetic/"}}


def bench_lookups(data: dict) -> tuple[float, float]:
    nodes = data["nodes"]
    links = data["links"]

    def node_id(n):
        return n["id"] if isinstance(n, dict) else n

    # Old style: linear scan per link endpoint (applyLayerFilter pattern).
    t0 = time.perf_counter()
    for l in links:
        s = next(n for n in nodes if n["id"] == node_id(l["source"]))
        t = next(n for n in nodes if n["id"] == node_id(l["target"]))
        _ = (s["id"], t["id"])
    old_ms = (time.perf_counter() - t0) * 1000.0

    # New style: Map + adjacency index.
    t0 = time.perf_counter()
    idx = {n["id"]: n for n in nodes}
    adj: dict[str, list] = {}
    for l in links:
        adj.setdefault(node_id(l["source"]), []).append(node_id(l["target"]))
        adj.setdefault(node_id(l["target"]), []).append(node_id(l["source"]))
    for l in links:
        s = idx[node_id(l["source"])]
        t = idx[node_id(l["target"])]
        _ = (s["id"], t["id"], len(adj[s["id"]]))
    new_ms = (time.perf_counter() - t0) * 1000.0
    return old_ms, new_ms


STATIC_CHECKS = {
    "progress bar element": "viz-progress",
    "progress stage label": "viz-progress-label",
    "paint yield helper": "vizNextFrame",
    "node index map": "NODE_INDEX",
    "adjacency index": "ADJ_INDEX",
    "sim tick budget": "SIM_TICK_BUDGET",
    "deferred cycles": "cycleEdgesDeferred",
    "CDN banner": "cdn-banner",
}


def check_html(path: Path) -> dict[str, bool]:
    text = path.read_text(encoding="utf-8")
    return {label: token in text for label, token in STATIC_CHECKS.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Viz loading benchmark")
    parser.add_argument("--sizes", default="1000,5000,20000")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    html_path = repo / "blastradius" / "static" / "explorer.html"

    print(f"{'nodes':>7} {'links':>7} {'bytes':>9} {'parse_ms':>9} {'old_ms':>9} {'new_ms':>9} speedup")
    print("-" * 70)
    for size in [int(s) for s in args.sizes.split(",") if s.strip()]:
        data = make_graph(size)
        payload = json.dumps(data)
        t0 = time.perf_counter()
        json.loads(payload)
        parse_ms = (time.perf_counter() - t0) * 1000.0
        old_ms, new_ms = bench_lookups(data)
        speedup = old_ms / new_ms if new_ms > 0 else float("inf")
        print(f"{size:>7} {len(data['links']):>7} {len(payload):>9} {parse_ms:>9.1f} {old_ms:>9.1f} {new_ms:>9.1f} {speedup:>6.1f}x")

    print("\n-- explorer.html static checks --")
    results = check_html(html_path)
    failed = 0
    for label, ok in results.items():
        print(f"  [{PASS if ok else FAIL}] {label}")
        if not ok:
            failed += 1
    if failed:
        print(f"\n{failed} static check(s) missing — implement the loading plan.", file=sys.stderr)
        sys.exit(1)
    print("\nAll viz loading checks passed.")


if __name__ == "__main__":
    main()
