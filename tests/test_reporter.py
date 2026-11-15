# Copyright 2026 Christopher Wilson
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

import copy

import pytest

from blastradius.reporter import build_agent_impact, format_markdown, format_stdout


def test_agent_impact_limits_each_group_and_preserves_totals():
    target = "auth.py"
    direct = [f"routes/route_{i:02d}.py" for i in range(25)]
    transitive = [f"services/service_{i:02d}.py" for i in range(24)]
    graph = {
        "nodes": [{"id": nid} for nid in [target, *direct, *transitive]],
        "links": [
            *({"source": nid, "target": target} for nid in direct),
            *({"source": nid, "target": direct[0]} for nid in transitive),
        ],
    }

    bounded = build_agent_impact(graph, target)
    full = build_agent_impact(graph, target, all_dependents=True)
    assert bounded["blast_score"] == full["blast_score"] == 37.0
    assert bounded["risk_level"] == full["risk_level"] == "CRITICAL"
    assert full["direct"] == {
        "items": [{"id": nid} for nid in direct],
        "total": 25,
        "omitted": 0,
    }
    assert full["transitive"] == {
        "items": [{"id": nid, "path": [target, direct[0], nid]} for nid in transitive],
        "total": 24,
        "omitted": 0,
    }
    for group, omitted in (("direct", 5), ("transitive", 4)):
        assert bounded[group] == {
            "items": full[group]["items"][:20],
            "total": full[group]["total"],
            "omitted": omitted,
        }


def test_agent_impact_orders_dependents_and_ties_independent_of_graph_order():
    graph = {
        "nodes": [
            {"id": nid}
            for nid in (
                "target.py",
                "left.py",
                "right.py",
                "leaf.py",
                "deep.py",
                "another.py",
            )
        ],
        "links": [
            {"source": "right.py", "target": "target.py"},
            {"source": "left.py", "target": "target.py"},
            {"source": "leaf.py", "target": "right.py"},
            {"source": "leaf.py", "target": "left.py"},
            {"source": "another.py", "target": "right.py"},
            {"source": "deep.py", "target": "leaf.py"},
            {"source": "target.py", "target": "leaf.py"},
        ],
    }

    response = build_agent_impact(graph, "target.py")
    reversed_graph = {key: values[::-1] for key, values in graph.items()}
    assert build_agent_impact(reversed_graph, "target.py") == response
    assert response == {
        "schema_version": 1,
        "target": {"id": "target.py"},
        "blast_score": 3.5,
        "risk_level": "CRITICAL",
        "direct": {
            "items": [{"id": "left.py"}, {"id": "right.py"}],
            "total": 2,
            "omitted": 0,
        },
        "transitive": {
            "items": [
                {"id": "another.py", "path": ["target.py", "right.py", "another.py"]},
                {"id": "leaf.py", "path": ["target.py", "left.py", "leaf.py"]},
                {
                    "id": "deep.py",
                    "path": ["target.py", "left.py", "leaf.py", "deep.py"],
                },
            ],
            "total": 3,
            "omitted": 0,
        },
    }


@pytest.fixture
def blast():
    return {
        "blast_score": 20,
        "direct_dependents": 9,
        "transitive_dependents": 22,
        "direct_ids": ["src/client.py", "tests/test_client.py"],
        "transitive_ids": ["src/app.py", "tests/test_client.py", "app.spec.ts"],
        "dep_paths": {"src/app.py": ["src/client.py", "src/app.py"]},
    }


@pytest.mark.parametrize("formatter", [format_stdout, format_markdown])
@pytest.mark.parametrize(
    "score, threshold, enabled",
    [(9.5, 10, False), (10, 10, True), (10.5, 10, True), (20, 21, False), (2, 2, True)],
)
def test_guidance_threshold_preserves_existing_report(
    formatter, blast, score, threshold, enabled
):
    blast["blast_score"] = score
    before = copy.deepcopy(blast)
    report = formatter("src/core.py", blast, 100, guidance_threshold=threshold)
    without_guidance = formatter("src/core.py", blast, 100, guidance_threshold=100)
    assert ("Mitigation suggestions" in report) is enabled
    # Everything before the appended advice, including metrics and risk, is retained.
    assert (
        report.split("Mitigation suggestions")[0]
        .removesuffix("## ")
        .split("_Generated")[0]
        .rstrip()
        == without_guidance.split("_Generated")[0].rstrip()
    )
    assert blast == before


@pytest.mark.parametrize(
    "score, direct, threshold, advised",
    [
        (19.5, 9, 10, False),
        (20, 8, 10, False),
        (20, 9, 10, True),
        (21, 10, 10, True),
        (20, 9, 21, False),
    ],
)
def test_interface_advice_boundaries(blast, score, direct, threshold, advised):
    blast.update(blast_score=score, direct_dependents=direct)
    for formatter in (format_stdout, format_markdown):
        report = formatter("src/core.py", blast, 100, guidance_threshold=threshold)
        assert ("Consider extracting an interface" in report) is advised
        assert (
            "https://github.com/IWasZ3r0Cool/blast-radius/blob/main/docs/refactor-patterns.md#extract-interface"
            in report
        ) is advised


@pytest.mark.parametrize(
    "path",
    [
        "tests/client.py",
        "pkg/test/client.rs",
        "pkg/__tests__/client.ts",
        "test_client.py",
        "pkg/client_test.go",
        "pkg/client_test.rb",
        "client.test.tsx",
        "client.spec.mjs",
    ],
)
def test_conventional_test_candidates(blast, path):
    blast.update(direct_ids=[path], transitive_ids=[])
    for formatter in (format_stdout, format_markdown):
        report = formatter("src/core.py", blast, 100)
        assert f"downstream test candidates (1): `{path}`" in report
        assert "dependency reachability does not establish coverage" in report


@pytest.mark.parametrize(
    "path",
    [
        "src/client.py",
        "contest/client.py",
        "src/testing.py",
        "tests/README.md",
        "tests/fixture.json",
        "tests/package",
        "test_client.txt",
        "src/client.test.json",
        "tests/test_core.py",
    ],
)
def test_no_candidates_does_not_claim_tests_are_missing(blast, path):
    blast.update(direct_ids=[path], transitive_ids=[])
    report = format_stdout("tests/test_core.py", blast, 100)
    assert (
        "No downstream test candidates were identified in the dependency graph. Review regression coverage before proceeding."
        in report
    )


def test_candidates_are_sorted_unique_and_shared_across_formats(blast):
    stdout = format_stdout("src/core.py", blast, 100)
    markdown = format_markdown("src/core.py", blast, 100)
    expected = "downstream test candidates (2): `app.spec.ts`, `tests/test_client.py`."
    assert expected in stdout
    assert expected in markdown
    stdout_advice = [line for line in stdout.splitlines() if line.startswith("- ")]
    assert all(line in markdown for line in stdout_advice)
    assert markdown.index("## Mitigation suggestions") < markdown.index("_Generated")


@pytest.mark.parametrize("formatter", [format_stdout, format_markdown])
def test_guidance_does_not_require_nonzero_total(blast, formatter):
    report = formatter("src/core.py", blast, 0)
    assert "UNKNOWN" in report
    assert "Mitigation suggestions" in report
