# Copyright 2026 David Scheiderman
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

"""Format blast-radius impact reports for stdout and markdown."""

from fnmatch import fnmatchcase
from pathlib import PurePosixPath

_SOURCE_SUFFIXES = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".mjs",
    ".cjs",
    ".mts",
    ".cts",
    ".vue",
    ".go",
    ".rs",
    ".rb",
    ".java",
    ".kt",
    ".kts",
    ".php",
}
_EXTRACT_INTERFACE_URL = (
    "https://github.com/IWasZ3r0Cool/blast-radius/blob/main/"
    "docs/refactor-patterns.md#extract-interface"
)


def _mitigation_guidance(
    file_id: str, blast: dict, guidance_threshold: int
) -> list[str]:
    """Suggest next steps from graph evidence, without claiming test coverage."""
    if blast["blast_score"] < guidance_threshold:
        return []

    candidates = []
    for nid in sorted(set(blast["direct_ids"]) | set(blast["transitive_ids"])):
        path = PurePosixPath(nid)
        if nid == file_id or path.suffix not in _SOURCE_SUFFIXES:
            continue
        if (
            {"test", "tests", "__tests__"}.intersection(path.parts[:-1])
            or path.stem.startswith("test_")
            or path.stem.endswith("_test")
            or fnmatchcase(path.name, "*.test.*")
            or fnmatchcase(path.name, "*.spec.*")
        ):
            candidates.append(nid)

    if candidates:
        paths = ", ".join(f"`{path}`" for path in candidates)
        guidance = [
            (
                f"Review and run downstream test candidates ({len(candidates)}): {paths}. "
                "Add regression cases where needed; dependency reachability does not establish coverage."
            )
        ]
    else:
        guidance = [
            (
                "No downstream test candidates were identified in the dependency graph. "
                "Review regression coverage before proceeding."
            )
        ]

    if blast["blast_score"] >= 20 and blast["direct_dependents"] > 8:
        guidance.append(
            "Consider extracting an interface to reduce coupling. "
            f"See [extract-interface guidance]({_EXTRACT_INTERFACE_URL})."
        )
    return guidance


def _risk_label(blast_score: float, total_nodes: int) -> str:
    if total_nodes == 0:
        return "UNKNOWN"
    pct = (blast_score / total_nodes) * 100
    if blast_score >= 20 or pct >= 30:
        return "CRITICAL"
    if blast_score >= 10 or pct >= 15:
        return "HIGH"
    if blast_score >= 4 or pct >= 5:
        return "MEDIUM"
    return "LOW"


def format_stdout(
    file_id: str,
    blast: dict,
    total_nodes: int,
    *,
    guidance_threshold: int = 10,
) -> str:
    d = blast["direct_dependents"]
    t = blast["transitive_dependents"]
    sc = blast["blast_score"]
    risk = _risk_label(sc, total_nodes)

    lines = [
        f"Impact: {file_id}",
        f"Blast Score: {sc}  ({d} direct · {t} transitive)  [{risk}]",
        "",
    ]

    if blast["direct_ids"]:
        lines.append(f"Direct dependents ({d})")
        for nid in sorted(blast["direct_ids"]):
            lines.append(f"  {nid}")
        lines.append("")

    if blast["transitive_ids"]:
        lines.append(f"Transitive dependents ({t})")
        paths = blast["dep_paths"]
        for nid in sorted(blast["transitive_ids"]):
            path = paths.get(nid, [nid])
            chain = " → ".join(reversed(path)) if len(path) > 1 else nid
            lines.append(f"  {nid}  ← {chain}" if len(path) > 1 else f"  {nid}")
        lines.append("")

    if total_nodes:
        affected = d + t
        pct = round((affected / total_nodes) * 100, 1)
        lines.append(
            f"Risk: {risk} — affects {affected}/{total_nodes} files ({pct}% of codebase)"
        )

    guidance = _mitigation_guidance(file_id, blast, guidance_threshold)
    if guidance:
        lines.extend(["", "Mitigation suggestions"])
        lines.extend(f"- {suggestion}" for suggestion in guidance)

    return "\n".join(lines)


def format_markdown(
    file_id: str,
    blast: dict,
    total_nodes: int,
    *,
    guidance_threshold: int = 10,
) -> str:
    d = blast["direct_dependents"]
    t = blast["transitive_dependents"]
    sc = blast["blast_score"]
    risk = _risk_label(sc, total_nodes)

    lines = [
        f"# Impact Report: `{file_id}`",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Blast Score | **{sc}** |",
        f"| Direct dependents | {d} |",
        f"| Transitive dependents | {t} |",
        f"| Risk | **{risk}** |",
        "",
    ]

    if blast["direct_ids"]:
        lines.append(f"## Direct dependents ({d})")
        lines.append("")
        for nid in sorted(blast["direct_ids"]):
            lines.append(f"- `{nid}`")
        lines.append("")

    if blast["transitive_ids"]:
        lines.append(f"## Transitive dependents ({t})")
        lines.append("")
        paths = blast["dep_paths"]
        for nid in sorted(blast["transitive_ids"]):
            path = paths.get(nid, [nid])
            chain = " → ".join(reversed(path)) if len(path) > 1 else ""
            suffix = f" ← `{chain}`" if chain else ""
            lines.append(f"- `{nid}`{suffix}")
        lines.append("")

    if total_nodes:
        affected = d + t
        pct = round((affected / total_nodes) * 100, 1)
        lines.append("## Summary")
        lines.append("")
        lines.append(
            f"Modifying `{file_id}` could affect **{affected}/{total_nodes}** files "
            f"(**{pct}%** of codebase). Risk level: **{risk}**."
        )
        lines.append("")

    guidance = _mitigation_guidance(file_id, blast, guidance_threshold)
    if guidance:
        lines.extend(["## Mitigation suggestions", ""])
        lines.extend(f"- {suggestion}" for suggestion in guidance)
        lines.append("")

    if total_nodes:
        lines.append(
            "_Generated by [blastradius](https://github.com/IWasZ3r0Cool/blast-radius)_"
        )

    return "\n".join(lines)
