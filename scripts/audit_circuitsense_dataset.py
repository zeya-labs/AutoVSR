#!/usr/bin/env python3
"""Audit a converted CircuitSense JSON before running the full benchmark."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


EXPECTED_TOTAL = 5020
EXPECTED_TASK_COUNTS = {
    "transfer_function": 1376,
    "transient_response": 3644,
}
EXPECTED_TYPE_COUNTS = {
    "type1": 1146,
    "type2": 2671,
    "type3": 464,
    "type4": 511,
    "type5": 228,
}


def _samples(data: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    samples = data.get("samples")
    if isinstance(samples, list):
        return samples
    results = data.get("results")
    if isinstance(results, list):
        return results
    return []


def _collect_issues(samples: List[Dict[str, Any]]) -> List[str]:
    issues: List[str] = []
    ids = [str(sample.get("id", "")) for sample in samples]
    duplicate_ids = sorted({sample_id for sample_id in ids if ids.count(sample_id) > 1})
    if duplicate_ids:
        issues.append(f"duplicate ids: {len(duplicate_ids)}")

    for idx, sample in enumerate(samples):
        sample_id = sample.get("id", f"sample_{idx}")
        for field in ("id", "question", "answer", "image_path", "task", "type"):
            if sample.get(field) in (None, ""):
                issues.append(f"{sample_id}: missing {field}")
        image_path = sample.get("image_path")
        if image_path and not Path(str(image_path)).exists():
            issues.append(f"{sample_id}: missing image file {image_path}")
    return issues


def _expected_mismatches(samples: List[Dict[str, Any]]) -> List[str]:
    task_counts = Counter(str(sample.get("task", "unknown")) for sample in samples)
    type_counts = Counter(str(sample.get("type", "unknown")) for sample in samples)
    mismatches = []
    if len(samples) != EXPECTED_TOTAL:
        mismatches.append(f"total={len(samples)} expected={EXPECTED_TOTAL}")
    for task, expected in EXPECTED_TASK_COUNTS.items():
        actual = task_counts.get(task, 0)
        if actual != expected:
            mismatches.append(f"{task}={actual} expected={expected}")
    for typ, expected in EXPECTED_TYPE_COUNTS.items():
        actual = type_counts.get(typ, 0)
        if actual != expected:
            mismatches.append(f"{typ}={actual} expected={expected}")
    return mismatches


def audit(data: Dict[str, Any]) -> Tuple[str, List[str], List[str]]:
    samples = list(_samples(data))
    task_counts = Counter(str(sample.get("task", "unknown")) for sample in samples)
    type_counts = Counter(str(sample.get("type", "unknown")) for sample in samples)
    source_counts = Counter(str(sample.get("source", "unknown")) for sample in samples)
    field_issues = _collect_issues(samples)
    expected_mismatches = _expected_mismatches(samples)

    lines = ["# CircuitSense Dataset Audit", ""]
    lines.append(f"- Total samples: {len(samples)}")
    lines.append(f"- Expected paper total: {EXPECTED_TOTAL}")
    lines.append("")
    if len(samples) != EXPECTED_TOTAL:
        lines.append("Status: INCOMPLETE for full-paper reproduction because total sample count differs from 5020.")
        lines.append("")
    if field_issues:
        lines.append(f"Field/path issues: {len(field_issues)}")
        for issue in field_issues[:20]:
            lines.append(f"- {issue}")
        if len(field_issues) > 20:
            lines.append(f"- ... {len(field_issues) - 20} more")
        lines.append("")

    lines.append("| Task | Count | Expected |")
    lines.append("|---|---:|---:|")
    for task, expected in EXPECTED_TASK_COUNTS.items():
        lines.append(f"| {task} | {task_counts.get(task, 0)} | {expected} |")
    for task in sorted(set(task_counts) - set(EXPECTED_TASK_COUNTS)):
        lines.append(f"| {task} | {task_counts[task]} | - |")

    lines.append("")
    lines.append("| Type | Count | Expected |")
    lines.append("|---|---:|---:|")
    for typ, expected in EXPECTED_TYPE_COUNTS.items():
        lines.append(f"| {typ} | {type_counts.get(typ, 0)} | {expected} |")
    for typ in sorted(set(type_counts) - set(EXPECTED_TYPE_COUNTS)):
        lines.append(f"| {typ} | {type_counts[typ]} | - |")

    lines.append("")
    lines.append("| Source | Count |")
    lines.append("|---|---:|")
    for source, count in sorted(source_counts.items()):
        lines.append(f"| {source} | {count} |")

    return "\n".join(lines), field_issues, expected_mismatches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_json", type=Path)
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero unless counts, fields, and image paths match the paper benchmark.")
    args = parser.parse_args()

    with args.dataset_json.open(encoding="utf-8") as f:
        data = json.load(f)
    report, field_issues, expected_mismatches = audit(data)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    else:
        print(report)
    if args.strict and (field_issues or expected_mismatches):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
