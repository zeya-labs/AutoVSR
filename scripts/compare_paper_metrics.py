#!/usr/bin/env python3
"""Compare an AutoVSR result JSON against the paper's GLM benchmark targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable


PAPER_TARGETS = {
    "total_samples": 5020,
    "task_counts": {
        "transfer_function": 1376,
        "transient_response": 3644,
    },
    "type_counts": {
        "type1": 1146,
        "type2": 2671,
        "type3": 464,
        "type4": 511,
        "type5": 228,
    },
    "overall_accuracy": 68.02,
    "type_accuracy": {
        "type1": 92.77,
        "type2": 94.15,
        "type3": 49.14,
        "type4": 69.89,
        "type5": 47.81,
    },
}


def _load(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _accuracy(success: int, total: int) -> float:
    return 100.0 * success / total if total else 0.0


def _iter_results(data: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    results = data.get("results")
    if isinstance(results, list):
        return results
    samples = data.get("samples")
    if isinstance(samples, list):
        return samples
    return []


def compare(data: Dict[str, Any]) -> str:
    rows = []
    results = list(_iter_results(data))
    total = len(results)
    success = sum(1 for row in results if row.get("success"))
    rows.append("# Paper Metric Comparison")
    rows.append("")
    rows.append(f"- Samples evaluated: {total}")
    rows.append(f"- Overall success: {success}/{total} ({_accuracy(success, total):.2f}%)")
    rows.append(f"- Paper full benchmark size: {PAPER_TARGETS['total_samples']}")
    rows.append(f"- Paper overall accuracy: {PAPER_TARGETS['overall_accuracy']:.2f}%")
    rows.append("")

    if total != PAPER_TARGETS["total_samples"]:
        rows.append("Status: INCOMPLETE for full-paper reproduction because the sample count does not match 5020.")
        rows.append("")

    rows.append("| Task | Success | Total | Accuracy | Count Target |")
    rows.append("|---|---:|---:|---:|---:|")
    for task in ("transfer_function", "transient_response"):
        task_rows = [row for row in results if row.get("task") == task]
        task_total = len(task_rows)
        task_success = sum(1 for row in task_rows if row.get("success"))
        rows.append(
            f"| {task} | {task_success} | {task_total} | "
            f"{_accuracy(task_success, task_total):.2f}% | "
            f"{PAPER_TARGETS['task_counts'][task]} |"
        )

    rows.append("")
    rows.append("| Type | Success | Total | Accuracy | Paper Accuracy | Count Target |")
    rows.append("|---|---:|---:|---:|---:|---:|")
    all_types = set(PAPER_TARGETS["type_counts"]) | {str(row.get("type", "unknown")) for row in results}
    for typ in sorted(all_types):
        type_rows = [row for row in results if str(row.get("type", "unknown")) == typ]
        type_total = len(type_rows)
        type_success = sum(1 for row in type_rows if row.get("success"))
        target = PAPER_TARGETS["type_counts"].get(typ, "-")
        paper_accuracy = PAPER_TARGETS["type_accuracy"].get(typ)
        paper_accuracy_text = f"{paper_accuracy:.2f}%" if paper_accuracy is not None else "-"
        rows.append(
            f"| {typ} | {type_success} | {type_total} | "
            f"{_accuracy(type_success, type_total):.2f}% | {paper_accuracy_text} | {target} |"
        )

    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json", type=Path)
    parser.add_argument("--output", "-o", type=Path)
    args = parser.parse_args()

    report = compare(_load(args.result_json))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    else:
        print(report)


if __name__ == "__main__":
    main()
