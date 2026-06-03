#!/usr/bin/env python3
"""Summarize AutoVSR evaluation JSON files into compact tables."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _pct(success: int, total: int) -> float:
    return success / total * 100 if total else 0.0


def _add(bucket: Dict[str, Dict[str, int]], key: Any, success: bool) -> None:
    label = str(key if key not in (None, "") else "unknown")
    bucket[label]["total"] += 1
    bucket[label]["success"] += int(bool(success))


def _rows(group: Dict[str, Dict[str, int]]) -> List[str]:
    lines = ["| Group | Success | Total | Accuracy |", "|---|---:|---:|---:|"]
    for key, val in sorted(group.items()):
        lines.append(f"| {key} | {val['success']} | {val['total']} | {_pct(val['success'], val['total']):.2f}% |")
    return lines


def _stage_rows(stages: Dict[str, Dict[str, Any]]) -> List[str]:
    lines = [
        "| Stage | Duration Seconds | Total Tokens | Input Tokens | Output Tokens | LLM Calls |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for stage, data in sorted(stages.items()):
        tokens = data.get("tokens", {}) if isinstance(data, dict) else {}
        lines.append(
            f"| {stage} | {data.get('duration_seconds', 0):.2f} | "
            f"{tokens.get('total_tokens', 0)} | {tokens.get('input_tokens', 0)} | "
            f"{tokens.get('output_tokens', 0)} | {data.get('llm_calls', 0)} |"
        )
    return lines


def summarize(result: Dict[str, Any]) -> str:
    results = result.get("results", [])
    total = len(results)
    success = sum(1 for item in results if item.get("success"))

    groups = {
        "By Type": defaultdict(lambda: {"total": 0, "success": 0}),
        "By Task": defaultdict(lambda: {"total": 0, "success": 0}),
        "By Level": defaultdict(lambda: {"total": 0, "success": 0}),
        "By Source": defaultdict(lambda: {"total": 0, "success": 0}),
    }

    for item in results:
        ok = bool(item.get("success"))
        _add(groups["By Type"], item.get("type"), ok)
        _add(groups["By Task"], item.get("task"), ok)
        _add(groups["By Level"], item.get("level"), ok)
        _add(groups["By Source"], item.get("source"), ok)

    metrics = result.get("metrics_summary", {})
    lines = [
        "# AutoVSR Evaluation Summary",
        "",
        f"- Total: {total}",
        f"- Success: {success}",
        f"- Accuracy: {_pct(success, total):.2f}%",
        f"- Total tokens: {metrics.get('total_tokens', {}).get('total_tokens', 0)}",
        f"- LLM calls: {metrics.get('total_llm_calls', 0)}",
        f"- Duration seconds: {metrics.get('total_duration_seconds', 0)}",
        "",
    ]

    for title, group in groups.items():
        lines.extend([f"## {title}", ""])
        lines.extend(_rows(group))
        lines.append("")

    stages = metrics.get("by_stage", {})
    if stages:
        lines.extend(["## By Stage", ""])
        lines.extend(_stage_rows(stages))
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json", type=Path, help="Path to an AutoVSR output JSON file")
    parser.add_argument("--output", "-o", type=Path, help="Optional Markdown output path")
    args = parser.parse_args()

    data = json.loads(args.result_json.read_text(encoding="utf-8"))
    report = summarize(data)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    else:
        print(report)


if __name__ == "__main__":
    main()
