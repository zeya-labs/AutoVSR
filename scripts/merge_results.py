#!/usr/bin/env python3
"""Merge multiple AutoVSR result JSON files from sliced benchmark runs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _empty_stats() -> Dict[str, Any]:
    return {
        "total": 0,
        "success": 0,
        "success_rate": 0.0,
        "by_source": {},
        "by_level": {},
        "by_type": {},
        "by_task": {},
    }


def _add_group(stats: Dict[str, Any], group: str, key: Any, success: bool) -> None:
    label = str(key if key not in (None, "") else "unknown")
    stats.setdefault(group, {})
    stats[group].setdefault(label, {"total": 0, "success": 0})
    stats[group][label]["total"] += 1
    stats[group][label]["success"] += int(bool(success))


def _recompute_statistics(results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    stats = _empty_stats()
    for item in results:
        success = bool(item.get("success"))
        stats["total"] += 1
        stats["success"] += int(success)
        _add_group(stats, "by_source", item.get("source"), success)
        _add_group(stats, "by_level", item.get("level"), success)
        _add_group(stats, "by_type", item.get("type"), success)
        _add_group(stats, "by_task", item.get("task"), success)
    if stats["total"]:
        stats["success_rate"] = stats["success"] / stats["total"] * 100
    return stats


def _add_stage(target: Dict[str, Any], stage_name: str, stage_data: Dict[str, Any]) -> None:
    target.setdefault(stage_name, {
        "duration_seconds": 0,
        "tokens": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "llm_calls": 0,
    })
    target[stage_name]["duration_seconds"] += stage_data.get("duration_seconds", 0)
    tokens = stage_data.get("tokens", {})
    if isinstance(tokens, dict):
        target[stage_name]["tokens"]["input_tokens"] += tokens.get("input_tokens", 0)
        target[stage_name]["tokens"]["output_tokens"] += tokens.get("output_tokens", 0)
        target[stage_name]["tokens"]["total_tokens"] += tokens.get("total_tokens", 0)
    target[stage_name]["llm_calls"] += stage_data.get("llm_calls", 0)


def _recompute_metrics(results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    total_duration = 0.0
    total_tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    total_llm_calls = 0
    by_stage: Dict[str, Any] = {}

    for item in results:
        metrics = item.get("metrics", {}) or {}
        total_duration += metrics.get("total_duration_seconds", 0)
        total_tokens["input_tokens"] += metrics.get("total_input_tokens", 0)
        total_tokens["output_tokens"] += metrics.get("total_output_tokens", 0)
        total_tokens["total_tokens"] += metrics.get("total_tokens", 0)
        total_llm_calls += metrics.get("total_llm_calls", 0)
        for stage_name, stage_data in (metrics.get("by_stage") or {}).items():
            if isinstance(stage_data, dict):
                _add_stage(by_stage, stage_name, stage_data)

    return {
        "total_duration_seconds": round(total_duration, 2),
        "total_tokens": total_tokens,
        "total_llm_calls": total_llm_calls,
        "by_stage": by_stage,
    }


def _load_results(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError(f"{path} does not contain a results list")
    return results


def merge(paths: List[Path], replace_duplicates: bool = False) -> Dict[str, Any]:
    merged_by_id: Dict[str, Dict[str, Any]] = {}
    duplicates = []
    for path in paths:
        for item in _load_results(path):
            sample_id = str(item.get("id", ""))
            if sample_id in merged_by_id and not replace_duplicates:
                duplicates.append(sample_id)
                continue
            merged_by_id[sample_id] = item

    if duplicates:
        unique = sorted(set(duplicates))
        preview = ", ".join(unique[:10])
        raise ValueError(f"Duplicate sample ids found ({len(unique)}): {preview}")

    results = list(merged_by_id.values())
    return {
        "timestamp": datetime.now().isoformat(),
        "merged_from": [str(path) for path in paths],
        "statistics": _recompute_statistics(results),
        "metrics_summary": _recompute_metrics(results),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json", type=Path, nargs="+")
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--replace-duplicates", action="store_true")
    args = parser.parse_args()

    merged = merge(args.result_json, replace_duplicates=args.replace_duplicates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    stats = merged["statistics"]
    print(f"Wrote {stats['total']} merged samples to {args.output}")


if __name__ == "__main__":
    main()
