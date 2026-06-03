#!/usr/bin/env python3
"""Check that sliced benchmark result files cover the full 5020-sample set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


EXPECTED_TOTAL = 5020


def _load_results(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError(f"{path} does not contain a results list")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json", type=Path, nargs="+")
    parser.add_argument("--expected-total", type=int, default=EXPECTED_TOTAL)
    args = parser.parse_args()

    seen: Dict[str, Path] = {}
    duplicates = []
    infrastructure_errors = []
    total_rows = 0
    for path in args.result_json:
        for item in _load_results(path):
            total_rows += 1
            sample_id = str(item.get("id", ""))
            if not sample_id:
                raise SystemExit(f"Missing sample id in {path}")
            error = str(item.get("error") or "")
            if any(marker in error for marker in ["429", "RateLimit", "rate limit", "速率限制"]):
                infrastructure_errors.append(f"{sample_id} in {path}")
            if sample_id in seen:
                duplicates.append(sample_id)
            seen[sample_id] = path

    if infrastructure_errors:
        preview = "; ".join(infrastructure_errors[:10])
        raise SystemExit(f"Infrastructure/rate-limit errors present in shard results: {preview}")
    if duplicates:
        preview = ", ".join(sorted(set(duplicates))[:10])
        raise SystemExit(f"Duplicate sample ids: {preview}")
    if total_rows != args.expected_total or len(seen) != args.expected_total:
        raise SystemExit(
            f"Incomplete shard coverage: rows={total_rows}, unique={len(seen)}, "
            f"expected={args.expected_total}"
        )
    print(f"Shard coverage OK: {len(seen)} unique samples")


if __name__ == "__main__":
    main()
