#!/usr/bin/env python3
"""Download CircuitSense synthetic data one level shard at a time."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEVELS = [
    "level0",
    "level1",
    "level2",
    "level3",
    "level4",
    "level5",
    "level5_bd",
]


def run(cmd: List[str], dry_run: bool) -> None:
    print(" ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "CircuitSense")
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--token")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=120.0)
    parser.add_argument("--between-shard-sleep", type=float, default=60.0)
    parser.add_argument("--level", action="append", dest="levels", help="Synthetic level shard to download. Repeatable.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    levels = args.levels or DEFAULT_LEVELS
    py = sys.executable

    for index, level in enumerate(levels, 1):
        print(f"\n[{index}/{len(levels)}] Downloading Analysis/synthetic/{level}/**")
        cmd = [
            py,
            "scripts/download_circuitsense.py",
            "--output-dir",
            str(args.output_dir),
            "--revision",
            args.revision,
            "--allow-pattern",
            f"Analysis/synthetic/{level}/**",
            "--max-workers",
            str(args.max_workers),
            "--retries",
            str(args.retries),
            "--retry-sleep",
            str(args.retry_sleep),
        ]
        if args.endpoint:
            cmd += ["--endpoint", args.endpoint]
        if args.cache_dir:
            cmd += ["--cache-dir", str(args.cache_dir)]
        if args.token:
            cmd += ["--token", args.token]
        run(cmd, args.dry_run)
        if index < len(levels) and args.between_shard_sleep > 0:
            print(f"Sleeping {args.between_shard_sleep:.1f}s before next shard...")
            time.sleep(args.between_shard_sleep)


if __name__ == "__main__":
    main()
