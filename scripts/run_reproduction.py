#!/usr/bin/env python3
"""Run the AutoVSR paper-reproduction command sequence."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: List[str], dry_run: bool) -> None:
    display = " ".join(cmd)
    print(display)
    if dry_run:
        return
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _run_checked(cmd: List[str], dry_run: bool) -> bool:
    display = " ".join(cmd)
    print(display)
    if dry_run:
        return True
    return subprocess.run(cmd, cwd=PROJECT_ROOT).returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "CircuitSense")
    parser.add_argument("--converted", type=Path, default=PROJECT_ROOT / "data" / "circuitsense_synthetic_autovsr.json")
    parser.add_argument("--result", default="circuitsense_result.json")
    parser.add_argument("--log", default="circuitsense_result.log")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-run", action="store_true", help="Download/convert/audit only; do not call the LLM benchmark.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--max-samples", type=int, help="Optional converter cap after filtering valid samples.")
    parser.add_argument("--run-max-samples", type=int, help="Optional runner cap passed to main.py without truncating the converted data.")
    parser.add_argument("--start-index", type=int, default=0, help="Inclusive zero-based dataset start index passed to main.py.")
    parser.add_argument("--end-index", type=int, help="Exclusive zero-based dataset end index passed to main.py.")
    parser.add_argument("--sample-id", action="append", dest="sample_ids", help="Run only this sample ID after index slicing. Repeatable.")
    parser.add_argument("--no-resume", action="store_true", help="Start benchmark from scratch instead of checkpoint resume.")
    parser.add_argument("--no-strict-audit", action="store_true", help="Do not require exact paper dataset counts before running.")
    parser.add_argument("--include-task", action="append", dest="include_tasks", help="Optional task filter passed to the converter. Repeatable.")
    parser.add_argument("--include-type", action="append", dest="include_types", help="Optional type filter passed to the converter. Repeatable.")
    parser.add_argument("--include-level", action="append", dest="include_levels", help="Optional level filter passed to the converter. Repeatable.")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip local environment preflight checks.")
    parser.add_argument("--hf-endpoint", help="Optional Hugging Face endpoint or mirror, e.g. https://hf-mirror.com.")
    parser.add_argument("--hf-cache-dir", type=Path, help="Optional Hugging Face cache directory.")
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--hf-local-files-only", action="store_true", help="Use only local Hugging Face cache/files for dataset download.")
    parser.add_argument("--hf-max-workers", type=int, default=8)
    parser.add_argument("--hf-retries", type=int, default=3, help="Retry failed Hugging Face snapshot downloads this many times.")
    parser.add_argument("--hf-retry-sleep", type=float, default=60.0, help="Seconds to sleep between Hugging Face retry attempts.")
    parser.add_argument("--download-audit-retries", type=int, default=0, help="Retry download/convert/audit cycles when strict audit detects partial data.")
    parser.add_argument("--download-audit-sleep", type=float, default=300.0, help="Seconds to sleep before retrying a partial-data download cycle.")
    args = parser.parse_args()

    py = sys.executable
    synthetic_dir = args.data_dir / "Analysis" / "synthetic"

    if not args.skip_preflight:
        _run([py, "scripts/check_reproduction_ready.py"], args.dry_run)

    def build_download_cmd() -> List[str]:
        cmd = [
            py,
            "scripts/download_circuitsense.py",
            "--output-dir",
            str(args.data_dir),
            "--revision",
            args.hf_revision,
            "--allow-pattern",
            "Analysis/synthetic/**",
            "--max-workers",
            str(args.hf_max_workers),
            "--retries",
            str(args.hf_retries),
            "--retry-sleep",
            str(args.hf_retry_sleep),
        ]
        if args.hf_endpoint:
            cmd += ["--endpoint", args.hf_endpoint]
        if args.hf_cache_dir:
            cmd += ["--cache-dir", str(args.hf_cache_dir)]
        if args.hf_local_files_only:
            cmd.append("--local-files-only")
        return cmd

    selected_tasks = args.include_tasks or ["transfer_function", "transient_response"]

    def build_convert_cmd() -> List[str]:
        cmd = [
            py,
            "scripts/convert_circuitsense.py",
            str(synthetic_dir),
            "--image-root",
            str(args.data_dir),
            "--output",
            str(args.converted),
        ]
        for task in selected_tasks:
            cmd += ["--include-task", task]
        for typ in args.include_types or []:
            cmd += ["--include-type", typ]
        for level in args.include_levels or []:
            cmd += ["--include-level", level]
        if args.max_samples:
            cmd += ["--max-samples", str(args.max_samples)]
        return cmd

    audit_path = PROJECT_ROOT / "output" / "circuitsense_dataset_audit.md"
    has_filters = bool(args.max_samples or args.include_tasks or args.include_types or args.include_levels)

    def build_audit_cmd() -> List[str]:
        cmd = [
            py,
            "scripts/audit_circuitsense_dataset.py",
            str(args.converted),
            "--output",
            str(audit_path),
        ]
        if not args.no_strict_audit and not has_filters:
            cmd.append("--strict")
        return cmd

    cycle_count = max(1, args.download_audit_retries + 1)
    for cycle in range(1, cycle_count + 1):
        if not args.skip_download:
            _run(build_download_cmd(), args.dry_run)

        if not args.dry_run and not synthetic_dir.exists():
            raise SystemExit(
                f"Missing synthetic dataset directory: {synthetic_dir}\n"
                "Run without --skip-download after network access is available, or place the dataset there."
            )

        _run(build_convert_cmd(), args.dry_run)
        audit_ok = _run_checked(build_audit_cmd(), args.dry_run)
        if audit_ok:
            break

        if args.skip_download or cycle >= cycle_count or args.no_strict_audit or has_filters:
            raise SystemExit("Dataset audit failed.")

        print(
            f"Strict audit failed after download cycle {cycle}/{cycle_count}; "
            f"sleeping {args.download_audit_sleep:.1f}s before retrying download."
        )
        time.sleep(args.download_audit_sleep)

    if args.skip_run:
        return

    run_cmd = [
        py,
        "main.py",
        "-t",
        "batch",
        "--data",
        str(args.converted),
        "--output",
        args.result,
        "--log",
        args.log,
    ]
    if args.run_max_samples:
        run_cmd += ["--max-samples", str(args.run_max_samples)]
    if args.start_index:
        run_cmd += ["--start-index", str(args.start_index)]
    if args.end_index is not None:
        run_cmd += ["--end-index", str(args.end_index)]
    for sample_id in args.sample_ids or []:
        run_cmd += ["--sample-id", sample_id]
    if args.no_resume:
        run_cmd.append("--no-resume")
    _run(run_cmd, args.dry_run)

    result_path = PROJECT_ROOT / "output" / args.result
    result_stem = Path(args.result).stem
    _run([
        py,
        "scripts/summarize_results.py",
        str(result_path),
        "--output",
        str(PROJECT_ROOT / "output" / f"{result_stem}_summary.md"),
    ], args.dry_run)
    _run([
        py,
        "scripts/compare_paper_metrics.py",
        str(result_path),
        "--output",
        str(PROJECT_ROOT / "output" / f"{result_stem}_paper_comparison.md"),
    ], args.dry_run)


if __name__ == "__main__":
    main()
