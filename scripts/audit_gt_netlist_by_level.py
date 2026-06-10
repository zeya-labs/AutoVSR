#!/usr/bin/env python3
"""Audit CircuitSense synthetic levels using ground-truth netlists.

This bypasses image parsing and LLM planning. For each sample with
`q*_netlist.txt`, it parses the task from the question, computes the requested
quantity with deterministic netlist tools, and compares against `q*_ta.txt`
with the repository's SymPy equivalence checker.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


WORKER_CODE = r'''
from pathlib import Path
import json
import re
import sys

case_dir = Path(sys.argv[1])
compare_timeout = float(sys.argv[2])
stem = case_dir.name

try:
    from src.tools.netlist_tools import LcapySolver
    import main

    qpath = case_dir / f"{stem}_question.txt"
    npath = case_dir / f"{stem}_netlist.txt"
    apath = case_dir / f"{stem}_ta.txt"

    missing = [str(p.name) for p in (qpath, npath, apath) if not p.exists()]
    if missing:
        print(json.dumps({"id": stem, "status": "missing_files", "missing": missing}))
        sys.exit(0)

    question = qpath.read_text().strip()
    reference = apath.read_text().strip()
    netlist = npath.read_text()

    def step_netlist(text):
        lines = []
        for raw_line in text.splitlines():
            parts = raw_line.split()
            if len(parts) >= 4 and parts[0].upper().startswith("V"):
                parts = parts[:3] + ["step"] + parts[4:]
                lines.append(" ".join(parts))
            else:
                lines.append(raw_line)
        return "\n".join(lines)

    solver = LcapySolver()
    solver.load_from_ir({"netlist": netlist})

    lowered = question.lower()
    task = ""
    source = ""
    target = ""
    predicted = None

    transfer_match = re.search(r"from\s+(\w+)\s+to\s+(\w+)", question, re.I)
    node_match = re.search(r"\bnode\s+(\d+)\b", question, re.I)
    current_match = re.search(r"\b(iv|il|ieint)\s*(\d+)\b", question, re.I)

    if "transfer function" in lowered or "gain" in lowered or transfer_match:
        if not transfer_match:
            print(json.dumps({"id": stem, "status": "question_parse_error", "question": question}))
            sys.exit(0)
        task = "transfer_function"
        source, target = transfer_match.group(1), transfer_match.group(2)
        result = solver.element_transfer_function(target, source)
        if result.get("success"):
            predicted = str(result.get("transfer_function"))
            if "Ad" in predicted or "Ad" in netlist:
                import sympy as sp
                import main as main_module

                expr = sp.sympify(main_module._normalize_expression_text(predicted))
                predicted = str(sp.limit(expr, sp.Symbol("Ad"), sp.oo))
        else:
            print(json.dumps({
                "id": stem,
                "status": "solver_error",
                "task": task,
                "src": source,
                "target": target,
                "error": result.get("error"),
            }))
            sys.exit(0)
    elif "voltage source current" in lowered or "branch current" in lowered or current_match:
        from lcapy import Circuit

        task = "branch_current"
        if current_match:
            prefix, number = current_match.group(1).lower(), current_match.group(2)
            target = f"{prefix}{number}"
        else:
            target = "iv1"
        if target.startswith("iv"):
            element = f"V{target[2:]}"
        elif target.startswith("il"):
            element = f"L{target[2:]}"
        elif target.startswith("ieint"):
            element = f"Eint{target[5:]}"
        else:
            element = target
        cct = Circuit(step_netlist(netlist))
        predicted = f"{target}(s) = {cct[element].I.s}"
    elif "nodal equation" in lowered or "node voltage" in lowered or node_match:
        from lcapy import Circuit

        if not node_match:
            print(json.dumps({"id": stem, "status": "question_parse_error", "question": question}))
            sys.exit(0)
        task = "node_voltage"
        target = node_match.group(1)
        cct = Circuit(step_netlist(netlist))
        predicted = f"Vn{target}(s) = {cct[target].V.s}"
    else:
        print(json.dumps({"id": stem, "status": "unsupported_question", "question": question}))
        sys.exit(0)

    equivalent = main.symbolic_equivalent(predicted, reference, timeout_seconds=compare_timeout)
    if equivalent is True:
        status = "correct"
    elif equivalent is False:
        status = "mismatch"
    else:
        status = "equiv_compare_failed"

    print(json.dumps({
        "id": stem,
        "status": status,
        "task": task,
        "src": source,
        "target": target,
        "pred": predicted,
        "ref": reference,
    }))
except Exception as exc:
    print(json.dumps({"id": stem, "status": "exception", "error": f"{type(exc).__name__}: {exc}"}))
'''


def case_key(path_or_id: Path | str) -> int:
    name = path_or_id.name if isinstance(path_or_id, Path) else str(path_or_id)
    if name.startswith("q") and name[1:].isdigit():
        return int(name[1:])
    return 10**9


def run_case(
    case_dir: Path,
    timeout: int,
    compare_timeout: float,
    lcapy_timeout: int | None = None,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    if lcapy_timeout is not None:
        env["LCAPY_TIMEOUT_SECONDS"] = str(lcapy_timeout)
    try:
        completed = subprocess.run(
            [sys.executable, "-c", WORKER_CODE, str(case_dir), str(compare_timeout)],
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "id": case_dir.name,
            "status": "timeout",
            "error": f">{timeout}s",
            "stdout": (exc.stdout or "")[-300:] if isinstance(exc.stdout, str) else "",
        }
    except Exception as exc:
        return {"id": case_dir.name, "status": "parent_exception", "error": f"{type(exc).__name__}: {exc}"}

    lines = (completed.stdout or "").strip().splitlines()
    if lines:
        try:
            item = json.loads(lines[-1])
        except json.JSONDecodeError:
            item = {"id": case_dir.name, "status": "bad_json", "stdout": completed.stdout[-500:]}
    else:
        item = {"id": case_dir.name, "status": "no_output", "stderr": completed.stderr[-500:]}

    if completed.returncode != 0 and item.get("status") != "exception":
        item["returncode"] = completed.returncode
        item["stderr"] = completed.stderr[-500:]
    return item


def audit_level(
    level_dir: Path,
    workers: int,
    timeout: int,
    compare_timeout: float,
    lcapy_timeout: int | None,
) -> dict[str, Any]:
    cases = sorted([p for p in level_dir.glob("q*") if p.is_dir()], key=case_key)
    start = time.time()
    results: list[dict[str, Any]] = []
    print(f"{level_dir.name}: total={len(cases)} workers={workers} timeout={timeout}s", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_case, case_dir, timeout, compare_timeout, lcapy_timeout): case_dir
            for case_dir in cases
        }
        for index, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if index % 100 == 0 or index == len(cases):
                counts = Counter(item["status"] for item in results)
                print(
                    f"  progress={index}/{len(cases)} "
                    f"correct={counts.get('correct', 0)} "
                    f"failures={index - counts.get('correct', 0)} "
                    f"elapsed={time.time() - start:.1f}s",
                    flush=True,
                )

    counts = Counter(item["status"] for item in results)
    total = len(results)
    correct = counts.get("correct", 0)
    return {
        "level": level_dir.name,
        "total": total,
        "correct": correct,
        "accuracy": (correct / total * 100) if total else 0.0,
        "by_status": dict(sorted(counts.items())),
        "elapsed_seconds": round(time.time() - start, 2),
        "results": sorted(results, key=lambda item: case_key(item.get("id", ""))),
    }


def main_cli() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "data" / "CircuitSense" / "Analysis" / "synthetic",
        help="CircuitSense synthetic directory containing level* folders.",
    )
    parser.add_argument("--level", action="append", help="Specific level name to audit. Repeatable.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=25)
    parser.add_argument("--compare-timeout", type=float, default=8)
    parser.add_argument(
        "--lcapy-timeout",
        type=int,
        default=None,
        help="Override LCAPY_TIMEOUT_SECONDS inside solver subprocesses.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output" / "gt_netlist_solver_by_level.json",
    )
    args = parser.parse_args()

    levels = sorted([p for p in args.root.glob("level*") if p.is_dir()], key=lambda p: p.name)
    if args.level:
        wanted = set(args.level)
        levels = [p for p in levels if p.name in wanted]

    summaries = []
    all_results: dict[str, Any] = {}
    start = time.time()
    for level_dir in levels:
        level_result = audit_level(
            level_dir,
            args.workers,
            args.timeout,
            args.compare_timeout,
            args.lcapy_timeout,
        )
        summaries.append({k: v for k, v in level_result.items() if k != "results"})
        all_results[level_dir.name] = level_result["results"]

    output = {
        "root": str(args.root),
        "workers": args.workers,
        "case_timeout_seconds": args.timeout,
        "compare_timeout_seconds": args.compare_timeout,
        "lcapy_timeout_seconds": args.lcapy_timeout,
        "elapsed_seconds": round(time.time() - start, 2),
        "summary": summaries,
        "results": all_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\nSUMMARY")
    print("level\ttotal\tcorrect\taccuracy\tby_status")
    for item in summaries:
        print(
            f"{item['level']}\t{item['total']}\t{item['correct']}\t"
            f"{item['accuracy']:.2f}%\t{item['by_status']}"
        )
    print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
