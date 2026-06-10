#!/usr/bin/env python3
"""Evaluate image-to-netlist generation on CircuitSense synthetic levels.

This runs only the Netlist build node, using the LLM configured in
`config/config.yaml`, and compares the generated netlist against each sample's
`q*_netlist.txt`.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _case_key(path: Path) -> int:
    if path.name.startswith("q") and path.name[1:].isdigit():
        return int(path.name[1:])
    return 10**9


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def _normalize_source_domains(line: str) -> str:
    parts = line.split()
    if len(parts) >= 4 and parts[0].upper().startswith("V"):
        if parts[3] in {"s", "step", "dc"}:
            parts = parts[:3] + parts[4:]
    return " ".join(parts)


def _netlist_lines(text: str) -> list[str]:
    lines = []
    for raw in str(text or "").splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line or line.startswith(("#", "*", ".")):
            continue
        lines.append(_normalize_source_domains(" ".join(line.split())))
    return lines


def _component_map(text: str) -> dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]]:
    comps: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {}
    for line in _netlist_lines(text):
        parts = line.split()
        if len(parts) < 4:
            continue
        name = parts[0]
        prefix = re.match(r"[A-Za-z]+", name)
        kind = prefix.group(0).upper() if prefix else name[0].upper()
        nodes = tuple(parts[1:3])
        args = tuple(parts[3:])
        comps[name] = (kind, nodes, args)
    return comps


def _component_signature(text: str, *, ignore_nodes: bool = False) -> Counter:
    sig = Counter()
    for name, (kind, nodes, args) in _component_map(text).items():
        key = (name, kind, args) if ignore_nodes else (name, kind, tuple(sorted(nodes)), args)
        sig[key] += 1
    return sig


def _score_netlist(predicted: str, expected: str) -> dict[str, Any]:
    pred_lines = _netlist_lines(predicted)
    exp_lines = _netlist_lines(expected)
    pred_set = Counter(pred_lines)
    exp_set = Counter(exp_lines)
    line_hits = sum((pred_set & exp_set).values())
    pred_components = _component_map(predicted)
    exp_components = _component_map(expected)
    common_names = sorted(set(pred_components) & set(exp_components))
    name_hits = len(common_names)
    kind_hits = sum(1 for name in common_names if pred_components[name][0] == exp_components[name][0])
    value_hits = sum(1 for name in common_names if pred_components[name][2] == exp_components[name][2])
    undirected_node_hits = sum(
        1
        for name in common_names
        if tuple(sorted(pred_components[name][1])) == tuple(sorted(exp_components[name][1]))
    )
    exact_component_hits = sum(1 for name in common_names if pred_components[name] == exp_components[name])

    return {
        "expected_components": len(exp_components),
        "predicted_components": len(pred_components),
        "exact_text_match": "\n".join(pred_lines) == "\n".join(exp_lines),
        "line_recall": line_hits / len(exp_lines) if exp_lines else 0.0,
        "line_precision": line_hits / len(pred_lines) if pred_lines else 0.0,
        "component_name_recall": name_hits / len(exp_components) if exp_components else 0.0,
        "component_name_precision": name_hits / len(pred_components) if pred_components else 0.0,
        "component_type_accuracy_on_common": kind_hits / name_hits if name_hits else 0.0,
        "component_value_accuracy_on_common": value_hits / name_hits if name_hits else 0.0,
        "undirected_terminal_accuracy_on_common": undirected_node_hits / name_hits if name_hits else 0.0,
        "exact_component_accuracy_on_common": exact_component_hits / name_hits if name_hits else 0.0,
        "component_multiset_match_ignore_nodes": _component_signature(predicted, ignore_nodes=True)
        == _component_signature(expected, ignore_nodes=True),
        "component_multiset_match_with_undirected_nodes": _component_signature(predicted)
        == _component_signature(expected),
    }


def _load_config_meta() -> dict[str, Any]:
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    llm = dict(config.get("llm") or {})
    if llm.get("api_key"):
        llm["api_key"] = "***"
    return {"llm": llm, "ir_netlist": (config.get("ir") or {}).get("netlist") or {}}


def _initial_state(case_dir: Path, *, return_build_prompt: bool = False) -> dict[str, Any]:
    qid = case_dir.name
    question = _read(case_dir / f"{qid}_question.txt")
    return {
        "image_path": str(case_dir / f"{qid}_image.png"),
        "question": question,
        "ir_type": "netlist",
        "analysis_type": "transfer_function" if "transfer function" in question.lower() else "transient_response",
        "input_source": None,
        "output_node": None,
        "constraints": None,
        "detected_components": None,
        "provided_netlist": None,
        "return_build_prompt": return_build_prompt,
        "metrics": {},
    }


def evaluate_case(case_dir: Path, llm: Any, quiet_build_log: bool, return_build_prompt: bool = False) -> dict[str, Any]:
    from src.nodes.netlist.build import build_netlist_node

    qid = case_dir.name
    expected = _read(case_dir / f"{qid}_netlist.txt")
    state = _initial_state(case_dir, return_build_prompt=return_build_prompt)
    case_start = time.time()
    try:
        if quiet_build_log:
            with contextlib.redirect_stdout(io.StringIO()):
                built = build_netlist_node(state, llm)
        else:
            built = build_netlist_node(state, llm)
        predicted = built.get("ir_code") or ""
        score = _score_netlist(predicted, expected)
        row = {
            "id": qid,
            "success": bool(built.get("ir")) and not built.get("error"),
            "error": built.get("error"),
            "question": state["question"],
            "image_path": state["image_path"],
            "expected_netlist": expected,
            "predicted_netlist": predicted,
            "score": score,
            "metrics": built.get("metrics") or {},
            "duration_seconds": round(time.time() - case_start, 2),
        }
        if built.get("build_prompt"):
            row["build_prompt"] = built["build_prompt"]
        return row
    except Exception as exc:
        return {
            "id": qid,
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "question": state["question"],
            "image_path": state["image_path"],
            "expected_netlist": expected,
            "predicted_netlist": "",
            "score": _score_netlist("", expected),
            "duration_seconds": round(time.time() - case_start, 2),
        }


def _run_case_subprocess(case_dir: Path, timeout: int, return_build_prompt: bool = False) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--case-dir",
        str(case_dir),
        "--quiet-build-log",
    ]
    if return_build_prompt:
        cmd.append("--capture-prompt")
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    marker = "__RESULT_JSON__"
    for line in reversed((completed.stdout or "").splitlines()):
        if line.startswith(marker):
            row = json.loads(line[len(marker) :])
            if completed.returncode != 0 and not row.get("error"):
                row["error"] = f"child_returncode={completed.returncode}; {completed.stderr[-500:]}"
                row["success"] = False
            return row
    return {
        "id": case_dir.name,
        "success": False,
        "error": f"child_no_result returncode={completed.returncode}; stderr={completed.stderr[-1000:]}",
        "question": _read(case_dir / f"{case_dir.name}_question.txt"),
        "image_path": str(case_dir / f"{case_dir.name}_image.png"),
        "expected_netlist": _read(case_dir / f"{case_dir.name}_netlist.txt"),
        "predicted_netlist": "",
        "score": _score_netlist("", _read(case_dir / f"{case_dir.name}_netlist.txt")),
        "duration_seconds": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--level",
        default="level0",
        choices=("level0", "level1", "level2", "level4"),
        help="Synthetic level name under data/CircuitSense/Analysis/synthetic.",
    )
    parser.add_argument(
        "--level-dir",
        type=Path,
        default=None,
        help="Explicit level directory. Overrides --level.",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--start", type=int, default=0, help="Start offset after numeric q sorting.")
    parser.add_argument(
        "--case-id",
        default=None,
        help="Evaluate a single case id under the selected level, e.g. q49 or 49. Overrides --start/--limit.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent samples to evaluate.")
    parser.add_argument("--case-timeout", type=int, default=240, help="Per-sample timeout in concurrent mode.")
    parser.add_argument("--case-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--capture-prompt", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument("--quiet-build-log", action="store_true", help="Suppress verbose build node stdout.")
    parser.add_argument("--print-prompt", action="store_true", help="Print and save the exact text prompts sent to the LLM.")
    parser.add_argument("--print-netlists", action="store_true", help="Print expected and predicted netlists.")
    args = parser.parse_args()

    import main as autovsr_main

    if args.case_dir is not None:
        llm = autovsr_main.create_llm()
        row = evaluate_case(args.case_dir, llm, args.quiet_build_log, args.capture_prompt)
        print("__RESULT_JSON__" + json.dumps(row, ensure_ascii=False))
        return 0

    if args.level_dir is None:
        args.level_dir = (
            PROJECT_ROOT
            / "data"
            / "CircuitSense"
            / "Analysis"
            / "synthetic"
            / args.level
        )
    else:
        args.level = args.level_dir.name
    if args.case_id is not None and not args.case_id.startswith("q"):
        args.case_id = f"q{args.case_id}"
    if args.output is None:
        output_stem = (
            f"vlm_netlist_{args.level}_{args.case_id}_workers{args.workers}"
            if args.case_id
            else f"vlm_netlist_{args.level}_start{args.start}_limit{args.limit}_workers{args.workers}"
        )
        args.output = (
            PROJECT_ROOT
            / "output"
            / f"{output_stem}.json"
        )

    cases = sorted([p for p in args.level_dir.glob("q*") if p.is_dir()], key=_case_key)
    if args.case_id:
        case_dir = args.level_dir / args.case_id
        if not case_dir.is_dir():
            parser.error(f"case not found: {case_dir}")
        selected = [case_dir]
    else:
        selected = cases[args.start : args.start + args.limit]

    results: list[dict[str, Any]] = []
    started = time.time()
    if args.workers <= 1:
        llm = autovsr_main.create_llm()
        for index, case_dir in enumerate(selected, 1):
            print(f"[{index}/{len(selected)}] {case_dir.name}", flush=True)
            row = evaluate_case(case_dir, llm, args.quiet_build_log, args.print_prompt)
            results.append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            payload = _payload(args, selected, results, started)
            args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        print(f"Running {len(selected)} cases with workers={args.workers}", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_run_case_subprocess, case_dir, args.case_timeout, args.print_prompt): case_dir
                for case_dir in selected
            }
            for index, future in enumerate(as_completed(futures), 1):
                case_dir = futures[future]
                try:
                    row = future.result()
                except subprocess.TimeoutExpired:
                    row = {
                        "id": case_dir.name,
                        "success": False,
                        "error": f"case_timeout>{args.case_timeout}s",
                        "question": _read(case_dir / f"{case_dir.name}_question.txt"),
                        "image_path": str(case_dir / f"{case_dir.name}_image.png"),
                        "expected_netlist": _read(case_dir / f"{case_dir.name}_netlist.txt"),
                        "predicted_netlist": "",
                        "score": _score_netlist("", _read(case_dir / f"{case_dir.name}_netlist.txt")),
                        "duration_seconds": args.case_timeout,
                    }
                except Exception as exc:
                    row = {
                        "id": case_dir.name,
                        "success": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "question": _read(case_dir / f"{case_dir.name}_question.txt"),
                        "image_path": str(case_dir / f"{case_dir.name}_image.png"),
                        "expected_netlist": _read(case_dir / f"{case_dir.name}_netlist.txt"),
                        "predicted_netlist": "",
                        "score": _score_netlist("", _read(case_dir / f"{case_dir.name}_netlist.txt")),
                        "duration_seconds": 0,
                    }
                results.append(row)
                payload = _payload(args, selected, sorted(results, key=lambda item: _case_key(Path(item["id"]))), started)
                args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
                print(
                    f"[{index}/{len(selected)}] {row['id']} success={row.get('success')} "
                    f"component_match={row['score'].get('component_multiset_match_ignore_nodes')} "
                    f"terminal_acc={row['score'].get('undirected_terminal_accuracy_on_common'):.3f}",
                    flush=True,
                )

    print_summary(_payload(args, selected, results, started))
    if args.print_prompt:
        print_prompts(results)
    if args.print_netlists:
        for row in sorted(results, key=lambda item: _case_key(Path(item["id"]))):
            print(f"\n===== {row['id']} EXPECTED =====")
            print(row.get("expected_netlist") or "")
            print(f"\n===== {row['id']} PREDICTED =====")
            print(row.get("predicted_netlist") or "")
    return 0


def _payload(args: argparse.Namespace, selected: list[Path], results: list[dict[str, Any]], started: float) -> dict[str, Any]:
    return {
        "config": _load_config_meta(),
        "level_dir": str(args.level_dir),
        "output_path": str(args.output),
        "start": args.start,
        "limit": args.limit,
        "case_id": args.case_id,
        "workers": args.workers,
        "selected_ids": [p.name for p in selected],
        "elapsed_seconds": round(time.time() - started, 2),
        "summary": _summary(results),
        "results": results,
    }


def _avg(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row["score"].get(key) or 0.0) for row in rows) / len(rows)


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    return {
        "total": total,
        "build_success": sum(1 for row in results if row.get("success")),
        "component_multiset_match_ignore_nodes": sum(
            1 for row in results if row["score"].get("component_multiset_match_ignore_nodes")
        ),
        "component_multiset_match_with_undirected_nodes": sum(
            1 for row in results if row["score"].get("component_multiset_match_with_undirected_nodes")
        ),
        "avg_component_name_recall": _avg(results, "component_name_recall"),
        "avg_component_name_precision": _avg(results, "component_name_precision"),
        "avg_type_accuracy_on_common": _avg(results, "component_type_accuracy_on_common"),
        "avg_value_accuracy_on_common": _avg(results, "component_value_accuracy_on_common"),
        "avg_undirected_terminal_accuracy_on_common": _avg(results, "undirected_terminal_accuracy_on_common"),
    }


def print_summary(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    print("\nSUMMARY")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print(f"saved: {payload.get('output_path')} -> {summary.get('total')} samples")
    print_summary_explanations()


def print_summary_explanations() -> None:
    print("\n指标说明")
    print("build_success: build_netlist_node 成功生成可解析的 NetlistIR 的样例数；只表示格式可解析，不表示网表正确。")
    print("component_multiset_match_ignore_nodes: 整个样例的元件名、类型、值集合是否完全一致；忽略所有端点/节点连接。")
    print("component_multiset_match_with_undirected_nodes: 整个样例的元件名、类型、值、无向两端节点集合是否完全一致。")
    print("avg_component_name_recall: 平均元件名召回率，即 GT 中有多少元件名被预测出来。")
    print("avg_component_name_precision: 平均元件名精确率，即预测元件名中有多少存在于 GT。")
    print("avg_type_accuracy_on_common: 对同名元件统计类型前缀是否正确，例如 R/C/L/V。")
    print("avg_value_accuracy_on_common: 对同名元件统计值/参数 token 是否正确。")
    print("avg_undirected_terminal_accuracy_on_common: 对同名元件统计无向端点集合是否正确；电阻两端顺序交换不算错。")


def print_prompts(results: list[dict[str, Any]]) -> None:
    for row in sorted(results, key=lambda item: _case_key(Path(item["id"]))):
        prompt = row.get("build_prompt")
        if not prompt:
            print(f"\n===== {row['id']} 提示词 =====")
            print("未捕获提示词。请确认运行时传入了 --print-prompt，且该样例走的是 LLM build 路径。")
            continue
        print(f"\n===== {row['id']} 系统提示词 =====")
        print(prompt.get("system_prompt") or "")
        print(f"\n===== {row['id']} 用户文本提示 =====")
        print(prompt.get("human_text") or "")
        print(f"\n===== {row['id']} 图片输入 =====")
        print(f"{prompt.get('image_path')} ({prompt.get('image_media_type')}, {prompt.get('image_attached_as')})")


if __name__ == "__main__":
    raise SystemExit(main())
