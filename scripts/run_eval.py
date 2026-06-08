#!/usr/bin/env python3
"""Run CircuitSense evaluation with sample-level concurrency and one run dir."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "output" / "eval"
DEFAULT_DATA = PROJECT_ROOT / "data" / "circuitsense_synthetic_autovsr.json"
INFRA_MARKERS = (
    "429",
    "RateLimit",
    "rate limit",
    "速率限制",
    "timeout",
    "timed out",
    "Connection error",
    "APIConnectionError",
    "Access denied",
    "Arrearage",
    "overdue-payment",
    "connection aborted",
    "connection reset",
    "server disconnected",
    "remote protocol",
    "peer closed",
    "network",
    "502",
    "503",
    "504",
)


@dataclass
class SampleJob:
    index: int
    sample_id: str
    attempts: int = 0
    next_start_time: float = 0.0
    last_error: str = ""


@dataclass
class RunningJob:
    job: SampleJob
    proc: subprocess.Popen
    launched_at: float
    stdout_handle: Any


@dataclass
class EvalState:
    results: List[Dict[str, Any]] = field(default_factory=list)
    by_index: Dict[int, Dict[str, Any]] = field(default_factory=dict)


class EvalLogger:
    def __init__(self, run_dir: Path):
        self.path = run_dir / "eval.log"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _now_iso() -> str:
    return datetime.now().isoformat()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _contains_infra_error(text: str) -> bool:
    low = text.lower()
    return any(marker.lower() in low for marker in INFRA_MARKERS)


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _tail_failure_summary(path: Path, max_lines: int = 20) -> str:
    text = _read_text(path)
    if not text:
        return ""
    lines = text.splitlines()
    interesting = [
        line.strip()
        for line in lines
        if "Traceback" in line
        or "Error" in line
        or "Exception" in line
        or "ModuleNotFound" in line
        or "ImportError" in line
        or "Connection error" in line
    ]
    if interesting:
        return " | ".join(interesting[-max_lines:])
    return " | ".join(line.strip() for line in lines[-max_lines:])


def _sample_stem(index: int) -> str:
    return f"sample_{index:06d}"


def _sample_paths(run_dir: Path, index: int) -> Dict[str, Path]:
    stem = _sample_stem(index)
    return {
        "result": run_dir / "samples" / f"{stem}_result.json",
        "checkpoint": run_dir / "samples" / f"{stem}_result.checkpoint.json",
        "log": run_dir / "logs" / f"{stem}.log",
        "stdout": run_dir / "logs" / f"{stem}.stdout.log",
    }


def _create_run_dir(tag: str) -> Path:
    now = datetime.now()
    run_dir = OUTPUT_ROOT / now.strftime("%Y-%m-%d") / now.strftime("%H%M%S")
    if tag:
        run_dir = run_dir.with_name(f"{run_dir.name}_{tag}")
    counter = 1
    base = run_dir
    while run_dir.exists():
        run_dir = base.with_name(f"{base.name}_{counter:02d}")
        counter += 1
    return run_dir


def _load_dataset(data_path: Path) -> List[Dict[str, Any]]:
    data = _load_json(data_path)
    if isinstance(data, dict):
        rows = data.get("samples") or data.get("results")
    else:
        rows = data
    if not isinstance(rows, list):
        raise SystemExit(f"Unsupported dataset format: {data_path}")
    return rows


def _load_config() -> Dict[str, Any]:
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def _llm_meta() -> Dict[str, Any]:
    llm = (_load_config().get("llm") or {}).copy()
    api_key = str(llm.get("api_key") or "")
    if api_key:
        llm["api_key"] = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
    return {
        "provider": llm.get("provider"),
        "model": llm.get("model"),
        "base_url": llm.get("base_url"),
        "api_key": llm.get("api_key"),
    }


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _result_payload(
    *,
    data_path: Path,
    args: argparse.Namespace,
    dataset_total: int,
    selected_indices: List[int],
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    stats = _stats(results)
    return {
        "data_file": str(data_path),
        "max_retries": args.retry_attempts,
        "run_slice": {
            "dataset_total": dataset_total,
            "start_index": args.start,
            "end_index": args.end,
            "sample_indices": selected_indices,
        },
        "stats": stats,
        "metrics_summary": _metrics_summary(results),
        "results": sorted(results, key=lambda row: row.get("sample_index", -1)),
    }


def _stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "total": len(results),
        "success": sum(1 for row in results if row.get("success")),
        "failed": sum(1 for row in results if not row.get("success") and not row.get("infrastructure_error")),
        "infrastructure_error": sum(1 for row in results if row.get("infrastructure_error")),
    }
    for key in ("level", "source", "type", "task"):
        bucket: Dict[str, Dict[str, int]] = {}
        for row in results:
            label = str(row.get(key) or "unknown")
            bucket.setdefault(label, {"total": 0, "success": 0})
            bucket[label]["total"] += 1
            bucket[label]["success"] += int(bool(row.get("success")))
        stats[f"by_{key}"] = bucket
    return stats


def _metrics_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    total_duration = 0.0
    total_calls = 0
    by_stage: Dict[str, Dict[str, Any]] = {}
    for row in results:
        metrics = row.get("metrics") or {}
        total_duration += float(metrics.get("total_duration_seconds") or 0)
        toks = metrics.get("total_tokens") or {}
        if isinstance(toks, dict):
            for key in total_tokens:
                total_tokens[key] += int(toks.get(key) or 0)
        else:
            total_tokens["total_tokens"] += int(toks or 0)
            total_tokens["input_tokens"] += int(metrics.get("total_input_tokens") or 0)
            total_tokens["output_tokens"] += int(metrics.get("total_output_tokens") or 0)
        total_calls += int(metrics.get("total_llm_calls") or 0)
        for stage, data in (metrics.get("by_stage") or {}).items():
            if not isinstance(data, dict):
                continue
            cur = by_stage.setdefault(
                stage,
                {
                    "duration_seconds": 0.0,
                    "tokens": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    "llm_calls": 0,
                },
            )
            cur["duration_seconds"] += float(data.get("duration_seconds") or 0)
            cur["llm_calls"] += int(data.get("llm_calls") or 0)
            stage_tokens = data.get("tokens") or {}
            if isinstance(stage_tokens, dict):
                for key in cur["tokens"]:
                    cur["tokens"][key] += int(stage_tokens.get(key) or 0)
    return {
        "total_duration_seconds": total_duration,
        "total_tokens": total_tokens,
        "total_llm_calls": total_calls,
        "by_stage": by_stage,
    }


def _load_state(run_dir: Path) -> EvalState:
    source = run_dir / "results.checkpoint.json"
    if not source.exists():
        source = run_dir / "results.json"
    state = EvalState()
    if not source.exists():
        return state
    data = _load_json(source)
    for row in data.get("results", []):
        idx = row.get("sample_index")
        if idx is None:
            continue
        state.by_index[int(idx)] = row
    state.results = list(state.by_index.values())
    return state


def _write_checkpoint(
    run_dir: Path,
    *,
    data_path: Path,
    args: argparse.Namespace,
    dataset_total: int,
    selected_indices: List[int],
    state: EvalState,
) -> None:
    payload = _result_payload(
        data_path=data_path,
        args=args,
        dataset_total=dataset_total,
        selected_indices=selected_indices,
        results=list(state.by_index.values()),
    )
    payload["timestamp"] = _now_iso()
    _write_json_atomic(run_dir / "results.checkpoint.json", payload)


def _write_meta(run_dir: Path, meta: Dict[str, Any]) -> None:
    _write_json_atomic(run_dir / "run_meta.json", meta)


def _build_meta(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    data_path: Path,
    dataset_total: int,
    status: str,
    start_time: str,
    end_time: Optional[str],
    state: EvalState,
) -> Dict[str, Any]:
    stats = _stats(list(state.by_index.values()))
    return {
        "start_time": start_time,
        "end_time": end_time,
        "status": status,
        "args": _jsonable(vars(args)),
        "data_path": str(data_path),
        "dataset_total": dataset_total,
        "start": args.start,
        "end": args.end,
        "jobs": args.jobs,
        "llm": _llm_meta(),
        "git_commit": _git_commit(),
        "total_completed": stats["total"],
        "success": stats["success"],
        "failed": stats["failed"],
        "infrastructure_error": stats["infrastructure_error"],
        "output_files": {
            "run_meta": str(run_dir / "run_meta.json"),
            "eval_log": str(run_dir / "eval.log"),
            "results": str(run_dir / "results.json"),
            "checkpoint": str(run_dir / "results.checkpoint.json"),
            "summary": str(run_dir / "summary.md"),
            "paper_comparison": str(run_dir / "paper_comparison.md"),
            "logs": str(run_dir / "logs"),
            "samples": str(run_dir / "samples"),
        },
    }


def _clean_sample_outputs(run_dir: Path, index: int) -> None:
    for path in _sample_paths(run_dir, index).values():
        if path.exists():
            path.unlink()


def _command_for(run_dir: Path, data_path: Path, index: int) -> List[str]:
    paths = _sample_paths(run_dir, index)
    return [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        "-t",
        "batch",
        "--data",
        str(data_path),
        "--output",
        str(paths["result"]),
        "--log",
        str(paths["log"]),
        "--start-index",
        str(index),
        "--end-index",
        str(index + 1),
    ]


def _launch(job: SampleJob, run_dir: Path, data_path: Path, env: Dict[str, str]) -> RunningJob:
    job.last_error = ""
    paths = _sample_paths(run_dir, job.index)
    paths["stdout"].parent.mkdir(parents=True, exist_ok=True)
    paths["result"].parent.mkdir(parents=True, exist_ok=True)
    stdout = paths["stdout"].open("a", encoding="utf-8")
    cmd = _command_for(run_dir, data_path, job.index)
    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=stdout,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        start_new_session=True,
    )
    return RunningJob(job=job, proc=proc, launched_at=time.time(), stdout_handle=stdout)


def _terminate(proc: subprocess.Popen, timeout: float = 10.0) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()
        proc.wait(timeout=timeout)


def _latest_progress_time(run_dir: Path, index: int, fallback: float) -> float:
    mtimes = [p.stat().st_mtime for p in _sample_paths(run_dir, index).values() if p.exists()]
    return max(mtimes) if mtimes else fallback


def _extract_sample_result(run_dir: Path, job: SampleJob, dataset_row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    result_path = _sample_paths(run_dir, job.index)["result"]
    if not result_path.exists():
        return None
    data = _load_json(result_path)
    rows = data.get("results", [])
    if not rows:
        return None
    row = rows[0]
    row["sample_index"] = job.index
    row["sample_id"] = row.get("id") or job.sample_id
    row["attempts"] = job.attempts
    row["sample_result_path"] = str(result_path)
    row.setdefault("id", job.sample_id)
    for key in ("source", "level", "type", "task", "question"):
        row.setdefault(key, dataset_row.get(key))
    return row


def _sample_has_infra_error(run_dir: Path, job: SampleJob, row: Optional[Dict[str, Any]], code: int) -> bool:
    if code == 75:
        return True
    if row and _contains_infra_error(str(row.get("error") or "")):
        return True
    paths = _sample_paths(run_dir, job.index)
    text = "\n".join(_read_text(paths[key]) for key in ("stdout", "log"))
    return _contains_infra_error(text) and code != 0


def _process_failure_summary(run_dir: Path, job: SampleJob, code: int) -> str:
    paths = _sample_paths(run_dir, job.index)
    summary = _tail_failure_summary(paths["stdout"]) or _tail_failure_summary(paths["log"])
    if summary:
        return f"process exited with code {code}: {summary}"
    return f"process exited with code {code}"


def _infra_result(job: SampleJob, dataset_row: Dict[str, Any], error: str) -> Dict[str, Any]:
    return {
        "id": job.sample_id,
        "sample_id": job.sample_id,
        "sample_index": job.index,
        "image_path": dataset_row.get("image_path"),
        "question": dataset_row.get("question"),
        "expected_answer": dataset_row.get("answer"),
        "predicted_answer": None,
        "success": False,
        "pipeline_success": False,
        "symbolic_equivalent": None,
        "source": dataset_row.get("source"),
        "level": dataset_row.get("level"),
        "type": dataset_row.get("type"),
        "task": dataset_row.get("task"),
        "analysis_type": dataset_row.get("task"),
        "error": error,
        "infrastructure_error": True,
        "attempts": job.attempts,
    }


def _backoff(args: argparse.Namespace, attempt: int) -> float:
    delay = min(args.retry_max_sleep, args.retry_base_sleep * (2 ** max(0, attempt - 1)))
    return delay


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _progress_line(
    *,
    state: EvalState,
    selected_total: int,
    pending: List[SampleJob],
    running: List[RunningJob],
    started_at: float,
) -> str:
    completed = len(state.by_index)
    stats = _stats(list(state.by_index.values()))
    pct = (completed / selected_total * 100.0) if selected_total else 100.0
    width = 28
    filled = int(width * completed / selected_total) if selected_total else width
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.time() - started_at
    eta = "unknown"
    if completed:
        remaining = selected_total - completed
        eta = _format_duration(elapsed / completed * remaining)
    running_labels = ",".join(f"{item.job.index}(try{item.job.attempts})" for item in running) or "-"
    retry_waiting = sum(1 for job in pending if job.next_start_time > time.time())
    return (
        f"progress [{bar}] {completed}/{selected_total} {pct:.1f}% | "
        f"ok={stats['success']} fail={stats['failed']} infra={stats['infrastructure_error']} | "
        f"running={running_labels} pending={len(pending)} retry_wait={retry_waiting} | "
        f"elapsed={_format_duration(elapsed)} eta={eta}"
    )


def _finalize(run_dir: Path, payload: Dict[str, Any]) -> None:
    results_path = run_dir / "results.json"
    _write_json_atomic(results_path, payload)
    subprocess.run(
        [sys.executable, "scripts/summarize_results.py", str(results_path), "--output", str(run_dir / "summary.md")],
        cwd=PROJECT_ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/compare_paper_metrics.py",
            str(results_path),
            "--output",
            str(run_dir / "paper_comparison.md"),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--tag", default="")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--min-interval", type=float, default=20.0)
    parser.add_argument("--retry-attempts", type=int, default=8)
    parser.add_argument("--retry-base-sleep", type=float, default=45.0)
    parser.add_argument("--retry-max-sleep", type=float, default=240.0)
    parser.add_argument("--stall-timeout", type=float, default=600.0)
    parser.add_argument("--progress-interval", type=float, default=30.0)
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--rerun-all", action="store_true")
    parser.add_argument(
        "--drop-infra-failures",
        action="store_true",
        help="When resuming, remove completed infrastructure-error rows from the checkpoint so they are retried.",
    )
    parser.add_argument("--summarize", action="store_true", help="Summarize an existing complete run dir and exit.")
    parser.add_argument("--merge", action="store_true", help="Alias for --summarize on an existing run dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.data if args.data.is_absolute() else PROJECT_ROOT / args.data
    dataset = _load_dataset(data_path)
    dataset_total = len(dataset)
    args.end = args.end if args.end is not None else dataset_total
    if args.start < 0 or args.end > dataset_total or args.start >= args.end:
        raise SystemExit(f"Invalid range: start={args.start}, end={args.end}, dataset_total={dataset_total}")
    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1")

    run_dir = args.run_dir if args.run_dir else _create_run_dir(args.tag)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    for child in ("logs", "samples"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    progress = EvalLogger(run_dir)

    state = _load_state(run_dir)
    selected_indices = list(range(args.start, args.end))
    start_time = _now_iso()
    started_at = time.time()

    if args.summarize or args.merge:
        progress.log(f"summarizing run_dir={run_dir}")
        payload = _result_payload(
            data_path=data_path,
            args=args,
            dataset_total=dataset_total,
            selected_indices=selected_indices,
            results=list(state.by_index.values()),
        )
        _finalize(run_dir, payload)
        _write_meta(
            run_dir,
            _build_meta(
                args=args,
                run_dir=run_dir,
                data_path=data_path,
                dataset_total=dataset_total,
                status="summarized",
                start_time=start_time,
                end_time=_now_iso(),
                state=state,
            ),
        )
        progress.log(f"summarized results={len(state.by_index)} summary={run_dir / 'summary.md'}")
        print(run_dir, flush=True)
        return

    if args.rerun_all:
        state = EvalState()
        for idx in selected_indices:
            _clean_sample_outputs(run_dir, idx)
    elif args.rerun_failed:
        for idx, row in list(state.by_index.items()):
            if idx in selected_indices and not row.get("success"):
                state.by_index.pop(idx, None)
                _clean_sample_outputs(run_dir, idx)
        state.results = list(state.by_index.values())
    elif args.drop_infra_failures:
        dropped = 0
        for idx, row in list(state.by_index.items()):
            error_text = str(row.get("error") or "")
            if idx in selected_indices and (row.get("infrastructure_error") or _contains_infra_error(error_text)):
                state.by_index.pop(idx, None)
                _clean_sample_outputs(run_dir, idx)
                dropped += 1
        state.results = list(state.by_index.values())

    pending = [
        SampleJob(index=idx, sample_id=str(dataset[idx].get("id") or dataset[idx].get("sample_id") or idx))
        for idx in selected_indices
        if idx not in state.by_index
    ]
    skipped = len(selected_indices) - len(pending)

    env = os.environ.copy()
    env["AUTOVSR_LLM_MIN_INTERVAL"] = str(args.min_interval)
    env["AUTOVSR_LLM_RETRY_ATTEMPTS"] = str(args.retry_attempts)
    env["AUTOVSR_LLM_RETRY_BASE_SLEEP"] = str(args.retry_base_sleep)
    env["AUTOVSR_LLM_RETRY_MAX_SLEEP"] = str(args.retry_max_sleep)
    env.setdefault("AUTOVSR_LLM_LOCK_FILE", str(run_dir / "autovsr_llm_rate.lock"))
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(PROJECT_ROOT)
        if not existing_pythonpath
        else str(PROJECT_ROOT) + os.pathsep + existing_pythonpath
    )

    _write_checkpoint(run_dir, data_path=data_path, args=args, dataset_total=dataset_total, selected_indices=selected_indices, state=state)
    _write_meta(
        run_dir,
        _build_meta(
            args=args,
            run_dir=run_dir,
            data_path=data_path,
            dataset_total=dataset_total,
            status="running",
            start_time=start_time,
            end_time=None,
            state=state,
        ),
    )
    progress.log(
        f"run_dir={run_dir} range={args.start}:{args.end} total={len(selected_indices)} "
        f"jobs={args.jobs} skipped_completed={skipped} pending={len(pending)}"
    )
    progress.log(
        f"outputs checkpoint={run_dir / 'results.checkpoint.json'} results={run_dir / 'results.json'} "
        f"log={run_dir / 'eval.log'}"
    )
    progress.log(
        _progress_line(
            state=state,
            selected_total=len(selected_indices),
            pending=pending,
            running=[],
            started_at=started_at,
        )
    )

    running: List[RunningJob] = []
    last_progress_log = 0.0
    try:
        while pending or running:
            now = time.time()
            while pending and len(running) < args.jobs and pending[0].next_start_time <= now:
                job = pending.pop(0)
                job.attempts += 1
                running_item = _launch(job, run_dir, data_path, env)
                running.append(running_item)
                progress.log(
                    f"launch sample_index={job.index} sample_id={job.sample_id} "
                    f"attempt={job.attempts}/{args.retry_attempts} pid={running_item.proc.pid}"
                )

            if not running:
                sleep_until = min((job.next_start_time for job in pending), default=time.time())
                if time.time() - last_progress_log >= args.progress_interval:
                    progress.log(
                        _progress_line(
                            state=state,
                            selected_total=len(selected_indices),
                            pending=pending,
                            running=running,
                            started_at=started_at,
                        )
                    )
                    last_progress_log = time.time()
                time.sleep(max(1.0, min(5.0, sleep_until - time.time())))
                continue

            time.sleep(2)
            if time.time() - last_progress_log >= args.progress_interval:
                progress.log(
                    _progress_line(
                        state=state,
                        selected_total=len(selected_indices),
                        pending=pending,
                        running=running,
                        started_at=started_at,
                    )
                )
                last_progress_log = time.time()
            for item in list(running):
                job = item.job
                code = item.proc.poll()
                if code is None and args.stall_timeout > 0:
                    stalled = time.time() - _latest_progress_time(run_dir, job.index, item.launched_at)
                    if stalled > args.stall_timeout:
                        job.last_error = f"stall timeout after {stalled:.0f}s"
                        progress.log(f"stall sample_index={job.index} sample_id={job.sample_id}: {job.last_error}")
                        _terminate(item.proc)
                        code = item.proc.poll()

                if code is None:
                    continue

                running.remove(item)
                item.stdout_handle.close()
                dataset_row = dataset[job.index]
                row = _extract_sample_result(run_dir, job, dataset_row)
                infra = _sample_has_infra_error(run_dir, job, row, int(code or 0)) or bool(job.last_error)

                if infra:
                    error = job.last_error or (row or {}).get("error") or _process_failure_summary(run_dir, job, int(code or 0))
                    if job.attempts < args.retry_attempts:
                        job.last_error = str(error)
                        job.next_start_time = time.time() + _backoff(args, job.attempts)
                        pending.append(job)
                        progress.log(
                            f"retry sample_index={job.index} sample_id={job.sample_id} "
                            f"attempt={job.attempts}/{args.retry_attempts} reason={error!s} "
                            f"next_in={_format_duration(job.next_start_time - time.time())}"
                        )
                        continue
                    row = _infra_result(job, dataset_row, str(error))
                elif row is None:
                    if job.attempts < args.retry_attempts:
                        job.last_error = _process_failure_summary(run_dir, job, int(code or 0))
                        job.next_start_time = time.time() + _backoff(args, job.attempts)
                        pending.append(job)
                        progress.log(
                            f"retry sample_index={job.index} sample_id={job.sample_id} "
                            f"attempt={job.attempts}/{args.retry_attempts} reason={job.last_error} "
                            f"next_in={_format_duration(job.next_start_time - time.time())}"
                        )
                        continue
                    row = _infra_result(job, dataset_row, f"missing result after process exit code {code}")

                state.by_index[job.index] = row
                state.results = list(state.by_index.values())
                status = "infra" if row.get("infrastructure_error") else ("ok" if row.get("success") else "fail")
                progress.log(
                    f"done sample_index={job.index} sample_id={job.sample_id} status={status} "
                    f"attempts={job.attempts} completed={len(state.by_index)}/{len(selected_indices)}"
                )
                _write_checkpoint(
                    run_dir,
                    data_path=data_path,
                    args=args,
                    dataset_total=dataset_total,
                    selected_indices=selected_indices,
                    state=state,
                )
                _write_meta(
                    run_dir,
                    _build_meta(
                        args=args,
                        run_dir=run_dir,
                        data_path=data_path,
                        dataset_total=dataset_total,
                        status="running",
                        start_time=start_time,
                        end_time=None,
                        state=state,
                    ),
                )

        payload = _result_payload(
            data_path=data_path,
            args=args,
            dataset_total=dataset_total,
            selected_indices=selected_indices,
            results=list(state.by_index.values()),
        )
        _finalize(run_dir, payload)
        status = "completed_with_infrastructure_errors" if payload["stats"]["infrastructure_error"] else "completed"
        _write_meta(
            run_dir,
            _build_meta(
                args=args,
                run_dir=run_dir,
                data_path=data_path,
                dataset_total=dataset_total,
                status=status,
                start_time=start_time,
                end_time=_now_iso(),
                state=state,
            ),
        )
        progress.log(
            f"finished status={status} total={payload['stats']['total']} "
            f"ok={payload['stats']['success']} fail={payload['stats']['failed']} "
            f"infra={payload['stats']['infrastructure_error']} elapsed={_format_duration(time.time() - started_at)}"
        )
        progress.log(f"summary={run_dir / 'summary.md'} paper_comparison={run_dir / 'paper_comparison.md'}")
        print(run_dir, flush=True)
    finally:
        for item in running:
            item.stdout_handle.close()


if __name__ == "__main__":
    main()
