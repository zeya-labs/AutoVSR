#!/usr/bin/env python3
"""Evaluate image-to-netlist generation on CircuitSense synthetic levels.

This runs only the Netlist build node, using the LLM configured in
`config/config.yaml`, and compares the generated netlist against each sample's
`q*_netlist.txt`.
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import html
import io
import json
import re
import shutil
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


def _is_extra_step_self_source(line: str, expected_names: set[str]) -> bool:
    parts = line.split()
    if len(parts) != 5:
        return False
    name = parts[0]
    if name in expected_names:
        return False
    prefix = re.match(r"[A-Za-z]+", name)
    kind = prefix.group(0).upper() if prefix else name[0].upper()
    return kind in {"I", "V"} and parts[3].lower() == "step" and parts[4] == name


def _drop_extra_step_self_sources(predicted: str, expected: str) -> str:
    expected_names = set(_component_map(expected))
    kept = []
    for line in _netlist_lines(predicted):
        if _is_extra_step_self_source(line, expected_names):
            continue
        kept.append(line)
    return "\n".join(kept)


def _excluded_by_extra_step_self_source(row: dict[str, Any]) -> bool:
    score = row.get("score") or {}
    if score.get("component_multiset_match_with_undirected_nodes"):
        return False
    expected = row.get("expected_netlist") or ""
    predicted = row.get("predicted_netlist") or ""
    cleaned = _drop_extra_step_self_sources(predicted, expected)
    if cleaned == "\n".join(_netlist_lines(predicted)):
        return False
    return bool(_score_netlist(cleaned, expected).get("component_multiset_match_with_undirected_nodes"))


def _component_map_for_report(text: str) -> dict[str, tuple[str, tuple[str, ...], tuple[str, ...], str]]:
    comps: dict[str, tuple[str, tuple[str, ...], tuple[str, ...], str]] = {}
    for line in _netlist_lines(text):
        parts = line.split()
        if len(parts) < 3:
            continue
        name = parts[0]
        prefix = re.match(r"[A-Za-z]+", name)
        kind = prefix.group(0).upper() if prefix else name[0].upper()
        nodes = tuple(parts[1:3])
        args = tuple(parts[3:])
        comps[name] = (kind, nodes, args, line)
    return comps


def _component_signature(text: str, *, ignore_nodes: bool = False) -> Counter:
    sig = Counter()
    for name, (kind, nodes, args) in _component_map(text).items():
        key = (name, kind, args) if ignore_nodes else (name, kind, tuple(sorted(nodes)), args)
        sig[key] += 1
    return sig


def _is_wrong(row: dict[str, Any]) -> bool:
    score = row.get("score") or {}
    return not row.get("success") or not score.get("component_multiset_match_with_undirected_nodes")


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


def _report_dir(output_path: Path) -> Path:
    if output_path.name == "results.json":
        return output_path.parent
    return output_path.with_suffix("")


def _relative_link(target: str | Path, base_dir: Path) -> str:
    try:
        return Path(target).resolve().relative_to(base_dir.resolve()).as_posix()
    except Exception:
        try:
            return Path(target).resolve().as_uri()
        except Exception:
            return str(target)


def _component_diff_rows(expected: str, predicted: str) -> list[dict[str, str]]:
    exp = _component_map_for_report(expected)
    pred = _component_map_for_report(predicted)
    rows: list[dict[str, str]] = []
    for name in sorted(set(exp) | set(pred), key=lambda item: (_case_key(Path(item)), item)):
        if name not in pred:
            rows.append({"name": name, "status": "missing", "expected": exp[name][3], "predicted": ""})
            continue
        if name not in exp:
            rows.append({"name": name, "status": "extra", "expected": "", "predicted": pred[name][3]})
            continue
        exp_kind, exp_nodes, exp_args, exp_line = exp[name]
        pred_kind, pred_nodes, pred_args, pred_line = pred[name]
        problems = []
        if exp_kind != pred_kind:
            problems.append("type")
        if tuple(sorted(exp_nodes)) != tuple(sorted(pred_nodes)):
            problems.append("nodes")
        elif exp_nodes != pred_nodes:
            problems.append("polarity")
        if exp_args != pred_args:
            problems.append("value")
        if problems:
            rows.append(
                {
                    "name": name,
                    "status": "+".join(problems),
                    "expected": exp_line,
                    "predicted": pred_line,
                }
            )
    return rows


ERROR_TAG_LABELS = {
    "excluded-step": "excluded step",
    "extra": "extra",
    "missing": "missing",
    "nodes": "nodes",
    "polarity": "polarity",
    "value": "value",
    "type": "type",
    "build": "build",
    "other": "other",
}


ERROR_TAG_ORDER = {
    "excluded-step": 0,
    "missing": 1,
    "extra": 2,
    "nodes": 3,
    "polarity": 4,
    "value": 5,
    "type": 6,
    "build": 7,
    "other": 8,
}


def _case_error_tags(row: dict[str, Any]) -> list[str]:
    tags = set()
    if _excluded_by_extra_step_self_source(row):
        tags.add("excluded-step")
    if not row.get("success"):
        tags.add("build")
    for diff in _component_diff_rows(row.get("expected_netlist") or "", row.get("predicted_netlist") or ""):
        for status in str(diff.get("status") or "").split("+"):
            if status:
                tags.add(status)
    if not tags:
        tags.add("other")
    return sorted(tags, key=lambda item: (ERROR_TAG_ORDER.get(item, 99), item))


def _render_error_chips(tags: list[str], *, compact: bool = False) -> str:
    chip_class = "error-chip compact" if compact else "error-chip"
    return "".join(
        f'<span class="{chip_class} {html.escape(tag)}">{html.escape(ERROR_TAG_LABELS.get(tag, tag))}</span>'
        for tag in tags
    )


def _unified_diff(expected: str, predicted: str) -> str:
    return "\n".join(
        difflib.unified_diff(
            _netlist_lines(expected),
            _netlist_lines(predicted),
            fromfile="gt",
            tofile="pred",
            lineterm="",
        )
    )


def _safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "item"


def _write_tiled_trace_assets(row: dict[str, Any], case_out: Path, image_src: Path) -> dict[str, str]:
    repair = row.get("repair") or {}
    if repair.get("method") != "tiled":
        return {}

    try:
        from PIL import Image
    except Exception:
        return {}

    if not image_src.exists():
        return {}

    tile_images: dict[str, str] = {}
    tiles_out = case_out / "tiles"
    tiles_out.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(image_src) as image:
            for idx, item in enumerate(repair.get("raw_tile_responses") or [], start=1):
                tile = item.get("tile") or {}
                label = str(tile.get("label") or f"tile_{idx}")
                crop = tile.get("crop") or []
                if not isinstance(crop, (list, tuple)) or len(crop) != 4:
                    continue
                left, top, right, bottom = [int(round(float(value))) for value in crop]
                left = max(0, min(left, image.width))
                top = max(0, min(top, image.height))
                right = max(left + 1, min(right, image.width))
                bottom = max(top + 1, min(bottom, image.height))
                tile_path = tiles_out / f"{idx:02d}_{_safe_filename(label)}.png"
                image.crop((left, top, right, bottom)).save(tile_path)
                tile_images[label] = str(tile_path)
    except Exception:
        return tile_images
    return tile_images


def _write_case_report(row: dict[str, Any], case_dir: Path, out_dir: Path) -> dict[str, Any]:
    case_out = out_dir / "cases" / row["id"]
    case_out.mkdir(parents=True, exist_ok=True)
    expected = row.get("expected_netlist") or ""
    predicted = row.get("predicted_netlist") or ""
    (case_out / "gt.netlist").write_text(expected + "\n", encoding="utf-8")
    (case_out / "pred.netlist").write_text(predicted + "\n", encoding="utf-8")
    (case_out / "diff.patch").write_text(_unified_diff(expected, predicted) + "\n", encoding="utf-8")

    image_src = Path(row.get("image_path") or case_dir / f"{row['id']}_image.png")
    if image_src.exists():
        image_dst = case_out / image_src.name
        if not image_dst.exists() or image_dst.stat().st_mtime < image_src.stat().st_mtime:
            shutil.copy2(image_src, image_dst)
    else:
        image_dst = image_src

    tile_images = _write_tiled_trace_assets(row, case_out, image_src)

    rows = _component_diff_rows(expected, predicted)
    md_lines = [
        f"# {row['id']}",
        "",
        f"- success: `{row.get('success')}`",
        f"- error: `{row.get('error')}`",
        f"- image: `{image_src}`",
        f"- question: {row.get('question') or ''}",
        "",
        "## Component Differences",
        "",
        "| component | status | gt | pred |",
        "| --- | --- | --- | --- |",
    ]
    for diff_row in rows:
        md_lines.append(
            "| {name} | {status} | `{expected}` | `{predicted}` |".format(
                name=diff_row["name"],
                status=diff_row["status"],
                expected=diff_row["expected"].replace("|", "\\|"),
                predicted=diff_row["predicted"].replace("|", "\\|"),
            )
        )
    md_lines.extend(
        [
            "",
            "## GT",
            "",
            "```netlist",
            expected,
            "```",
            "",
            "## Pred",
            "",
            "```netlist",
            predicted,
            "```",
            "",
            "## Unified Diff",
            "",
            "```diff",
            _unified_diff(expected, predicted),
            "```",
        ]
    )
    (case_out / "README.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return {
        "case_dir": case_out.name,
        "readme": str(case_out / "README.md"),
        "image": str(image_dst),
        "tile_images": tile_images,
    }


def _write_eval_report(payload: dict[str, Any], selected: list[Path]) -> None:
    output_path = Path(payload["output_path"])
    out_dir = _report_dir(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_by_id = {p.name: p for p in selected}
    rows = sorted(payload.get("results") or [], key=lambda item: _case_key(Path(item["id"])))
    wrong_rows = [row for row in rows if _is_wrong(row)]
    display_rows = wrong_rows
    node_only = [
        row
        for row in wrong_rows
        if row.get("success") and (row.get("score") or {}).get("component_multiset_match_ignore_nodes")
    ]
    component_wrong = [row for row in wrong_rows if row not in node_only]

    case_links: dict[str, dict[str, Any]] = {}
    for report_subdir in ("cases", "wrong_cases"):
        report_dir = out_dir / report_subdir
        if report_dir.exists():
            shutil.rmtree(report_dir)
    (out_dir / "cases").mkdir(parents=True, exist_ok=True)
    for row in display_rows:
        case_dir = selected_by_id.get(row["id"], Path(row.get("image_path", "")).parent)
        case_links[row["id"]] = _write_case_report(row, case_dir, out_dir)

    manifest = {
        "json": str(output_path),
        "report_dir": str(out_dir),
        "display_count": len(display_rows),
        "wrong_count": len(wrong_rows),
        "node_only_count": len(node_only),
        "component_wrong_count": len(component_wrong),
        "display_ids": [row["id"] for row in display_rows],
        "wrong_ids": [row["id"] for row in wrong_rows],
        "node_only_ids": [row["id"] for row in node_only],
        "component_wrong_ids": [row["id"] for row in component_wrong],
        "error_tags_by_id": {row["id"]: _case_error_tags(row) for row in display_rows},
    }
    (out_dir / "wrong_cases.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "wrong_ids.txt").write_text("\n".join(manifest["wrong_ids"]) + ("\n" if wrong_rows else ""), encoding="utf-8")

    html_text = _render_html_report(payload, display_rows, wrong_rows, node_only, component_wrong, case_links, out_dir)
    (out_dir / "index.html").write_text(html_text, encoding="utf-8")


def _render_tiled_trace(
    repair: dict[str, Any],
    tile_images: dict[str, str],
    full_image_rel: str,
    out_dir: Path,
) -> str:
    if repair.get("method") != "tiled":
        return ""
    tile_items = []
    for item in repair.get("raw_tile_responses") or []:
        tile = item.get("tile") or {}
        label = str(tile.get("label") or "tile")
        image_src = tile_images.get(label) or tile_images.get(str(tile.get("label") or ""))
        image_rel = _relative_link(image_src, out_dir) if image_src else ""
        image_html = f'<img src="{html.escape(image_rel)}" alt="{html.escape(label)} tile">' if image_rel else (
            f"<p class=\"muted\">tile image unavailable; crop={html.escape(str(tile.get('crop') or ''))}</p>"
        )
        tile_items.append(
            f"""
            <section class="trace-tile">
              <h4>{html.escape(label)} crop {html.escape(str(tile.get('crop') or ''))}</h4>
              <div class="trace-grid">
                <div class="image-wrap">{image_html}</div>
                <div>
                  <h5>Tile system prompt</h5>
                  <pre>{html.escape(str(item.get('system_prompt') or ''))}</pre>
                  <h5>Tile human prompt</h5>
                  <pre>{html.escape(str(item.get('human_text') or ''))}</pre>
                  <h5>Tile response</h5>
                  <pre>{html.escape(str(item.get('raw_response') or ''))}</pre>
                </div>
              </div>
            </section>
            """
        )
    merge_prompt = str(repair.get("merge_human_text") or "")
    merge_system = str(repair.get("merge_system_prompt") or "")
    merge_response = str(repair.get("raw_response") or "")
    structured = json.dumps(repair.get("structured") or {}, indent=2, ensure_ascii=False)
    return f"""
    <details class="trace">
      <summary>Tiled VLM trace: tile prompts, tile images, tile responses, merge prompt, merge response</summary>
      {''.join(tile_items) or '<p class="muted">No tile trace saved for this row.</p>'}
      <section class="trace-merge">
        <h4>Merge full image</h4>
        <div class="image-wrap merge-image"><img src="{html.escape(full_image_rel)}" alt="full image for merge"></div>
        <h4>Merge system prompt</h4>
        <pre>{html.escape(merge_system)}</pre>
        <h4>Merge human prompt</h4>
        <pre>{html.escape(merge_prompt)}</pre>
        <h4>Merge response</h4>
        <pre>{html.escape(merge_response)}</pre>
        <h4>Parsed structured graph</h4>
        <pre>{html.escape(structured)}</pre>
      </section>
    </details>
    """


def _render_html_report(
    payload: dict[str, Any],
    display_rows: list[dict[str, Any]],
    wrong_rows: list[dict[str, Any]],
    node_only: list[dict[str, Any]],
    component_wrong: list[dict[str, Any]],
    case_links: dict[str, dict[str, Any]],
    out_dir: Path,
) -> str:
    summary = payload.get("summary") or {}
    total = int(summary.get("total") or 0)
    raw_total = int(summary.get("raw_total") or total)
    excluded = int(summary.get("excluded_extra_step_self_source") or 0)
    strict_ok = int(summary.get("component_multiset_match_with_undirected_nodes") or 0)
    ignore_ok = int(summary.get("component_multiset_match_ignore_nodes") or 0)
    strict_pct = (strict_ok / total * 100) if total else 0
    ignore_pct = (ignore_ok / total * 100) if total else 0

    def metric_card(label: str, value: str, detail: str = "") -> str:
        return (
            '<div class="metric">'
            f"<div>{html.escape(label)}</div>"
            f"<strong>{html.escape(value)}</strong>"
            f"<span>{html.escape(detail)}</span>"
            "</div>"
        )

    def case_section(row: dict[str, Any]) -> str:
        score = row.get("score") or {}
        links = case_links.get(row["id"], {})
        error_tags = _case_error_tags(row)
        error_chips = _render_error_chips(error_tags)
        image_rel = _relative_link(links.get("image") or row.get("image_path") or "", out_dir)
        readme_rel = _relative_link(links.get("readme") or "", out_dir)
        repair = row.get("repair") or {}
        tiled_trace = _render_tiled_trace(repair, links.get("tile_images") or {}, image_rel, out_dir)
        diffs = _component_diff_rows(row.get("expected_netlist") or "", row.get("predicted_netlist") or "")
        diff_rows = "\n".join(
            "<tr>"
            f"<td>{html.escape(item['name'])}</td>"
            f"<td><span class=\"tag {html.escape(item['status'].split('+')[0])}\">{html.escape(item['status'])}</span></td>"
            f"<td><code>{html.escape(item['expected'])}</code></td>"
            f"<td><code>{html.escape(item['predicted'])}</code></td>"
            "</tr>"
            for item in diffs
        )
        if not diff_rows:
            diff_rows = '<tr><td colspan="4">No component-level differences detected by name.</td></tr>'
        raw_diff = _unified_diff(row.get("expected_netlist") or "", row.get("predicted_netlist") or "")
        return f"""
        <section class="case" id="{html.escape(row['id'])}">
          <div class="case-head">
            <div>
              <h2>{html.escape(row['id'])}</h2>
              <div class="case-tags">{error_chips}</div>
              <p>{html.escape(row.get('question') or '')}</p>
            </div>
            <a href="{html.escape(readme_rel)}">files</a>
          </div>
          <div class="case-grid">
            <div class="image-wrap"><img src="{html.escape(image_rel)}" alt="{html.escape(row['id'])} schematic"></div>
            <div class="score-panel">
              <div><b>success</b><span>{html.escape(str(row.get('success')))}</span></div>
              <div><b>error</b><span>{html.escape(str(row.get('error')))}</span></div>
              <div><b>components</b><span>{score.get('expected_components')} GT / {score.get('predicted_components')} pred</span></div>
              <div><b>node acc</b><span>{float(score.get('undirected_terminal_accuracy_on_common') or 0):.3f}</span></div>
              <div><b>ignore nodes</b><span>{html.escape(str(score.get('component_multiset_match_ignore_nodes')))}</span></div>
            </div>
          </div>
          <table>
            <thead><tr><th>component</th><th>diff</th><th>GT</th><th>Pred</th></tr></thead>
            <tbody>{diff_rows}</tbody>
          </table>
          <div class="netlists">
            <div><h3>GT</h3><pre>{html.escape(row.get('expected_netlist') or '')}</pre></div>
            <div><h3>Pred</h3><pre>{html.escape(row.get('predicted_netlist') or '')}</pre></div>
          </div>
          <details><summary>Unified diff</summary><pre>{html.escape(raw_diff)}</pre></details>
          {tiled_trace}
        </section>
        """

    case_nav = " ".join(
        f'<a class="nav-case {html.escape(_case_error_tags(row)[0])}" href="#{html.escape(row["id"])}">'
        f'<span>{html.escape(row["id"])}</span>{_render_error_chips(_case_error_tags(row), compact=True)}</a>'
        for row in display_rows
    )
    case_html = "\n".join(case_section(row) for row in display_rows)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Netlist Eval Report</title>
  <style>
    :root {{ color-scheme: light; --bg:#f6f7f9; --panel:#ffffff; --ink:#17202a; --muted:#68717d; --line:#d9dee6; --bad:#b42318; --warn:#9a6700; --ok:#116329; --blue:#2451a6; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; font:14px/1.45 system-ui, -apple-system, Segoe UI, sans-serif; background:var(--bg); color:var(--ink); }}
    header {{ padding:24px 28px 12px; background:var(--panel); border-bottom:1px solid var(--line); position:sticky; top:0; z-index:2; }}
    h1 {{ margin:0 0 14px; font-size:24px; letter-spacing:0; }}
    h2 {{ margin:0; font-size:20px; }}
    h3 {{ margin:0 0 8px; font-size:14px; }}
    .metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:10px; margin-bottom:12px; }}
    .metric {{ background:#f9fafb; border:1px solid var(--line); border-radius:8px; padding:10px 12px; }}
    .metric div {{ color:var(--muted); font-size:12px; }}
    .metric strong {{ display:block; font-size:22px; margin:2px 0; }}
    .metric span {{ color:var(--muted); font-size:12px; }}
    nav {{ display:flex; gap:6px; flex-wrap:wrap; max-height:98px; overflow:auto; }}
    nav a, .case-head a {{ color:var(--blue); text-decoration:none; border:1px solid var(--line); background:#fff; border-radius:6px; padding:3px 7px; }}
    nav a.nav-case {{ display:flex; align-items:center; gap:5px; color:var(--ink); border-left-width:5px; }}
    .nav-case.excluded-step {{ border-left-color:#7c3aed; }}
    .nav-case.missing {{ border-left-color:#b42318; }}
    .nav-case.extra {{ border-left-color:#c2410c; }}
    .nav-case.nodes {{ border-left-color:#9a6700; }}
    .nav-case.polarity {{ border-left-color:#a16207; }}
    .nav-case.value {{ border-left-color:#0369a1; }}
    .nav-case.type {{ border-left-color:#be185d; }}
    .nav-case.build {{ border-left-color:#374151; }}
    .nav-case.other {{ border-left-color:#4f46e5; }}
    main {{ padding:18px 28px 36px; }}
    .case {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; margin:0 0 18px; padding:16px; }}
    .case-head {{ display:flex; justify-content:space-between; gap:12px; align-items:flex-start; border-bottom:1px solid var(--line); padding-bottom:12px; margin-bottom:12px; }}
    .case-head p {{ margin:6px 0 0; color:var(--muted); }}
    .case-tags {{ display:flex; gap:6px; flex-wrap:wrap; margin-top:7px; }}
    .case-grid {{ display:grid; grid-template-columns:minmax(260px,420px) 1fr; gap:14px; align-items:start; margin-bottom:14px; }}
    .image-wrap {{ border:1px solid var(--line); border-radius:8px; background:#fff; padding:8px; }}
    img {{ display:block; width:100%; height:auto; max-height:300px; object-fit:contain; }}
    .score-panel {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:8px; }}
    .score-panel div {{ border:1px solid var(--line); border-radius:8px; padding:9px; background:#fafafa; }}
    .score-panel b {{ display:block; color:var(--muted); font-size:12px; margin-bottom:3px; }}
    table {{ width:100%; border-collapse:collapse; margin:12px 0; table-layout:fixed; }}
    th, td {{ border:1px solid var(--line); padding:7px 8px; text-align:left; vertical-align:top; overflow-wrap:anywhere; }}
    th {{ background:#f2f4f7; color:#4b5563; }}
    code, pre {{ font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:12px; }}
    pre {{ margin:0; padding:10px; background:#101828; color:#f8fafc; border-radius:8px; overflow:auto; max-height:340px; }}
    .netlists {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
    .tag {{ display:inline-block; border-radius:999px; padding:2px 7px; background:#eef2ff; color:#3730a3; font-size:12px; }}
    .tag.missing, .tag.extra, .tag.type {{ background:#fee4e2; color:var(--bad); }}
    .tag.nodes, .tag.polarity {{ background:#fef0c7; color:var(--warn); }}
    .error-chip {{ display:inline-flex; align-items:center; border-radius:999px; padding:3px 8px; font-size:12px; font-weight:650; line-height:1.2; border:1px solid transparent; }}
    .error-chip.compact {{ padding:1px 5px; font-size:10px; max-width:78px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
    .error-chip.excluded-step {{ background:#f3e8ff; color:#6d28d9; border-color:#d8b4fe; }}
    .error-chip.missing {{ background:#fee2e2; color:#991b1b; border-color:#fecaca; }}
    .error-chip.extra {{ background:#ffedd5; color:#9a3412; border-color:#fed7aa; }}
    .error-chip.nodes {{ background:#fef3c7; color:#92400e; border-color:#fde68a; }}
    .error-chip.polarity {{ background:#fef9c3; color:#854d0e; border-color:#fde047; }}
    .error-chip.value {{ background:#e0f2fe; color:#075985; border-color:#bae6fd; }}
    .error-chip.type {{ background:#fce7f3; color:#9d174d; border-color:#fbcfe8; }}
    .error-chip.build {{ background:#f3f4f6; color:#374151; border-color:#d1d5db; }}
    .error-chip.other {{ background:#e0e7ff; color:#3730a3; border-color:#c7d2fe; }}
    details {{ margin-top:12px; }}
    summary {{ cursor:pointer; color:var(--blue); }}
    .trace {{ border-top:1px solid var(--line); padding-top:10px; }}
    .trace h4 {{ margin:16px 0 8px; font-size:15px; }}
    .trace h5 {{ margin:10px 0 6px; font-size:12px; color:var(--muted); }}
    .trace-grid {{ display:grid; grid-template-columns:minmax(220px,360px) 1fr; gap:12px; align-items:start; }}
    .trace .image-wrap img {{ max-height:360px; }}
    .muted {{ color:var(--muted); }}
    @media (max-width: 860px) {{ header {{ position:static; }} main, header {{ padding-left:14px; padding-right:14px; }} .case-grid, .netlists {{ grid-template-columns:1fr; }} }}
    @media (max-width: 860px) {{ .trace-grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Netlist Eval Report</h1>
    <div class="metrics">
      {metric_card("Total", str(total))}
      {metric_card("Raw/Excluded", f"{raw_total}/{excluded}", "extra step self-source")}
      {metric_card("Strict match", f"{strict_ok}/{total}", f"{strict_pct:.1f}% with nodes")}
      {metric_card("Ignore-node match", f"{ignore_ok}/{total}", f"{ignore_pct:.1f}% components")}
      {metric_card("Wrong", str(len(wrong_rows)), f"{len(node_only)} node-only, {len(component_wrong)} component")}
      {metric_card("Shown", str(len(display_rows)), "wrong cases")}
    </div>
    <nav>{case_nav}</nav>
  </header>
  <main>
    {case_html or '<section class="case"><h2>No wrong cases found.</h2></section>'}
  </main>
</body>
</html>
"""


def _write_outputs(args: argparse.Namespace, selected: list[Path], results: list[dict[str, Any]], started: float) -> dict[str, Any]:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = _payload(args, selected, results, started)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_eval_report(payload, selected)
    return payload


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
            / output_stem
            / "results.json"
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
            _write_outputs(args, selected, results, started)
    else:
        print(f"Running {len(selected)} cases with workers={args.workers}", flush=True)
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
                _write_outputs(args, selected, sorted(results, key=lambda item: _case_key(Path(item["id"]))), started)
                print(
                    f"[{index}/{len(selected)}] {row['id']} success={row.get('success')} "
                    f"component_match={row['score'].get('component_multiset_match_ignore_nodes')} "
                    f"terminal_acc={row['score'].get('undirected_terminal_accuracy_on_common'):.3f}",
                    flush=True,
                )

    final_results = sorted(results, key=lambda item: _case_key(Path(item["id"])))
    final_payload = _write_outputs(args, selected, final_results, started)
    print_summary(final_payload)
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
    excluded_rows = [row for row in results if _excluded_by_extra_step_self_source(row)]
    excluded_ids = {row["id"] for row in excluded_rows}
    scored_results = [row for row in results if row["id"] not in excluded_ids]
    total = len(scored_results)
    return {
        "total": total,
        "raw_total": len(results),
        "excluded_extra_step_self_source": len(excluded_rows),
        "excluded_extra_step_self_source_ids": sorted(excluded_ids, key=lambda item: _case_key(Path(item))),
        "build_success": sum(1 for row in scored_results if row.get("success")),
        "component_multiset_match_ignore_nodes": sum(
            1 for row in scored_results if row["score"].get("component_multiset_match_ignore_nodes")
        ),
        "component_multiset_match_with_undirected_nodes": sum(
            1 for row in scored_results if row["score"].get("component_multiset_match_with_undirected_nodes")
        ),
        "avg_component_name_recall": _avg(scored_results, "component_name_recall"),
        "avg_component_name_precision": _avg(scored_results, "component_name_precision"),
        "avg_type_accuracy_on_common": _avg(scored_results, "component_type_accuracy_on_common"),
        "avg_value_accuracy_on_common": _avg(scored_results, "component_value_accuracy_on_common"),
        "avg_undirected_terminal_accuracy_on_common": _avg(scored_results, "undirected_terminal_accuracy_on_common"),
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
