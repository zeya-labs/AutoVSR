#!/usr/bin/env python3
"""Recognize real EDA block diagrams into the project JSON standard.

The output contract is defined in `specs/eda_block_diagram/standard.md` and
validated by `specs/eda_block_diagram/validate_block_diagram_json.py`.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from specs.eda_block_diagram.validate_block_diagram_json import validate  # noqa: E402


SPEC_DIR = PROJECT_ROOT / "specs" / "eda_block_diagram"
STANDARD_PATH = SPEC_DIR / "standard.md"
GRAPH_EDITOR_JS = PROJECT_ROOT / "scripts" / "assets" / "eda_graph_editor.js"
SCHEMA_NAME = "eda_block_diagram.v0.1"
VALID_EDGE_DIRECTIONS = {"directed", "undirected", "unknown"}


SYSTEM_PROMPT = """You are an expert EDA/system block-diagram transcriber.

Convert the image into the EDA Block Diagram JSON standard. Return ONLY one JSON
object, with no markdown fence and no prose.

Core requirements:
- Preserve both semantic blocks and signal-flow structure.
- Include all visible functional blocks, operators, sources/sinks, connectors,
  groups, important texts, and annotation-only marks.
- Use nodes for operators such as sum, product/mixer, gain, delay, integrator,
  switch, mux, saturation, and transfer-function blocks.
- Use ports to attach edges to blocks/operators whenever the attachment is
  clear. If exact port placement is unclear, still create logical ports with
  side="unknown".
- Preserve signal names on ports and/or edges, for example U1, D_in, f_REF,
  phi_d, theta, clk, reset.
- Represent dashed boxes, colored regions, chip boundaries, analog/digital
  domains, and named loop regions as groups.
- Distinguish real signal/data/control/feedback edges from annotation arrows or
  explanatory loop arrows. Use annotations for non-signal arrows.
- Do not invent blocks from filename, surrounding paper context, or prior
  knowledge. Only transcribe visible content.
- If an item is visible but uncertain, include it with kind="unknown" or a low
  confidence and add a warning.
- Use stable IDs: n_*, p_*, e_*, g_*, t_*, a_*.
- The JSON must validate structurally with the included validator.

Important schema reminders:
- Top-level required keys: schema, source, diagram, nodes, ports, edges, groups.
- schema must be "eda_block_diagram.v0.1".
- Node kind is one of: functional_block, operator, source_sink, connector,
  subsystem, unknown.
- Edge kind is one of: signal, feedback, bus, power_or_clock, physical,
  annotation_link, unknown.
- Edge source/target may reference either a port id or a node id.
- Coordinates are image pixels with origin at top-left; approximate bboxes are
  acceptable when confidence is lower.
"""


AUDIT_SYSTEM_PROMPT = """You are a meticulous EDA block-diagram visual auditor.

You receive the original image and a candidate graph-only JSON. Correct visual
transcription errors and return ONLY one complete JSON object in the same
eda_block_diagram.v0.1 graph-only format.

Audit rules:
- Use the image as authority; do not trust the candidate if the image disagrees.
- Keep graph-only format: ports=[], no bboxes, no text objects, no annotations
  unless absolutely necessary.
- Edges attach directly to node IDs.
- Verify every node is visibly present.
- Verify every edge is a real visible signal/control/feedback connection.
- Verify feedback labels carefully. Do not map z/x/y feedback onto phi/theta/psi
  nodes or vice versa unless the image shows that exact branch.
- Dashed arrows labeled Loop I, Loop II, Loop 1, etc. are loop annotations unless
  they are visibly connected as ordinary signal wires. Keep them as
  annotation_link edges or group annotations, not feedback signal edges.
- For summing nodes, preserve whether the inputs are reference, feedback, or
  attitude-error signals in the edge labels where visible.
- Remove hallucinated edges and add missing visible major edges.
- Preserve groups for dashed/colored functional regions.
- Keep output under 2200 tokens and ensure the JSON is complete.
"""


def _read_standard_excerpt() -> str:
    text = STANDARD_PATH.read_text(encoding="utf-8")
    sections = []
    keep = False
    for line in text.splitlines():
        if line.startswith("## Top-Level Object"):
            keep = True
        if line.startswith("## Example: Hidden EDA_TESTs_015"):
            break
        if keep:
            sections.append(line)
    return "\n".join(sections).strip()


def _image_message(image_path: Path) -> dict[str, Any]:
    suffix = image_path.suffix.lower()
    media_type = "image/png" if suffix == ".png" else "image/jpeg"
    data = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}


def _json_from_text(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text or ""):
        try:
            data, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if (
            isinstance(data, dict)
            and data.get("schema") == SCHEMA_NAME
            and all(key in data for key in ("nodes", "edges", "groups"))
        ):
            return data
    raise ValueError("no complete top-level block diagram JSON object found in VLM response")


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                elif "text" in item:
                    parts.append(str(item.get("text") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _image_size(image_path: Path) -> tuple[int | None, int | None]:
    try:
        with Image.open(image_path) as image:
            return image.size
    except Exception:
        return None, None


def _normalize_output(data: dict[str, Any], image_path: Path) -> dict[str, Any]:
    width, height = _image_size(image_path)
    data["schema"] = SCHEMA_NAME
    source = data.get("source")
    if not isinstance(source, dict):
        source = {}
    source["image_path"] = str(image_path)
    if width:
        source["image_width"] = width
    if height:
        source["image_height"] = height
    data["source"] = source
    data.setdefault("diagram", {"title": None, "domain": [], "view_type": "block_diagram"})
    data.setdefault("nodes", [])
    data.setdefault("ports", [])
    data.setdefault("edges", [])
    data.setdefault("groups", [])
    data.setdefault("texts", [])
    data.setdefault("annotations", [])
    data.setdefault("legend", [])
    data.setdefault("warnings", [])
    warnings = data["warnings"]
    if not isinstance(warnings, list):
        warnings = [str(warnings)]
        data["warnings"] = warnings
    for edge in data.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        direction = edge.get("direction")
        if direction not in VALID_EDGE_DIRECTIONS:
            if direction is not None:
                warnings.append(f"normalized invalid edge direction for {edge.get('id')}: {direction}")
            edge["direction"] = "directed"
        label = str(edge.get("label") or "").strip().lower()
        if re.fullmatch(r"loop\s*(?:[ivx]+|\d+)", label) and edge.get("kind") in {"signal", "feedback"}:
            warnings.append(f"converted loop annotation edge to annotation_link: {edge.get('id')}")
            edge["kind"] = "annotation_link"
            edge["direction"] = "unknown"
            if isinstance(edge.get("confidence"), (int, float)):
                edge["confidence"] = min(float(edge["confidence"]), 0.75)
    return data


def _quality_errors(data: dict[str, Any], min_nodes: int, min_edges: int) -> list[str]:
    errors: list[str] = []
    if len(data.get("nodes") or []) < min_nodes:
        errors.append(f"too few nodes: {len(data.get('nodes') or [])} < {min_nodes}")
    if len(data.get("edges") or []) < min_edges:
        errors.append(f"too few edges: {len(data.get('edges') or [])} < {min_edges}")
    return errors


def _audit_graph(
    llm: Any,
    image_path: Path,
    candidate: dict[str, Any],
    min_nodes: int,
    min_edges: int,
) -> tuple[dict[str, Any], str, list[str]]:
    compact_candidate = {
        "schema": candidate.get("schema"),
        "source": candidate.get("source"),
        "diagram": candidate.get("diagram"),
        "nodes": candidate.get("nodes") or [],
        "ports": [],
        "edges": candidate.get("edges") or [],
        "groups": candidate.get("groups") or [],
        "warnings": candidate.get("warnings") or [],
    }
    user_text = f"""Candidate graph JSON:
{json.dumps(compact_candidate, ensure_ascii=False)}

Return the corrected complete graph-only JSON object."""
    response = llm.invoke(
        [
            SystemMessage(content=AUDIT_SYSTEM_PROMPT),
            HumanMessage(content=[_image_message(image_path), {"type": "text", "text": user_text}]),
        ]
    )
    raw = _extract_text_content(response.content)
    audited = _normalize_output(_json_from_text(raw), image_path)
    errors = validate(audited)
    errors.extend(_quality_errors(audited, min_nodes, min_edges))
    return audited, raw, errors


def _case_id(image_path: Path) -> str:
    return image_path.stem


def _safe_json_name(image_path: Path) -> str:
    return f"{_case_id(image_path)}.json"


def _artifact_base(image_path: Path) -> str:
    return _case_id(image_path)


def _dot_quote(value: Any) -> str:
    text = str(value or "")
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def _short_label(value: Any, limit: int = 42) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."


def _edge_color(kind: str) -> str:
    return {
        "feedback": "#b42318",
        "bus": "#7a4cc2",
        "power_or_clock": "#a05a00",
        "annotation_link": "#6a7382",
        "physical": "#14784a",
        "unknown": "#6a7382",
    }.get(kind, "#1f5fae")


def _node_shape(kind: str) -> str:
    return {
        "operator": "circle",
        "source_sink": "oval",
        "connector": "point",
        "subsystem": "component",
        "unknown": "box",
    }.get(kind, "box")


def _diagram_to_dot(data: dict[str, Any]) -> str:
    nodes = [node for node in data.get("nodes") or [] if isinstance(node, dict)]
    edges = [edge for edge in data.get("edges") or [] if isinstance(edge, dict)]
    groups = [group for group in data.get("groups") or [] if isinstance(group, dict)]
    ports = {str(port.get("id")): str(port.get("node")) for port in data.get("ports") or [] if isinstance(port, dict)}
    node_ids = {str(node.get("id")) for node in nodes if node.get("id")}
    lines = [
        "digraph G {",
        "  graph [rankdir=LR, bgcolor=\"white\", pad=\"0.2\", nodesep=\"0.45\", ranksep=\"0.7\"];",
        "  node [fontname=\"Helvetica\", fontsize=11, margin=\"0.08,0.05\", color=\"#334155\", penwidth=1.2];",
        "  edge [fontname=\"Helvetica\", fontsize=9, color=\"#1f5fae\", arrowsize=0.7, penwidth=1.4];",
    ]
    grouped_nodes: set[str] = set()
    for index, group in enumerate(groups):
        contains = group.get("contains") or {}
        contained = []
        if isinstance(contains, dict):
            contained = [str(item) for item in contains.get("nodes") or [] if str(item) in node_ids]
        if not contained:
            continue
        grouped_nodes.update(contained)
        label = _short_label(group.get("label") or group.get("id") or f"group {index}")
        lines.extend(
            [
                f"  subgraph cluster_{index} {{",
                f"    label={_dot_quote(label)};",
                "    color=\"#9aa7b8\";",
                "    style=\"rounded,dashed\";",
                "    fontname=\"Helvetica\";",
                "    fontsize=12;",
            ]
        )
        for node_id in contained:
            node = next((item for item in nodes if str(item.get("id")) == node_id), {})
            label_text = _short_label(node.get("label") or node_id)
            kind = str(node.get("kind") or "")
            shape = _node_shape(kind)
            fill = "#fff7ed" if kind == "operator" else "#f8fafc"
            lines.append(
                f"    {_dot_quote(node_id)} [label={_dot_quote(label_text)}, shape={_dot_quote(shape)}, "
                f"style=\"filled\", fillcolor={_dot_quote(fill)}];"
            )
        lines.append("  }")
    for node in nodes:
        node_id = str(node.get("id") or "")
        if not node_id or node_id in grouped_nodes:
            continue
        label_text = _short_label(node.get("label") or node_id)
        kind = str(node.get("kind") or "")
        shape = _node_shape(kind)
        fill = "#fff7ed" if kind == "operator" else "#f8fafc"
        lines.append(
            f"  {_dot_quote(node_id)} [label={_dot_quote(label_text)}, shape={_dot_quote(shape)}, "
            f"style=\"filled\", fillcolor={_dot_quote(fill)}];"
        )
    for edge in edges:
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        source = ports.get(source, source)
        target = ports.get(target, target)
        if source not in node_ids or target not in node_ids:
            continue
        kind = str(edge.get("kind") or "signal")
        label = _short_label(edge.get("label") or kind, 28)
        attrs = [
            f"label={_dot_quote(label)}",
            f"color={_dot_quote(_edge_color(kind))}",
            f"fontcolor={_dot_quote(_edge_color(kind))}",
        ]
        if kind in {"feedback", "annotation_link"}:
            attrs.append('style="dashed"')
        if edge.get("direction") == "undirected":
            attrs.append('dir="none"')
        elif edge.get("direction") == "unknown":
            attrs.append('dir="both"')
        lines.append(f"  {_dot_quote(source)} -> {_dot_quote(target)} [{', '.join(attrs)}];")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _write_graph_svg(output_dir: Path, case_id: str, data: dict[str, Any]) -> str | None:
    dot_path = output_dir / f"{case_id}.graph.dot"
    svg_path = output_dir / f"{case_id}.graph.svg"
    dot_path.write_text(_diagram_to_dot(data), encoding="utf-8")
    try:
        subprocess.run(
            ["dot", "-Tsvg", str(dot_path), "-o", str(svg_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    _write_graph_layout_json(output_dir, case_id, dot_path, data)
    return str(svg_path)


def _write_graph_layout_json(output_dir: Path, case_id: str, dot_path: Path, data: dict[str, Any]) -> str | None:
    try:
        result = subprocess.run(
            ["dot", "-Tplain", str(dot_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    layout = _plain_to_edit_layout(result.stdout, data)
    layout_path = output_dir / f"{case_id}.graph.layout.json"
    layout_path.write_text(json.dumps(layout, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(layout_path)


def _plain_to_edit_layout(plain: str, data: dict[str, Any]) -> dict[str, Any]:
    scale = 72.0
    graph_width = 800.0
    graph_height = 600.0
    nodes_by_id = {
        str(node.get("id")): node for node in data.get("nodes") or [] if isinstance(node, dict) and node.get("id")
    }
    node_layout: dict[str, dict[str, Any]] = {}
    for line in plain.splitlines():
        if not line.strip():
            continue
        parts = shlex.split(line)
        if not parts:
            continue
        if parts[0] == "graph" and len(parts) >= 4:
            graph_width = max(float(parts[2]) * scale, 320.0)
            graph_height = max(float(parts[3]) * scale, 240.0)
        elif parts[0] == "node" and len(parts) >= 6:
            node_id = parts[1]
            source_node = nodes_by_id.get(node_id, {})
            kind = str(source_node.get("kind") or "")
            label = str(source_node.get("label") or (parts[6] if len(parts) > 6 else node_id))
            node_layout[node_id] = {
                "id": node_id,
                "label": label,
                "kind": kind,
                "shape": _node_shape(kind),
                "x": float(parts[2]) * scale,
                "y": graph_height - (float(parts[3]) * scale),
                "w": max(float(parts[4]) * scale, 28.0),
                "h": max(float(parts[5]) * scale, 24.0),
            }
    ports = {str(port.get("id")): str(port.get("node")) for port in data.get("ports") or [] if isinstance(port, dict)}
    edges = []
    for edge in data.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        source = ports.get(str(edge.get("source") or ""), str(edge.get("source") or ""))
        target = ports.get(str(edge.get("target") or ""), str(edge.get("target") or ""))
        if source not in node_layout or target not in node_layout:
            continue
        kind = str(edge.get("kind") or "signal")
        edges.append(
            {
                "id": str(edge.get("id") or f"{source}_to_{target}"),
                "source": source,
                "target": target,
                "label": str(edge.get("label") or ""),
                "kind": kind,
                "direction": str(edge.get("direction") or "directed"),
                "color": _edge_color(kind),
            }
        )
    groups = []
    for group in data.get("groups") or []:
        if not isinstance(group, dict):
            continue
        contains = group.get("contains") or {}
        nodes = []
        if isinstance(contains, dict):
            nodes = [str(node_id) for node_id in contains.get("nodes") or [] if str(node_id) in node_layout]
        if nodes:
            groups.append(
                {
                    "id": str(group.get("id") or ""),
                    "label": str(group.get("label") or group.get("id") or ""),
                    "nodes": nodes,
                }
            )
    return {
        "width": round(graph_width + 36, 2),
        "height": round(graph_height + 36, 2),
        "nodes": list(node_layout.values()),
        "edges": edges,
        "groups": groups,
    }


def recognize_one(
    image_path: Path,
    output_dir: Path,
    quiet: bool = False,
    mode: str = "compact",
    min_nodes: int = 1,
    min_edges: int = 1,
    audit: bool = False,
) -> dict[str, Any]:
    import main as autovsr_main

    started = time.time()
    llm = autovsr_main.create_llm()
    standard_excerpt = _read_standard_excerpt()
    width, height = _image_size(image_path)
    if mode == "graph":
        detail_instruction = """Use graph-only mode:
- Output ONLY top-level keys: schema, source, diagram, nodes, ports, edges, groups, warnings.
- Set ports to [].
- Set texts, annotations, and legend to [] or omit them.
- Edges MUST attach directly to node IDs, not port IDs.
- For each node use only: id, kind, subtype, label, confidence.
- For each edge use only: id, kind, source, target, direction, label, confidence.
- For each group use only: id, kind, label, contains, confidence.
- Do NOT include bbox, path, text objects, style, evidence, properties, or port lists.
- Keep output under 2200 tokens and make sure the JSON object is complete."""
    elif mode == "compact":
        detail_instruction = """Use compact mode:
- Include all major visible blocks/operators/groups and all major signal edges.
- Omit bbox for small ports and minor text if needed to stay concise.
- You may attach edges directly to node IDs when exact ports are too verbose.
- Keep ports only for important named interfaces.
- Target less than 3500 output tokens."""
    else:
        detail_instruction = """Use full mode:
- Include ports and approximate bboxes wherever visible.
- Preserve important text and annotations in detail."""

    user_text = f"""IMAGE:
{image_path}
size: {width}x{height}

Use this standard excerpt:
{standard_excerpt}

{detail_instruction}

Transcribe the complete visible diagram into one JSON object. Prefer a complete
semantic graph over exact coordinates when the image is dense, but include
approximate bboxes for major objects."""

    raw = ""
    audit_raw = ""
    audit_errors: list[str] = []
    parsed: dict[str, Any] | None = None
    errors: list[str] = []
    try:
        response = llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=[_image_message(image_path), {"type": "text", "text": user_text}]),
            ]
        )
        raw = _extract_text_content(response.content)
        parsed = _normalize_output(_json_from_text(raw), image_path)
        errors = validate(parsed)
        errors.extend(_quality_errors(parsed, min_nodes, min_edges))
        if audit and not errors:
            audited, audit_raw, audit_errors = _audit_graph(llm, image_path, parsed, min_nodes, min_edges)
            if not audit_errors:
                parsed = audited
            else:
                parsed.setdefault("warnings", []).append(
                    f"audit failed; kept first-pass graph ({len(audit_errors)} audit errors)"
                )
            raw = raw + "\n\n__AUDIT_RESPONSE__\n" + audit_raw
    except Exception as exc:
        errors = [f"{type(exc).__name__}: {exc}"]
        parsed = {
            "schema": SCHEMA_NAME,
            "source": {"image_path": str(image_path), "image_width": width, "image_height": height},
            "diagram": {"title": None, "domain": [], "view_type": "unknown"},
            "nodes": [],
            "ports": [],
            "edges": [],
            "groups": [],
            "texts": [],
            "annotations": [],
            "legend": [],
            "warnings": ["recognition failed"],
        }

    row = {
        "id": _case_id(image_path),
        "image_path": str(image_path),
        "output_json": str(output_dir / _safe_json_name(image_path)),
        "raw_response": raw,
        "validation_errors": errors,
        "audit_errors": audit_errors,
        "valid": not errors,
        "mode": mode,
        "audit": audit,
        "duration_seconds": round(time.time() - started, 2),
        "summary": {
            "nodes": len(parsed.get("nodes") or []),
            "ports": len(parsed.get("ports") or []),
            "edges": len(parsed.get("edges") or []),
            "groups": len(parsed.get("groups") or []),
            "texts": len(parsed.get("texts") or []),
            "annotations": len(parsed.get("annotations") or []),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / _safe_json_name(image_path)).write_text(
        json.dumps(parsed, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    graph_svg = _write_graph_svg(output_dir, _artifact_base(image_path), parsed)
    row["graph_svg"] = graph_svg or ""
    (output_dir / f"{_artifact_base(image_path)}.raw.txt").write_text(raw, encoding="utf-8")
    (output_dir / f"{_artifact_base(image_path)}.meta.json").write_text(
        json.dumps(row, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not quiet:
        status = "OK" if row["valid"] else f"INVALID {len(errors)}"
        print(
            f"{image_path.name}: {status} "
            f"nodes={row['summary']['nodes']} edges={row['summary']['edges']} "
            f"groups={row['summary']['groups']} duration={row['duration_seconds']}s",
            flush=True,
        )
    return row


def _select_images(args: argparse.Namespace) -> list[Path]:
    if args.image:
        return [args.image]
    if not args.input_dir:
        raise ValueError("either --image or --input-dir is required")
    images = sorted(
        [
            path
            for path in args.input_dir.rglob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
    )
    if args.start:
        images = images[args.start :]
    if args.limit is not None:
        images = images[: args.limit]
    return images


def _write_summary(output_dir: Path, rows: list[dict[str, Any]], started: float) -> None:
    payload = {
        "schema": SCHEMA_NAME,
        "elapsed_seconds": round(time.time() - started, 2),
        "total": len(rows),
        "valid": sum(1 for row in rows if row.get("valid")),
        "invalid": sum(1 for row in rows if not row.get("valid")),
        "rows": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    if GRAPH_EDITOR_JS.exists():
        shutil.copy2(GRAPH_EDITOR_JS, output_dir / GRAPH_EDITOR_JS.name)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "index.html").write_text(_render_html_report(output_dir, payload), encoding="utf-8")


def _rel_link(target: str | Path | None, base_dir: Path) -> str:
    if not target:
        return ""
    path = Path(target)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    try:
        return os.path.relpath(path, base_dir)
    except ValueError:
        return str(path)


def _read_diagram_json(row: dict[str, Any]) -> dict[str, Any]:
    output_json = row.get("output_json")
    if not output_json:
        return {}
    path = Path(str(output_json))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _artifact_link(row: dict[str, Any], suffix: str, output_dir: Path) -> str:
    output_json = row.get("output_json")
    if not output_json:
        return ""
    try:
        return _rel_link(Path(str(output_json)).with_suffix(suffix), output_dir)
    except ValueError:
        return ""


def _ensure_graph_svg(output_dir: Path, row: dict[str, Any], data: dict[str, Any]) -> str:
    graph_svg = str(row.get("graph_svg") or "")
    if graph_svg:
        path = Path(graph_svg)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.exists():
            layout_path = path.with_suffix(".layout.json")
            dot_path = path.with_suffix(".dot")
            if data and not layout_path.exists() and dot_path.exists():
                _write_graph_layout_json(output_dir, path.name.removesuffix(".graph.svg"), dot_path, data)
            return str(path)
    if not data:
        return ""
    case_id = str(row.get("id") or "")
    if not case_id:
        output_json = row.get("output_json")
        case_id = Path(str(output_json)).stem if output_json else ""
    if not case_id:
        return ""
    return _write_graph_svg(output_dir, case_id, data) or ""


def _layout_link_from_svg(graph_svg_path: str, output_dir: Path) -> str:
    if not graph_svg_path:
        return ""
    path = Path(graph_svg_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return _rel_link(path.with_suffix(".layout.json"), output_dir)


def _node_label_map(nodes: list[dict[str, Any]]) -> dict[str, str]:
    labels = {}
    for node in nodes:
        node_id = str(node.get("id") or "")
        label = str(node.get("label") or node_id)
        if node_id:
            labels[node_id] = label
    return labels


def _fmt_conf(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"
    return ""


def _render_list(items: list[Any]) -> str:
    if not items:
        return '<span class="muted">none</span>'
    return "<ul>" + "".join(f"<li>{html.escape(str(item))}</li>" for item in items) + "</ul>"


def _render_nodes(nodes: list[dict[str, Any]]) -> str:
    if not nodes:
        return '<p class="muted">No nodes.</p>'
    rows = []
    for node in nodes:
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(node.get('id') or ''))}</code></td>"
            f"<td>{html.escape(str(node.get('label') or ''))}</td>"
            f"<td>{html.escape(str(node.get('kind') or ''))}</td>"
            f"<td>{html.escape(str(node.get('subtype') or ''))}</td>"
            f"<td>{html.escape(_fmt_conf(node.get('confidence')))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>ID</th><th>Label</th><th>Kind</th><th>Subtype</th><th>Conf</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _render_edges(edges: list[dict[str, Any]], labels: dict[str, str]) -> str:
    if not edges:
        return '<p class="muted">No edges.</p>'
    rows = []
    for edge in edges:
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        source_text = f"{labels.get(source, source)} ({source})" if source else ""
        target_text = f"{labels.get(target, target)} ({target})" if target else ""
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(edge.get('id') or ''))}</code></td>"
            f"<td>{html.escape(str(edge.get('kind') or ''))}</td>"
            f"<td>{html.escape(source_text)}</td>"
            f"<td>{html.escape(target_text)}</td>"
            f"<td>{html.escape(str(edge.get('label') or ''))}</td>"
            f"<td>{html.escape(str(edge.get('direction') or ''))}</td>"
            f"<td>{html.escape(_fmt_conf(edge.get('confidence')))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>ID</th><th>Kind</th><th>Source</th><th>Target</th>"
        f"<th>Label</th><th>Dir</th><th>Conf</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def _render_groups(groups: list[dict[str, Any]]) -> str:
    if not groups:
        return '<p class="muted">No groups.</p>'
    rows = []
    for group in groups:
        contains = group.get("contains") or {}
        if isinstance(contains, dict):
            contains_text = ", ".join(
                f"{key}:{len(value)}" for key, value in contains.items() if isinstance(value, list)
            )
        else:
            contains_text = str(contains)
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(group.get('id') or ''))}</code></td>"
            f"<td>{html.escape(str(group.get('label') or ''))}</td>"
            f"<td>{html.escape(str(group.get('kind') or ''))}</td>"
            f"<td>{html.escape(contains_text)}</td>"
            f"<td>{html.escape(_fmt_conf(group.get('confidence')))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>ID</th><th>Label</th><th>Kind</th><th>Contains</th><th>Conf</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _render_case(output_dir: Path, row: dict[str, Any]) -> str:
    data = _read_diagram_json(row)
    nodes = data.get("nodes") or []
    edges = data.get("edges") or []
    groups = data.get("groups") or []
    warnings = data.get("warnings") or []
    label_map = _node_label_map(nodes)
    image_rel = _rel_link(row.get("image_path"), output_dir)
    graph_svg_path = _ensure_graph_svg(output_dir, row, data)
    graph_svg_rel = _rel_link(graph_svg_path, output_dir)
    graph_layout_rel = _layout_link_from_svg(graph_svg_path, output_dir)
    json_rel = _rel_link(row.get("output_json"), output_dir)
    raw_rel = _artifact_link(row, ".raw.txt", output_dir)
    meta_rel = _artifact_link(row, ".meta.json", output_dir)
    status = "valid" if row.get("valid") else "invalid"
    errors = list(row.get("validation_errors") or [])
    audit_errors = list(row.get("audit_errors") or [])
    summary = row.get("summary") or {}
    image_html = (
        f'<img src="{html.escape(image_rel)}" alt="{html.escape(str(row.get("id") or ""))} diagram">'
        if image_rel
        else '<p class="muted">image unavailable</p>'
    )
    graph_html = (
        f"""
        <div class="graph-editor" data-layout="{html.escape(graph_layout_rel)}" data-case="{html.escape(str(row.get('id') or ''))}">
          <div class="editor-toolbar">
            <button type="button" class="save-graph">Save</button>
            <button type="button" class="auto-tidy">Auto tidy</button>
            <button type="button" class="undo-tidy">Undo tidy</button>
            <button type="button" class="add-node">Add node</button>
            <button type="button" class="connect-edge">Connect</button>
            <button type="button" class="delete-selected">Delete</button>
            <input class="edit-label" type="text" placeholder="selected label">
            <select class="edit-kind">
              <option value="functional_block">block</option>
              <option value="operator">operator</option>
              <option value="source_sink">source/sink</option>
              <option value="signal">signal edge</option>
              <option value="feedback">feedback edge</option>
              <option value="bus">bus edge</option>
              <option value="annotation_link">annotation edge</option>
            </select>
            <a href="{html.escape(graph_svg_rel)}">static svg</a>
            <span class="save-status muted"></span>
          </div>
          <div class="editable-graph" aria-label="{html.escape(str(row.get('id') or ''))} editable graph"></div>
        </div>
        """
        if graph_layout_rel
        else '<p class="muted">graph editor unavailable</p>'
    )
    return f"""
    <section class="case {status}" id="{html.escape(str(row.get('id') or ''))}">
      <header>
        <div>
          <h2>{html.escape(str(row.get('id') or ''))}</h2>
          <p>{html.escape(str(row.get('mode') or ''))} mode · audit={html.escape(str(row.get('audit')))} · {html.escape(str(row.get('duration_seconds')))}s</p>
        </div>
        <span class="badge {status}">{status}</span>
      </header>
      <div class="compare-grid">
        <div>
          <h3>Original Image</h3>
          <div class="image-wrap">{image_html}</div>
        </div>
        <div>
          <h3>Reconstructed Graph SVG</h3>
          <div class="image-wrap graph-wrap">{graph_html}</div>
        </div>
      </div>
      <div class="case-grid">
        <div>
          <div class="metrics">
            <div><b>{html.escape(str(summary.get('nodes', len(nodes))))}</b><span>nodes</span></div>
            <div><b>{html.escape(str(summary.get('edges', len(edges))))}</b><span>edges</span></div>
            <div><b>{html.escape(str(summary.get('groups', len(groups))))}</b><span>groups</span></div>
            <div><b>{html.escape(str(summary.get('ports', len(data.get('ports') or []))))}</b><span>ports</span></div>
          </div>
          <div class="links">
            <a href="{html.escape(json_rel)}">json</a>
            <a href="{html.escape(raw_rel)}">raw</a>
            <a href="{html.escape(meta_rel)}">meta</a>
            <a href="{html.escape(image_rel)}">image</a>
            <a href="{html.escape(graph_svg_rel)}">graph svg</a>
          </div>
          <h3>Validation Errors</h3>
          {_render_list(errors)}
          <h3>Audit Errors</h3>
          {_render_list(audit_errors)}
          <h3>Warnings</h3>
          {_render_list(warnings)}
        </div>
      </div>
      <details open><summary>Edges</summary>{_render_edges(edges, label_map)}</details>
      <details><summary>Nodes</summary>{_render_nodes(nodes)}</details>
      <details><summary>Groups</summary>{_render_groups(groups)}</details>
      <details><summary>Raw Response</summary><pre>{html.escape(str(row.get('raw_response') or ''))}</pre></details>
    </section>
    """


def _render_html_report(output_dir: Path, payload: dict[str, Any]) -> str:
    rows = payload.get("rows") or []
    nav = "\n".join(
        f'<a class="{html.escape("valid" if row.get("valid") else "invalid")}" href="#{html.escape(str(row.get("id") or ""))}">'
        f'{html.escape(str(row.get("id") or ""))}</a>'
        for row in rows
    )
    cases = "\n".join(_render_case(output_dir, row) for row in rows)
    total = int(payload.get("total") or 0)
    valid = int(payload.get("valid") or 0)
    invalid = int(payload.get("invalid") or 0)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>EDA Block Diagram Recognition</title>
  <style>
    :root {{
      color-scheme: light;
      --text:#18212f; --muted:#6a7382; --line:#d9dee7; --bg:#f7f8fa;
      --panel:#ffffff; --ok:#14784a; --bad:#b42318; --blue:#1f5fae;
    }}
    body {{ margin:0; font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif; color:var(--text); background:var(--bg); }}
    .layout {{ display:grid; grid-template-columns:220px minmax(0,1fr); min-height:100vh; }}
    nav {{ position:sticky; top:0; height:100vh; overflow:auto; padding:16px; border-right:1px solid var(--line); background:#eef2f6; }}
    nav h1 {{ font-size:16px; margin:0 0 12px; }}
    nav .totals {{ display:grid; grid-template-columns:repeat(3,1fr); gap:6px; margin-bottom:14px; }}
    nav .totals div {{ background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:8px; text-align:center; }}
    nav .totals b {{ display:block; font-size:18px; }}
    nav a {{ display:block; color:var(--text); text-decoration:none; padding:6px 8px; border-left:3px solid var(--line); }}
    nav a.valid {{ border-left-color:var(--ok); }}
    nav a.invalid {{ border-left-color:var(--bad); }}
    main {{ padding:18px; }}
    .case {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; margin:0 0 18px; padding:16px; }}
    .case.invalid {{ border-color:#e6aaa5; }}
    header {{ display:flex; align-items:flex-start; justify-content:space-between; gap:12px; margin-bottom:12px; }}
    h2 {{ font-size:20px; margin:0; }}
    h3 {{ font-size:13px; margin:14px 0 6px; color:#313a47; }}
    p {{ margin:3px 0 0; color:var(--muted); }}
    .badge {{ border-radius:999px; padding:4px 10px; font-weight:700; text-transform:uppercase; font-size:12px; }}
    .badge.valid {{ color:var(--ok); background:#e9f6ef; }}
    .badge.invalid {{ color:var(--bad); background:#fff0ee; }}
    .case-grid {{ display:grid; grid-template-columns:minmax(300px, 0.9fr) minmax(300px, 1.1fr); gap:16px; align-items:start; }}
    .compare-grid {{ display:grid; grid-template-columns:repeat(2,minmax(300px,1fr)); gap:16px; align-items:start; margin-bottom:12px; }}
    .image-wrap {{ border:1px solid var(--line); background:#fff; overflow:auto; max-height:560px; }}
    .image-wrap img {{ display:block; width:100%; height:auto; }}
    .graph-wrap {{ min-height:260px; }}
    .graph-editor {{ min-width:520px; }}
    .editor-toolbar {{ display:flex; gap:8px; align-items:center; padding:8px; border-bottom:1px solid var(--line); background:#f8fafc; }}
    .editor-toolbar button, .editor-toolbar input, .editor-toolbar select {{ border:1px solid #b8cbe7; color:#164a8b; background:#fff; border-radius:6px; padding:5px 9px; }}
    .editor-toolbar button {{ cursor:pointer; }}
    .editor-toolbar button.active {{ background:#dbeafe; border-color:#1f5fae; }}
    .editor-toolbar input {{ min-width:150px; color:#18212f; }}
    .editable-graph {{ min-height:420px; overflow:auto; background:#fff; }}
    .editable-graph svg {{ display:block; min-width:100%; height:auto; }}
    .editable-group rect {{ pointer-events:none; }}
    .editable-group text {{ pointer-events:none; user-select:none; }}
    .editable-node {{ cursor:grab; }}
    .editable-node.dragging {{ cursor:grabbing; }}
    .editable-node.selected rect, .editable-node.selected ellipse {{ stroke:#b42318; stroke-width:2.4; }}
    .editable-edge {{ cursor:pointer; }}
    .editable-edge.selected line {{ stroke-width:3; }}
    .editable-node text {{ pointer-events:none; user-select:none; }}
    .metrics {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px; }}
    .metrics div {{ border:1px solid var(--line); border-radius:6px; padding:8px; background:#fbfcfd; }}
    .metrics b {{ display:block; font-size:20px; }}
    .metrics span, .muted {{ color:var(--muted); }}
    .links {{ display:flex; flex-wrap:wrap; gap:8px; margin:12px 0; }}
    .links a {{ color:var(--blue); text-decoration:none; border:1px solid #b8cbe7; padding:4px 8px; border-radius:6px; }}
    ul {{ margin:0; padding-left:18px; }}
    details {{ margin-top:12px; border-top:1px solid var(--line); padding-top:10px; }}
    summary {{ cursor:pointer; font-weight:700; color:#283446; }}
    table {{ width:100%; border-collapse:collapse; margin-top:8px; table-layout:fixed; }}
    th,td {{ border-bottom:1px solid var(--line); padding:6px 8px; text-align:left; vertical-align:top; overflow-wrap:anywhere; }}
    th {{ background:#f2f5f8; font-size:12px; }}
    code {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:12px; }}
    pre {{ white-space:pre-wrap; overflow:auto; max-height:520px; background:#101820; color:#e8eef7; padding:12px; border-radius:6px; }}
    @media (max-width: 900px) {{
      .layout {{ display:block; }}
      nav {{ position:relative; height:auto; }}
      .case-grid, .compare-grid {{ grid-template-columns:1fr; }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <nav>
      <h1>EDA Recognition</h1>
      <div class="totals">
        <div><b>{total}</b><span>total</span></div>
        <div><b>{valid}</b><span>valid</span></div>
        <div><b>{invalid}</b><span>invalid</span></div>
      </div>
      {nav}
    </nav>
    <main>{cases or '<section class="case"><h2>No cases.</h2></section>'}</main>
  </div>
  <script src="eda_graph_editor.js"></script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, help="Single image to recognize.")
    parser.add_argument("--input-dir", type=Path, help="Directory of images to recognize recursively.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output" / "eda_block_diagram_recognition")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--mode", choices=("graph", "compact", "full"), default="graph")
    parser.add_argument("--min-nodes", type=int, default=1)
    parser.add_argument("--min-edges", type=int, default=1)
    parser.add_argument("--audit", action="store_true", help="Run a second visual audit/refinement pass.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    images = _select_images(args)
    if not images:
        parser.error("no images selected")

    started = time.time()
    rows: list[dict[str, Any]] = []
    if args.workers <= 1:
        for image_path in images:
            rows.append(
                recognize_one(
                    image_path,
                    args.output_dir,
                    args.quiet,
                    mode=args.mode,
                    min_nodes=args.min_nodes,
                    min_edges=args.min_edges,
                    audit=args.audit,
                )
            )
            _write_summary(args.output_dir, rows, started)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    recognize_one,
                    image_path,
                    args.output_dir,
                    args.quiet,
                    args.mode,
                    args.min_nodes,
                    args.min_edges,
                    args.audit,
                ): image_path
                for image_path in images
            }
            for future in as_completed(futures):
                rows.append(future.result())
                _write_summary(args.output_dir, sorted(rows, key=lambda row: row["id"]), started)

    rows = sorted(rows, key=lambda row: row["id"])
    _write_summary(args.output_dir, rows, started)
    print(
        f"done: {sum(1 for row in rows if row.get('valid'))}/{len(rows)} valid -> {args.output_dir / 'summary.json'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
