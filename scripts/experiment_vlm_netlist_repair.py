#!/usr/bin/env python3
"""Run fast VLM netlist improvement experiments on an existing eval result.

The script is intentionally separate from eval_vlm_netlist.py. It takes a
baseline results.json, selects wrong cases, and tries repair strategies that are
meant to generalize beyond CircuitSense-specific detectors:

0. wire_sanitize: deterministic no-self-loop/no-ordinary-wire baseline.
1. targeted: ask the VLM to audit only suspicious topology/component issues.
2. structured: ask the VLM for a structured component/short table, then compile.
3. vote: run multiple structured generations and vote component endpoints.
4. cascade: structured on all selected cases, then vote only structured residuals.
5. tiled: transcribe overlapping image tiles, then merge tile observations with
   the full image.
6. audit: visually audit a candidate netlist against the full image.
7. tile_consensus: use complete tile observations as a conservative fallback.

Methods named oracle_* use ground-truth eval scores to select cases or choose
between candidates. They are diagnostics/upper bounds, not deployable inference.

Each method writes its own JSON and HTML report under the experiment directory.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import copy
import io
import json
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_vlm_netlist import (  # noqa: E402
    _case_key,
    _component_map_for_report,
    _excluded_by_extra_step_self_source,
    _is_wrong,
    _score_netlist,
    _summary,
    _write_eval_report,
)
from src.utils.response_parser import extract_text_content  # noqa: E402


TARGETED_SYSTEM_PROMPT = """You are a circuit-netlist topology reviewer.

You will receive one schematic image, the original question, a baseline Lcapy
netlist, and a list of suspicious differences found by an automatic auditor.

Return ONLY one JSON object:
{
  "netlist": "complete corrected netlist, one component per line",
  "changes": ["short explanation of each changed line"]
}

Rules:
- Do not solve the circuit. Only correct image-to-netlist topology.
- Preserve all component names and values exactly as drawn.
- Ordinary continuous wires are node identity, not components.
- Do not output self-loop wires such as W1 1 1.
- Use W only for an explicit short between two different labeled nodes.
- If the baseline is already correct for a suspicious item, keep it unchanged.
- The returned netlist must be complete, not a patch.
"""


STRUCTURED_SYSTEM_PROMPT = """You are a circuit schematic transcriber.

Convert the schematic into a structured circuit graph, not prose.
Return ONLY one JSON object:
{
  "components": [
    {"name": "R1", "type": "R", "nodes": ["1", "2"], "value": "R1"}
  ],
  "shorts": [
    {"name": "W1", "nodes": ["1", "2"], "evidence": "explicit short between different labeled nodes"}
  ]
}

Rules:
- Use integer node labels from the image; node 0 is ground.
- Preserve drawn component names and values exactly.
- Include passives and sources as components.
- Do not include ordinary wire segments as components.
- Do not include a short whose endpoints are identical.
- Add a short only when the schematic explicitly shorts two different labeled nodes.
- For transfer functions, voltage/current sources use s-domain form in the final compiled netlist.
- For transient/nodal s-domain questions, source labels may remain symbolic; the compiler will normalize.
"""


TILE_SYSTEM_PROMPT = """You are reading one cropped tile from a larger circuit schematic.

Return ONLY one JSON object:
{
  "tile": "tile label from the prompt",
  "visible_components": [
    {"name": "R1", "type": "R", "nodes": ["1", "2"], "value": "R1", "visibility": "complete|partial"}
  ],
  "visible_shorts": [
    {"nodes": ["1", "2"], "evidence": "explicit short between different labeled nodes", "visibility": "complete|partial"}
  ],
  "notes": ["brief uncertainty notes"]
}

Rules:
- This is only a tile. Report what is visible; do not infer hidden off-tile endpoints.
- If a component is cut off by the tile boundary, mark visibility as "partial".
- Prefer exact component labels and exact integer node labels when visible.
- Do not include ordinary wire segments as shorts.
- Include a short only if the crop visibly shorts two different labeled nodes.
- Do not invent components or node labels hidden outside the crop.
"""


TILE_MERGE_SYSTEM_PROMPT = """You are merging local tile observations into one complete circuit graph.

You will receive the full schematic image and JSON observations from overlapping
tiles. The tile observations are hints, not ground truth. Use the full image to
resolve duplicates, partial components, boundary crossings, and node labels.

Return ONLY one JSON object:
{
  "components": [
    {"name": "R1", "type": "R", "nodes": ["1", "2"], "value": "R1"}
  ],
  "shorts": [
    {"name": "W1", "nodes": ["1", "2"], "evidence": "explicit short between different labeled nodes"}
  ]
}

Rules:
- Produce one complete global circuit graph.
- Preserve component names and values exactly as drawn.
- Use integer node labels from the full image; node 0 is ground.
- Merge duplicate observations of the same component from adjacent tiles.
- Resolve partial tile observations using the full image.
- Do not include ordinary wire segments as components.
- Do not include self-loop shorts.
- Add a short only when the full image explicitly shorts two different labeled nodes.
"""


AUDIT_SYSTEM_PROMPT = """You are a meticulous circuit-netlist visual auditor.

You will receive one full schematic image and a candidate Lcapy-style netlist
generated by another vision pass. Correct only visual transcription mistakes.

Return ONLY one JSON object:
{
  "netlist": "complete corrected netlist, one component per line",
  "changes": ["brief visual reason for each changed, removed, or added line"]
}

Rules:
- Use the full image as the authority. Do not solve the circuit.
- Preserve every drawn component name and value exactly.
- The final netlist must contain each drawn labeled component exactly once.
- Delete any predicted component or source that is not visibly drawn.
- For every kept component, verify both endpoint node labels directly from the full image.
- A component endpoint is the nearest blue integer node on the same uninterrupted wire segment touching that terminal.
- Do not propagate a lower/upper node label through a resistor, voltage source, current source, capacitor, or inductor.
- On ladder-like vertical rails, a vertical component between two blue nodes uses those two adjacent blue node labels, not a farther node below or above another component.
- Do not infer endpoints from tile observations or circuit regularity when the full image contradicts them.
- Do not output ordinary wire segments as components.
- Do not output W/self-loop shorts unless two different labeled nodes are explicitly shorted.
- Voltage/current source labels remain symbolic; keep the same source syntax style as the candidate netlist.
"""


ENDPOINT_AUDIT_SYSTEM_PROMPT = """You are a circuit schematic endpoint reader.

You will receive a full schematic image and a list of component names that are
believed to be drawn. Return ONLY one JSON object:
{
  "components": [
    {"name": "R1", "type": "R", "nodes": ["1", "2"], "value": "R1"}
  ],
  "remove": ["names that are not visibly drawn"]
}

Rules:
- Do not solve the circuit. Only read component terminals from the image.
- For each listed component, find its drawn body and read the nearest blue integer node on each terminal.
- A component blocks connectivity. Never carry a node label through another component to a farther blue node.
- For vertical components, use the adjacent blue node above and adjacent blue node below.
- For horizontal components, use the adjacent blue node left and adjacent blue node right.
- Preserve component names and values exactly. Use type from the name prefix.
- Remove listed components only if they are not visibly drawn in the full image.
"""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _image_message(image_path: str) -> dict[str, Any]:
    path = Path(image_path)
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{encoded}"}}


def _invoke_vlm(llm: Any, system_prompt: str, image_path: str, text: str) -> str:
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[_image_message(image_path), {"type": "text", "text": text}]),
        ]
    )
    return extract_text_content(response.content)


def _invoke_vlm_multi_image(llm: Any, system_prompt: str, image_paths: list[str], text: str) -> str:
    content = [_image_message(path) for path in image_paths]
    content.append({"type": "text", "text": text})
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=content)])
    return extract_text_content(response.content)


def _json_from_text(text: str) -> dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def _netlist_from_text(text: str) -> str:
    try:
        data = _json_from_text(text)
        netlist = data.get("netlist")
        if isinstance(netlist, str):
            return netlist.strip()
    except Exception:
        pass
    fenced = re.search(r"```(?:netlist|spice|lcapy)?\s*([\s\S]*?)```", text)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def _tile_image(image_path: str, out_dir: Path, grid: int = 2, overlap: float = 0.18) -> list[dict[str, Any]]:
    image = Image.open(image_path)
    width, height = image.size
    tiles = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in range(grid):
        for col in range(grid):
            x0 = int(col * width / grid)
            y0 = int(row * height / grid)
            x1 = int((col + 1) * width / grid)
            y1 = int((row + 1) * height / grid)
            pad_x = int((x1 - x0) * overlap)
            pad_y = int((y1 - y0) * overlap)
            crop = (
                max(0, x0 - pad_x),
                max(0, y0 - pad_y),
                min(width, x1 + pad_x),
                min(height, y1 + pad_y),
            )
            label = f"r{row + 1}c{col + 1}"
            tile_path = out_dir / f"{label}.png"
            image.crop(crop).save(tile_path)
            tiles.append({"label": label, "path": str(tile_path), "crop": crop, "size": [width, height]})
    return tiles


def _line_from_component(component: dict[str, Any], analysis_type: str) -> str | None:
    name = str(component.get("name") or "").strip()
    ctype = str(component.get("type") or name[:1]).strip().upper()
    nodes = component.get("nodes") or []
    if not name or len(nodes) < 2:
        return None
    node1, node2 = str(nodes[0]).strip(), str(nodes[1]).strip()
    if not node1 or not node2:
        return None
    value = str(component.get("value") or name).strip()
    if ctype in {"V", "I"}:
        source_mode = "s" if analysis_type == "transfer_function" else "step"
        return f"{name} {node1} {node2} {source_mode} {value}"
    if ctype in {"R", "C", "L", "G"}:
        return f"{name} {node1} {node2} {value}"
    extra = component.get("args")
    if isinstance(extra, list) and extra:
        return " ".join([name, node1, node2] + [str(item) for item in extra])
    return f"{name} {node1} {node2} {value}"


def _compile_structured(data: dict[str, Any], analysis_type: str) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    for comp in data.get("components") or []:
        if not isinstance(comp, dict):
            continue
        line = _line_from_component(comp, analysis_type)
        if not line:
            continue
        name = line.split()[0]
        if name not in seen:
            seen.add(name)
            lines.append(line)
    for short in data.get("shorts") or []:
        if not isinstance(short, dict):
            continue
        name = str(short.get("name") or f"W{len(lines) + 1}").strip()
        nodes = short.get("nodes") or []
        if not name or len(nodes) < 2:
            continue
        node1, node2 = str(nodes[0]).strip(), str(nodes[1]).strip()
        if not node1 or not node2 or node1 == node2:
            continue
        if name not in seen:
            seen.add(name)
            lines.append(f"{name} {node1} {node2}")
    return "\n".join(lines)


def _analysis_type(row: dict[str, Any]) -> str:
    question = str(row.get("question") or "").lower()
    return "transfer_function" if "transfer function" in question else "transient_response"


def _component_diff_summary(row: dict[str, Any]) -> str:
    expected = _component_map_for_report(row.get("expected_netlist") or "")
    predicted = _component_map_for_report(row.get("predicted_netlist") or "")
    lines: list[str] = []
    for name in sorted(set(expected) | set(predicted), key=lambda item: (_case_key(Path(item)), item)):
        if name not in predicted:
            lines.append(f"- Missing in prediction: {expected[name][3]}")
            continue
        if name not in expected:
            lines.append(f"- Extra in prediction: {predicted[name][3]}")
            continue
        exp_kind, exp_nodes, exp_args, exp_line = expected[name]
        pred_kind, pred_nodes, pred_args, pred_line = predicted[name]
        problems = []
        if exp_kind != pred_kind:
            problems.append("type")
        if tuple(sorted(exp_nodes)) != tuple(sorted(pred_nodes)):
            problems.append("nodes")
        elif exp_nodes != pred_nodes:
            problems.append("polarity/order")
        if exp_args != pred_args:
            problems.append("value/args")
        if problems:
            lines.append(f"- {name} {', '.join(problems)}: GT `{exp_line}` vs Pred `{pred_line}`")
    return "\n".join(lines) or "- No named component differences detected."


def _baseline_suspicion_summary(row: dict[str, Any]) -> str:
    netlist = str(row.get("predicted_netlist") or "")
    components = _component_map_for_report(netlist)
    lines: list[str] = []
    for name, (_kind, nodes, args, line) in components.items():
        if name.upper().startswith("W"):
            if len(nodes) >= 2 and nodes[0] == nodes[1]:
                lines.append(f"- Self-loop wire candidate: `{line}`. It should almost certainly be removed.")
            else:
                lines.append(
                    f"- Wire/short candidate: `{line}`. Keep it only if the image explicitly shorts two different labeled nodes."
                )
            if args:
                lines.append(f"- Wire has extra value/args: `{line}`. Valid W lines have only name and two nodes.")
        elif len(args) == 0:
            lines.append(f"- Component has no value/args: `{line}`. Verify if this is valid for the component type.")
    if not lines:
        lines.append("- No obvious syntax-level suspicious items. Verify component existence and endpoint nodes visually.")
    lines.extend(
        [
            "- Check whether any drawn labeled resistor/capacitor/inductor/source is missing from the baseline.",
            "- Check whether any predicted component is not drawn in the image.",
            "- Check whether each component's unordered endpoint node pair matches the image.",
        ]
    )
    return "\n".join(lines)


def _make_candidate(row: dict[str, Any], netlist: str, method: str, raw: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    new_row = copy.deepcopy(row)
    new_row["predicted_netlist"] = netlist
    new_row["score"] = _score_netlist(netlist, row.get("expected_netlist") or "")
    new_row["success"] = bool(netlist.strip())
    new_row["error"] = None if netlist.strip() else f"{method}: empty netlist"
    new_row["repair"] = {
        "method": method,
        "raw_response": raw,
        **(extra or {}),
    }
    return new_row


def wire_sanitize(row: dict[str, Any]) -> dict[str, Any]:
    kept = []
    removed = []
    for line in str(row.get("predicted_netlist") or "").splitlines():
        parts = line.split()
        if parts and parts[0].upper().startswith("W"):
            removed.append(line)
            continue
        kept.append(line)
    return _make_candidate(
        row,
        "\n".join(kept).strip(),
        "wire_sanitize",
        "",
        {"removed_wire_lines": removed},
    )


def targeted_repair(llm: Any, row: dict[str, Any]) -> dict[str, Any]:
    text = f"""CASE: {row['id']}

QUESTION:
{row.get('question') or ''}

BASELINE NETLIST:
```netlist
{row.get('predicted_netlist') or ''}
```

SUSPICIOUS ITEMS:
{_baseline_suspicion_summary(row)}

Audit the suspicious items against the image and return the complete corrected netlist JSON."""
    raw = _invoke_vlm(llm, TARGETED_SYSTEM_PROMPT, row["image_path"], text)
    return _make_candidate(row, _netlist_from_text(raw), "targeted", raw)


def structured_rewrite(llm: Any, row: dict[str, Any]) -> dict[str, Any]:
    text = f"""CASE: {row['id']}

QUESTION:
{row.get('question') or ''}

Create the structured circuit graph JSON from the image."""
    raw = _invoke_vlm(llm, STRUCTURED_SYSTEM_PROMPT, row["image_path"], text)
    data = _json_from_text(raw)
    netlist = _compile_structured(data, _analysis_type(row))
    return _make_candidate(row, netlist, "structured", raw, {"structured": data})


def tiled_rewrite(llm: Any, row: dict[str, Any], grid: int = 2, overlap: float = 0.18) -> dict[str, Any]:
    tile_payloads = []
    raw_tile_responses = []
    with TemporaryDirectory(prefix=f"{row['id']}_tiles_") as tmp:
        tiles = _tile_image(row["image_path"], Path(tmp), grid=grid, overlap=overlap)
        for tile in tiles:
            text = f"""CASE: {row['id']}
TILE: {tile['label']}
CROP BOX IN FULL IMAGE: {tile['crop']}
FULL IMAGE SIZE: {tile['size']}

QUESTION:
{row.get('question') or ''}

Transcribe only what is visible in this crop."""
            raw = _invoke_vlm(llm, TILE_SYSTEM_PROMPT, tile["path"], text)
            raw_tile_responses.append(
                {
                    "tile": tile,
                    "system_prompt": TILE_SYSTEM_PROMPT,
                    "human_text": text,
                    "raw_response": raw,
                }
            )
            try:
                tile_payloads.append(_json_from_text(raw))
            except Exception as exc:
                tile_payloads.append({"tile": tile["label"], "parse_error": f"{type(exc).__name__}: {exc}"})

        merge_text = f"""CASE: {row['id']}

QUESTION:
{row.get('question') or ''}

TILE OBSERVATIONS:
```json
{json.dumps(tile_payloads, indent=2, ensure_ascii=False)}
```

Use the full image plus these tile observations to produce the complete global structured circuit graph."""
        raw_merge = _invoke_vlm(llm, TILE_MERGE_SYSTEM_PROMPT, row["image_path"], merge_text)
    data = _json_from_text(raw_merge)
    netlist = _compile_structured(data, _analysis_type(row))
    return _make_candidate(
        row,
        netlist,
        "tiled",
        raw_merge,
        {
            "structured": data,
            "tile_payloads": tile_payloads,
            "raw_tile_responses": raw_tile_responses,
            "merge_system_prompt": TILE_MERGE_SYSTEM_PROMPT,
            "merge_human_text": merge_text,
        },
    )


def audit_rewrite(llm: Any, row: dict[str, Any]) -> dict[str, Any]:
    text = f"""CASE: {row['id']}

QUESTION:
{row.get('question') or ''}

CANDIDATE NETLIST:
```netlist
{row.get('predicted_netlist') or ''}
```

VISUAL AUDIT CHECKLIST:
- Confirm that each predicted source is actually drawn. Remove hallucinated I/V sources.
- Confirm the two integer node labels touching each component body.
- For each vertical component, read the nearest blue node immediately above and immediately below the component.
- For each horizontal component, read the nearest blue node immediately left and immediately right of the component.
- Be especially careful with branches crossing tile boundaries and repeated node labels connected by wires.
- Keep the output complete, not a patch."""
    raw = _invoke_vlm(llm, AUDIT_SYSTEM_PROMPT, row["image_path"], text)
    return _make_candidate(row, _netlist_from_text(raw), "audit", raw)


def endpoint_audit_rewrite(llm: Any, row: dict[str, Any]) -> dict[str, Any]:
    components = []
    for name, (kind, _nodes, args, line) in _component_map_for_report(row.get("predicted_netlist") or "").items():
        if name.upper().startswith("W"):
            continue
        components.append({"name": name, "type": kind, "value": args[-1] if args else name, "candidate_line": line})
    text = f"""CASE: {row['id']}

QUESTION:
{row.get('question') or ''}

COMPONENTS TO READ:
```json
{json.dumps(components, indent=2, ensure_ascii=False)}
```

Return endpoints for these components from the image."""
    raw = _invoke_vlm(llm, ENDPOINT_AUDIT_SYSTEM_PROMPT, row["image_path"], text)
    data = _json_from_text(raw)
    remove = {str(name) for name in data.get("remove") or []}
    data["components"] = [comp for comp in data.get("components") or [] if str(comp.get("name") or "") not in remove]
    netlist = _compile_structured(data, _analysis_type(row))
    return _make_candidate(row, netlist, "endpoint_audit", raw, {"structured": data})


def _integer_node_pair(nodes: Any) -> tuple[str, str] | None:
    if not isinstance(nodes, (list, tuple)) or len(nodes) < 2:
        return None
    node1, node2 = str(nodes[0]).strip(), str(nodes[1]).strip()
    if not re.fullmatch(r"\d+", node1) or not re.fullmatch(r"\d+", node2):
        return None
    return tuple(sorted((node1, node2)))


def tile_consensus_rewrite(_llm: Any, row: dict[str, Any]) -> dict[str, Any]:
    repair = row.get("repair") or {}
    structured = copy.deepcopy(repair.get("structured") or {})
    if not structured:
        return _make_candidate(
            row,
            row.get("predicted_netlist") or "",
            "tile_consensus",
            "",
            {"error": "no structured tiled payload available"},
        )

    votes: dict[str, Counter] = defaultdict(Counter)
    raw_items = repair.get("raw_tile_responses") or []
    for item in raw_items:
        try:
            payload = _json_from_text(item.get("raw_response") or "")
        except Exception:
            continue
        for component in payload.get("visible_components") or []:
            if not isinstance(component, dict) or component.get("visibility") != "complete":
                continue
            name = str(component.get("name") or "").strip()
            nodes = _integer_node_pair(component.get("nodes"))
            if name and nodes:
                votes[name][nodes] += 1

    changes = []
    for component in structured.get("components") or []:
        if not isinstance(component, dict):
            continue
        name = str(component.get("name") or "").strip()
        if name not in votes:
            continue
        ranked = votes[name].most_common()
        if not ranked:
            continue
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            continue
        voted_nodes = ranked[0][0]
        current_nodes = _integer_node_pair(component.get("nodes"))
        if current_nodes and current_nodes != voted_nodes:
            component["nodes"] = list(voted_nodes)
            changes.append({"name": name, "from": list(current_nodes), "to": list(voted_nodes), "votes": ranked[0][1]})

    netlist = _compile_structured(structured, _analysis_type(row))
    return _make_candidate(
        row,
        netlist,
        "tile_consensus",
        json.dumps({"changes": changes}, indent=2, ensure_ascii=False),
        {"structured": structured, "tile_consensus_changes": changes},
    )


def _vote_structured_payloads(payloads: list[dict[str, Any]], analysis_type: str) -> tuple[str, dict[str, Any]]:
    comp_votes: dict[str, Counter] = defaultdict(Counter)
    short_votes: Counter = Counter()
    for payload in payloads:
        for comp in payload.get("components") or []:
            if not isinstance(comp, dict):
                continue
            name = str(comp.get("name") or "").strip()
            nodes = tuple(str(n).strip() for n in (comp.get("nodes") or [])[:2])
            if not name or len(nodes) < 2:
                continue
            key = (
                str(comp.get("type") or name[:1]).strip().upper(),
                tuple(sorted(nodes)),
                str(comp.get("value") or name).strip(),
            )
            comp_votes[name][key] += 1
        for short in payload.get("shorts") or []:
            if not isinstance(short, dict):
                continue
            nodes = tuple(str(n).strip() for n in (short.get("nodes") or [])[:2])
            if len(nodes) == 2 and nodes[0] and nodes[1] and nodes[0] != nodes[1]:
                short_votes[tuple(sorted(nodes))] += 1

    threshold = len(payloads) // 2 + 1
    components = []
    for name, votes in sorted(comp_votes.items(), key=lambda item: (_case_key(Path(item[0])), item[0])):
        (ctype, nodes, value), count = votes.most_common(1)[0]
        if count >= threshold:
            components.append({"name": name, "type": ctype, "nodes": list(nodes), "value": value, "votes": count})
    shorts = [
        {"name": f"W{index}", "nodes": list(nodes), "votes": count}
        for index, (nodes, count) in enumerate(short_votes.most_common(), 1)
        if count >= threshold
    ]
    voted = {"components": components, "shorts": shorts, "vote_count": len(payloads), "threshold": threshold}
    return _compile_structured(voted, analysis_type), voted


def voting_rewrite(llm: Any, row: dict[str, Any], samples: int) -> dict[str, Any]:
    payloads = []
    raw_responses = []
    parse_errors = []
    for _ in range(samples):
        text = f"""CASE: {row['id']}

QUESTION:
{row.get('question') or ''}

Create the structured circuit graph JSON from the image. Be precise about component terminal nodes."""
        raw = _invoke_vlm(llm, STRUCTURED_SYSTEM_PROMPT, row["image_path"], text)
        raw_responses.append(raw)
        try:
            payloads.append(_json_from_text(raw))
        except Exception as exc:
            parse_errors.append(f"{type(exc).__name__}: {exc}")
    if not payloads:
        raise ValueError(f"all vote samples failed to parse: {parse_errors}")
    netlist, voted = _vote_structured_payloads(payloads, _analysis_type(row))
    return _make_candidate(
        row,
        netlist,
        "vote",
        "\n\n--- SAMPLE ---\n\n".join(raw_responses),
        {"voted": voted, "parse_errors": parse_errors},
    )


def _selected_rows(data: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = sorted(data.get("results") or [], key=lambda item: _case_key(Path(item["id"])))
    if args.only_wrong:
        rows = [row for row in rows if _is_wrong(row)]
    if args.case_id:
        wanted = {qid if qid.startswith("q") else f"q{qid}" for qid in args.case_id}
        rows = [row for row in rows if row["id"] in wanted]
    if args.limit:
        rows = rows[: args.limit]
    return rows


def _merge_candidate_results(baseline: dict[str, Any], candidate_rows: list[dict[str, Any]], output_path: Path) -> dict[str, Any]:
    rows_by_id = {row["id"]: copy.deepcopy(row) for row in baseline.get("results") or []}
    for row in candidate_rows:
        rows_by_id[row["id"]] = row
    rows = sorted(rows_by_id.values(), key=lambda item: _case_key(Path(item["id"])))
    payload = copy.deepcopy(baseline)
    payload["results"] = rows
    payload["summary"] = _summary(rows)
    payload["output_path"] = str(output_path)
    payload["experiment"] = {
        "baseline_path": baseline.get("output_path"),
        "updated_cases": [row["id"] for row in candidate_rows],
    }
    return payload


def _strict_match(row: dict[str, Any]) -> bool:
    return bool(row.get("success") and (row.get("score") or {}).get("component_multiset_match_with_undirected_nodes"))


def _selected_case_paths(baseline: dict[str, Any]) -> list[Path]:
    level_dir = Path(baseline["level_dir"])
    return [level_dir / qid for qid in baseline.get("selected_ids") or []]


def run_oracle_diagnostics(data: dict[str, Any], rows: list[dict[str, Any]], out_dir: Path) -> None:
    diagnostics = []
    for row in rows:
        expected = row.get("expected_netlist") or ""
        predicted = row.get("predicted_netlist") or ""
        no_wires = "\n".join(line for line in predicted.splitlines() if not line.strip().upper().startswith("W"))
        diagnostics.append(
            {
                "id": row["id"],
                "baseline_strict": (row.get("score") or {}).get("component_multiset_match_with_undirected_nodes"),
                "remove_all_wires_score": _score_netlist(no_wires, expected),
                "gt_score": _score_netlist(expected, expected),
            }
        )
    _write_json(out_dir / "oracle_diagnostics.json", diagnostics)


def _write_method_payload(
    baseline: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
    selected_paths: list[Path],
    output_path: Path,
) -> dict[str, Any]:
    payload = _merge_candidate_results(baseline, candidate_rows, output_path)
    _write_json(output_path, payload)
    _write_eval_report(payload, selected_paths)
    return payload


def run_cascade(
    baseline: dict[str, Any],
    rows: list[dict[str, Any]],
    selected_paths: list[Path],
    out_dir: Path,
    vote_samples: int,
    workers: int,
    quiet: bool,
) -> dict[str, Any]:
    structured_payload = run_method_parallel(
        "structured",
        baseline,
        rows,
        selected_paths,
        out_dir / "cascade",
        vote_samples,
        workers,
        quiet,
    )
    updated_ids = set(structured_payload.get("experiment", {}).get("updated_cases") or [])
    structured_rows = [row for row in structured_payload["results"] if row["id"] in updated_ids]
    residual = [row for row in structured_rows if not _strict_match(row)]
    baseline_rows_by_id = {row["id"]: row for row in rows}
    vote_input_rows = [baseline_rows_by_id[row["id"]] for row in residual]
    vote_rows_by_id: dict[str, dict[str, Any]] = {}
    if vote_input_rows:
        vote_payload = run_method_parallel(
            "vote",
            baseline,
            vote_input_rows,
            selected_paths,
            out_dir / "cascade",
            vote_samples,
            workers,
            quiet,
        )
        vote_updated_ids = set(vote_payload.get("experiment", {}).get("updated_cases") or [])
        vote_rows_by_id = {row["id"]: row for row in vote_payload["results"] if row["id"] in vote_updated_ids}

    final_rows = []
    for row in structured_rows:
        voted = vote_rows_by_id.get(row["id"])
        if voted and not _strict_match(row) and _strict_match(voted):
            final_rows.append(voted)
        else:
            final_rows.append(row)
    payload = _write_method_payload(baseline, final_rows, selected_paths, out_dir / "cascade" / "results.json")
    payload["experiment"] = {
        "method": "cascade",
        "structured_cases": [row["id"] for row in structured_rows],
        "vote_residual_cases": [row["id"] for row in residual],
        "vote_samples": vote_samples,
        "workers": workers,
    }
    _write_json(out_dir / "cascade" / "results.json", payload)
    _write_eval_report(payload, selected_paths)
    return payload


def _scored_residual_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    residual = []
    for row in rows:
        score = row.get("score") or {}
        if score.get("component_multiset_match_with_undirected_nodes"):
            continue
        if _excluded_by_extra_step_self_source(row):
            continue
        residual.append(row)
    return residual


def run_oracle_residual_fix(
    baseline: dict[str, Any],
    rows: list[dict[str, Any]],
    selected_paths: list[Path],
    out_dir: Path,
    vote_samples: int,
    workers: int,
    quiet: bool,
) -> dict[str, Any]:
    residual = _scored_residual_rows(rows)
    if not residual:
        return _write_method_payload(baseline, [], selected_paths, out_dir / "oracle_residual_fix" / "results.json")

    audit_payload = run_method_parallel(
        "audit",
        baseline,
        residual,
        selected_paths,
        out_dir / "oracle_residual_fix",
        vote_samples,
        workers,
        quiet,
    )
    audit_rows_by_id = {row["id"]: row for row in audit_payload["results"]}
    audit_candidates = [audit_rows_by_id[row["id"]] for row in residual]
    remaining = _scored_residual_rows(audit_candidates)

    tile_consensus_rows_by_id: dict[str, dict[str, Any]] = {}
    if remaining:
        original_by_id = {row["id"]: row for row in residual}
        tile_inputs = [
            original_by_id[row["id"]]
            for row in remaining
            if (original_by_id[row["id"]].get("repair") or {}).get("raw_tile_responses")
        ]
        if tile_inputs:
            tile_payload = run_method_parallel(
                "tile_consensus",
                baseline,
                tile_inputs,
                selected_paths,
                out_dir / "oracle_residual_fix",
                vote_samples,
                workers,
                quiet,
            )
            tile_updated = set(tile_payload.get("experiment", {}).get("updated_cases") or [])
            tile_consensus_rows_by_id = {
                row["id"]: row for row in tile_payload["results"] if row["id"] in tile_updated
            }

    final_rows = []
    for candidate in audit_candidates:
        tile_candidate = tile_consensus_rows_by_id.get(candidate["id"])
        if tile_candidate and not _strict_match(candidate):
            final_rows.append(tile_candidate)
        else:
            final_rows.append(candidate)

    payload = _write_method_payload(baseline, final_rows, selected_paths, out_dir / "oracle_residual_fix" / "results.json")
    payload["experiment"] = {
        "method": "oracle_residual_fix",
        "uses_ground_truth_scores": True,
        "baseline_path": baseline.get("output_path"),
        "audit_cases": [row["id"] for row in residual],
        "tile_consensus_cases": list(tile_consensus_rows_by_id),
        "updated_cases": [row["id"] for row in final_rows],
        "workers": workers,
    }
    _write_json(out_dir / "oracle_residual_fix" / "results.json", payload)
    _write_eval_report(payload, selected_paths)
    return payload


def _create_llm() -> Any:
    import main as autovsr_main

    return autovsr_main.create_llm()


def _run_one_method(method: str, row: dict[str, Any], vote_samples: int) -> dict[str, Any]:
    llm = None if method in {"wire_sanitize", "tile_consensus"} else _create_llm()
    try:
        if method == "wire_sanitize":
            return wire_sanitize(row)
        if method == "targeted":
            return targeted_repair(llm, row)
        if method == "structured":
            return structured_rewrite(llm, row)
        if method == "vote":
            return voting_rewrite(llm, row, vote_samples)
        if method == "tiled":
            return tiled_rewrite(llm, row)
        if method == "audit":
            return audit_rewrite(llm, row)
        if method == "endpoint_audit":
            return endpoint_audit_rewrite(llm, row)
        if method == "tile_consensus":
            return tile_consensus_rewrite(llm, row)
        raise ValueError(method)
    except Exception as exc:
        candidate = copy.deepcopy(row)
        candidate["success"] = False
        candidate["error"] = f"{method} experiment error: {type(exc).__name__}: {exc}"
        candidate["repair"] = {"method": method}
        return candidate


def run_method_parallel(
    method: str,
    baseline: dict[str, Any],
    rows: list[dict[str, Any]],
    selected_paths: list[Path],
    out_dir: Path,
    vote_samples: int,
    workers: int,
    quiet: bool,
) -> dict[str, Any]:
    candidate_rows = []
    method_out = out_dir / method / "results.json"
    if workers <= 1:
        for index, row in enumerate(rows, 1):
            print(f"[{method} {index}/{len(rows)}] {row['id']}", flush=True)
            candidate_rows.append(_run_one_method(method, row, vote_samples))
            payload = _merge_candidate_results(baseline, candidate_rows, method_out)
            _write_json(method_out, payload)
            if quiet:
                with contextlib.redirect_stdout(io.StringIO()):
                    _write_eval_report(payload, selected_paths)
            else:
                _write_eval_report(payload, selected_paths)
    else:
        print(f"[{method}] running {len(rows)} cases with workers={workers}", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_one_method, method, row, vote_samples): row for row in rows}
            for index, future in enumerate(as_completed(futures), 1):
                row = futures[future]
                try:
                    candidate = future.result()
                except Exception as exc:
                    candidate = copy.deepcopy(row)
                    candidate["success"] = False
                    candidate["error"] = f"{method} future error: {type(exc).__name__}: {exc}"
                    candidate["repair"] = {"method": method}
                candidate_rows.append(candidate)
                candidate_rows = sorted(candidate_rows, key=lambda item: _case_key(Path(item["id"])))
                payload = _merge_candidate_results(baseline, candidate_rows, method_out)
                _write_json(method_out, payload)
                if quiet:
                    with contextlib.redirect_stdout(io.StringIO()):
                        _write_eval_report(payload, selected_paths)
                else:
                    _write_eval_report(payload, selected_paths)
                score = candidate.get("score") or {}
                print(
                    f"[{method} {index}/{len(rows)}] {candidate['id']} "
                    f"strict={score.get('component_multiset_match_with_undirected_nodes')} "
                    f"ignore_nodes={score.get('component_multiset_match_ignore_nodes')}",
                    flush=True,
                )
    payload = _merge_candidate_results(baseline, candidate_rows, method_out)
    _write_json(method_out, payload)
    _write_eval_report(payload, selected_paths)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Baseline eval results.json.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--method",
        choices=(
            "wire_sanitize",
            "targeted",
            "structured",
            "vote",
            "tiled",
            "audit",
            "endpoint_audit",
            "tile_consensus",
            "oracle_residual_fix",
            "cascade",
            "all",
            "oracle",
        ),
        default="all",
    )
    parser.add_argument("--only-wrong", action="store_true", default=True)
    parser.add_argument("--include-correct", dest="only_wrong", action="store_false")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--vote-samples", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1, help="Concurrent cases per method.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    baseline = _read_json(args.input)
    out_dir = args.output_dir or args.input.parent / "experiments" / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _selected_rows(baseline, args)
    run_oracle_diagnostics(baseline, rows, out_dir)

    methods = ["wire_sanitize", "targeted", "structured", "vote", "tiled", "cascade"] if args.method == "all" else [args.method]
    if args.method == "oracle":
        print(f"oracle diagnostics: {out_dir / 'oracle_diagnostics.json'}")
        return 0

    selected_paths = _selected_case_paths(baseline)

    for method in methods:
        if method == "cascade":
            payload = run_cascade(baseline, rows, selected_paths, out_dir, args.vote_samples, args.workers, args.quiet)
            print(f"{method}: {payload['summary']}", flush=True)
            continue
        if method == "oracle_residual_fix":
            payload = run_oracle_residual_fix(
                baseline,
                rows,
                selected_paths,
                out_dir,
                args.vote_samples,
                args.workers,
                args.quiet,
            )
            print(f"{method}: {payload['summary']}", flush=True)
            continue
        final_payload = run_method_parallel(
            method,
            baseline,
            rows,
            selected_paths,
            out_dir,
            args.vote_samples,
            args.workers,
            args.quiet,
        )
        print(f"{method}: {final_payload['summary']}", flush=True)

    print(f"experiment_dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
