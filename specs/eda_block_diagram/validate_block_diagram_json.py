#!/usr/bin/env python3
"""Validate EDA block diagram JSON without third-party dependencies.

This is not a full JSON Schema validator. It checks the invariants that matter
most for VLM-generated graph outputs: required top-level sections, unique IDs,
and cross-references between nodes, ports, edges, groups, texts, and annotations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REQUIRED_TOP_LEVEL = {"schema", "source", "diagram", "nodes", "ports", "edges", "groups"}
VALID_NODE_KINDS = {"functional_block", "operator", "source_sink", "connector", "subsystem", "unknown"}
VALID_PORT_DIRECTIONS = {"in", "out", "inout", "unknown"}
VALID_EDGE_KINDS = {"signal", "feedback", "bus", "power_or_clock", "physical", "annotation_link", "unknown"}
VALID_EDGE_DIRECTIONS = {"directed", "undirected", "unknown"}
VALID_GROUP_KINDS = {"region", "subsystem_boundary", "chip_or_board", "loop_annotation", "lane_or_domain"}


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("top-level JSON value must be an object")
    return data


def _require_object_list(data: dict[str, Any], key: str, errors: list[str]) -> list[dict[str, Any]]:
    value = data.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        errors.append(f"`{key}` must be a list")
        return []
    out: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if isinstance(item, dict):
            out.append(item)
        else:
            errors.append(f"`{key}[{index}]` must be an object")
    return out


def _collect_ids(items: list[dict[str, Any]], section: str, errors: list[str]) -> set[str]:
    ids: set[str] = set()
    for index, item in enumerate(items):
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            errors.append(f"`{section}[{index}].id` must be a non-empty string")
            continue
        if item_id in ids:
            errors.append(f"duplicate id in `{section}`: {item_id}")
        ids.add(item_id)
    return ids


def _check_confidence(item: dict[str, Any], path: str, errors: list[str]) -> None:
    if "confidence" not in item:
        return
    value = item["confidence"]
    if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
        errors.append(f"`{path}.confidence` must be a number in [0, 1]")


def _check_bbox(item: dict[str, Any], path: str, errors: list[str]) -> None:
    bbox = item.get("bbox")
    if bbox is None:
        return
    if not isinstance(bbox, dict):
        errors.append(f"`{path}.bbox` must be an object")
        return
    for key in ("x", "y", "w", "h"):
        if not isinstance(bbox.get(key), (int, float)):
            errors.append(f"`{path}.bbox.{key}` must be numeric")
    if isinstance(bbox.get("w"), (int, float)) and bbox["w"] < 0:
        errors.append(f"`{path}.bbox.w` must be non-negative")
    if isinstance(bbox.get("h"), (int, float)) and bbox["h"] < 0:
        errors.append(f"`{path}.bbox.h` must be non-negative")


def validate(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    missing = sorted(REQUIRED_TOP_LEVEL - set(data))
    for key in missing:
        errors.append(f"missing top-level key `{key}`")
    if data.get("schema") != "eda_block_diagram.v0.1":
        errors.append("`schema` must be `eda_block_diagram.v0.1`")

    source = data.get("source")
    if not isinstance(source, dict) or not source.get("image_path"):
        errors.append("`source.image_path` is required")
    diagram = data.get("diagram")
    if not isinstance(diagram, dict) or not diagram.get("view_type"):
        errors.append("`diagram.view_type` is required")

    nodes = _require_object_list(data, "nodes", errors)
    ports = _require_object_list(data, "ports", errors)
    edges = _require_object_list(data, "edges", errors)
    groups = _require_object_list(data, "groups", errors)
    texts = _require_object_list(data, "texts", errors)
    annotations = _require_object_list(data, "annotations", errors)

    node_ids = _collect_ids(nodes, "nodes", errors)
    port_ids = _collect_ids(ports, "ports", errors)
    edge_ids = _collect_ids(edges, "edges", errors)
    group_ids = _collect_ids(groups, "groups", errors)
    text_ids = _collect_ids(texts, "texts", errors)
    annotation_ids = _collect_ids(annotations, "annotations", errors)
    all_ids = node_ids | port_ids | edge_ids | group_ids | text_ids | annotation_ids

    duplicated_across_sections = (
        sum(len(ids) for ids in [node_ids, port_ids, edge_ids, group_ids, text_ids, annotation_ids]) - len(all_ids)
    )
    if duplicated_across_sections:
        errors.append("IDs must be globally unique across nodes, ports, edges, groups, texts, and annotations")

    for index, node in enumerate(nodes):
        path = f"nodes[{index}]"
        if node.get("kind") not in VALID_NODE_KINDS:
            errors.append(f"`{path}.kind` must be one of {sorted(VALID_NODE_KINDS)}")
        if "label" not in node:
            errors.append(f"`{path}.label` is required, use null if unlabeled")
        for port_id in node.get("ports") or []:
            if port_id not in port_ids:
                errors.append(f"`{path}.ports` references missing port `{port_id}`")
        _check_bbox(node, path, errors)
        _check_confidence(node, path, errors)

    for index, port in enumerate(ports):
        path = f"ports[{index}]"
        if port.get("node") not in node_ids:
            errors.append(f"`{path}.node` references missing node `{port.get('node')}`")
        if port.get("direction") not in VALID_PORT_DIRECTIONS:
            errors.append(f"`{path}.direction` must be one of {sorted(VALID_PORT_DIRECTIONS)}")
        _check_bbox(port, path, errors)
        _check_confidence(port, path, errors)

    endpoint_ids = node_ids | port_ids
    for index, edge in enumerate(edges):
        path = f"edges[{index}]"
        if edge.get("kind") not in VALID_EDGE_KINDS:
            errors.append(f"`{path}.kind` must be one of {sorted(VALID_EDGE_KINDS)}")
        if edge.get("direction") not in VALID_EDGE_DIRECTIONS:
            errors.append(f"`{path}.direction` must be one of {sorted(VALID_EDGE_DIRECTIONS)}")
        if edge.get("source") not in endpoint_ids:
            errors.append(f"`{path}.source` references missing node/port `{edge.get('source')}`")
        if edge.get("target") not in endpoint_ids:
            errors.append(f"`{path}.target` references missing node/port `{edge.get('target')}`")
        for point_index, point in enumerate(edge.get("path") or []):
            if not isinstance(point, dict) or not isinstance(point.get("x"), (int, float)) or not isinstance(point.get("y"), (int, float)):
                errors.append(f"`{path}.path[{point_index}]` must contain numeric x/y")
        _check_confidence(edge, path, errors)

    for index, group in enumerate(groups):
        path = f"groups[{index}]"
        if group.get("kind") not in VALID_GROUP_KINDS:
            errors.append(f"`{path}.kind` must be one of {sorted(VALID_GROUP_KINDS)}")
        contains = group.get("contains")
        if not isinstance(contains, dict):
            errors.append(f"`{path}.contains` must be an object")
            continue
        for node_id in contains.get("nodes") or []:
            if node_id not in node_ids:
                errors.append(f"`{path}.contains.nodes` references missing node `{node_id}`")
        for edge_id in contains.get("edges") or []:
            if edge_id not in edge_ids:
                errors.append(f"`{path}.contains.edges` references missing edge `{edge_id}`")
        for nested_group_id in contains.get("groups") or []:
            if nested_group_id not in group_ids:
                errors.append(f"`{path}.contains.groups` references missing group `{nested_group_id}`")
        for port_id in group.get("ports") or []:
            if port_id not in port_ids:
                errors.append(f"`{path}.ports` references missing port `{port_id}`")
        _check_bbox(group, path, errors)
        _check_confidence(group, path, errors)

    for index, text in enumerate(texts):
        path = f"texts[{index}]"
        if not isinstance(text.get("text"), str):
            errors.append(f"`{path}.text` must be a string")
        attached_to = text.get("attached_to")
        if attached_to is not None and attached_to not in all_ids:
            errors.append(f"`{path}.attached_to` references missing id `{attached_to}`")
        _check_bbox(text, path, errors)
        _check_confidence(text, path, errors)

    for index, annotation in enumerate(annotations):
        path = f"annotations[{index}]"
        for related_id in annotation.get("related_to") or []:
            if related_id not in all_ids:
                errors.append(f"`{path}.related_to` references missing id `{related_id}`")
        _check_bbox(annotation, path, errors)
        _check_confidence(annotation, path, errors)

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path)
    args = parser.parse_args()

    try:
        data = _load(args.json_path)
    except Exception as exc:
        print(f"invalid JSON: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    errors = validate(data)
    if errors:
        print(f"{args.json_path}: INVALID ({len(errors)} errors)", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"{args.json_path}: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
