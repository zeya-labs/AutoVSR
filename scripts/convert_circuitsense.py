#!/usr/bin/env python3
"""Convert CircuitSense-style records to AutoVSR batch JSON.

The converter is intentionally field-name tolerant because local exports from
Hugging Face datasets may be JSON, JSONL, or CSV with slightly different column
names. Image files are not copied; image paths are resolved relative to the
input file or an optional image root. Reference netlists may be read for dataset
type inference, but they are never emitted into the AutoVSR evaluation JSON.

It also supports the native CircuitSense Analysis directory layouts:
`Analysis/curated/level*/<sample_folder>/q*_question.txt`, `q*_a.txt`,
`q*_category.txt`, and `q*_image.png`; and
`Analysis/synthetic/level*/q*/q*_question.txt`, `q*_ta.txt`,
`q*_netlist.txt`, and `q*_image.png`.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


FIELD_ALIASES = {
    "id": ["id", "data_id", "question_id", "sample_id", "uid"],
    "question": ["question", "query", "prompt", "instruction"],
    "answer": ["answer", "expected_answer", "label", "target", "ground_truth", "gt"],
    "image_path": ["image_path", "image", "img", "figure", "file_name", "filename", "path"],
    "netlist": ["netlist", "circuit", "spice", "netlist_code"],
    "type": ["type", "circuit_type", "problem_type"],
    "task": ["task", "analysis_type", "category"],
    "level": ["level", "difficulty"],
    "source": ["source", "split"],
}


def _first(record: Dict[str, Any], field: str) -> Optional[Any]:
    for key in FIELD_ALIASES[field]:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def _load_records(path: Path) -> List[Dict[str, Any]]:
    if path.is_dir():
        return _load_analysis_directory(path)

    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("samples", "data", "train", "test", "validation"):
                if isinstance(data.get(key), list):
                    return data[key]
        raise ValueError(f"Unsupported JSON structure in {path}")
    if suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    if suffix == ".parquet":
        raise ValueError("Parquet input requires pyarrow. Export to JSON/JSONL/CSV or install pyarrow.")
    raise ValueError(f"Unsupported input extension: {suffix}")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _load_analysis_directory(root: Path) -> List[Dict[str, Any]]:
    records = []
    for question_path in sorted(root.rglob("*_question.txt")):
        folder = question_path.parent
        stem = question_path.name[: -len("_question.txt")]
        rel_parts = folder.relative_to(root).parts
        level = next((part for part in rel_parts if part.startswith("level")), "unknown")
        answer_path = _select_analysis_answer_path(folder, stem, level)
        image_path = folder / f"{stem}_image.png"
        if not image_path.exists():
            svg_path = folder / f"{stem}_image.svg"
            image_path = svg_path if svg_path.exists() else image_path
        if not answer_path.exists() or not image_path.exists():
            continue

        category_path = folder / f"{stem}_category.txt"
        mc_path = folder / f"{stem}_mc.txt"
        ta_path = folder / f"{stem}_ta.txt"
        der_path = folder / f"{stem}_der.txt"
        netlist_path = folder / f"{stem}_netlist.txt"
        netlist = _read_text(netlist_path) if netlist_path.exists() else ""

        source = _infer_source(root, folder)
        folder_name = folder.name
        sample_id = f"{level}_{folder_name}_{stem}"
        path_hint = " ".join((*root.parts, *rel_parts))

        record = {
            "id": sample_id,
            "data_id": stem,
            "folder": folder_name,
            "question": _read_text(question_path),
            "answer": _read_text(answer_path),
            "image_path": str(image_path),
            "source": source,
            "level": level,
            "task": _infer_task(_read_text(question_path), ta_path),
            "type": _infer_type(path_hint, category_path, netlist, image_path),
        }
        if netlist:
            record["netlist"] = netlist
        if category_path.exists():
            record["category"] = _read_text(category_path)
        if mc_path.exists():
            record["choices"] = _read_text(mc_path)
        if ta_path.exists():
            record["target_analysis"] = _read_text(ta_path)
        if der_path.exists():
            record["derivation"] = _read_text(der_path)
        records.append(record)
    return records


def _select_analysis_answer_path(folder: Path, stem: str, level: str) -> Path:
    if level == "level5_bd":
        candidates = (
            folder / f"{stem}_ta_exact.txt",
            folder / f"{stem}_ta_high.txt",
            folder / f"{stem}_ta.txt",
            folder / f"{stem}_a.txt",
        )
    else:
        candidates = (
            folder / f"{stem}_a.txt",
            folder / f"{stem}_ta.txt",
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _infer_source(root: Path, folder: Path) -> str:
    parts = {part.lower() for part in (*root.parts, *folder.parts)}
    if "synthetic" in parts:
        return "CircuitSense_synthetic"
    if "curated" in parts:
        return "CircuitSense_curated"
    return "CircuitSense"


def _infer_task(question: str, ta_path: Path) -> str:
    text = question.lower()
    if ta_path.exists():
        text += " " + _read_text(ta_path).lower()
    if any(term in text for term in ["matrix", "two-port", "transmission parameter"]):
        return "other"
    if "transfer" in text or "gain" in text or "transmission function" in text:
        return "transfer_function"
    if "transient" in text or "s-domain" in text or "node voltage" in text or "current" in text:
        return "transient_response"
    return "unknown"


def _infer_type(path_hint: str, category_path: Path, netlist: str = "", image_path: Optional[Path] = None) -> str:
    text = path_hint.lower()
    if category_path.exists():
        text += " " + _read_text(category_path).lower()
    for idx in range(1, 6):
        if f"type{idx}" in text or f"type {idx}" in text:
            return f"type{idx}"

    synthetic_level_types = {
        "level0": "type1",
        "level1": "type2",
        "level2": "type3",
        "level4": "type4",
        "level5_bd": "type5",
    }
    for level, sample_type in synthetic_level_types.items():
        if level in text:
            return sample_type

    if "level5_bd" in text or (image_path and image_path.suffix.lower() == ".svg"):
        return "type5"

    # Curated folder names encode broad CircuitSense categories.
    folder_hints = {
        "zerofive": "type2",
        "zero": "type1",
        "one": "type3",
        "two": "type4",
    }
    for hint, typ in folder_hints.items():
        if hint in text:
            return typ

    # Synthetic symbolic-expression folders do not always encode the paper type,
    # so use a conservative component-level fallback.
    normalized_netlist = netlist.lower()
    if any(token in normalized_netlist for token in (" opamp", " laplace", " module", "block")):
        return "type4"
    element_prefixes = {
        line.strip()[:1].upper()
        for line in netlist.splitlines()
        if line.strip() and not line.lstrip().startswith(("*", "#", ";", "."))
    }
    if element_prefixes & {"E", "F", "G", "H"}:
        return "type3"
    if element_prefixes & {"C", "L"}:
        return "type2"
    if element_prefixes & {"R"}:
        return "type1"
    return "unknown"


def _resolve_image(raw: Any, input_path: Path, image_root: Optional[Path]) -> str:
    if raw is None:
        return ""
    raw_text = str(raw)
    if raw_text.startswith("{") and "path" in raw_text:
        try:
            parsed = json.loads(raw_text.replace("'", '"'))
            raw_text = str(parsed.get("path", raw_text))
        except Exception:
            pass
    candidate = Path(raw_text)
    if candidate.is_absolute():
        return str(candidate)
    if image_root:
        rooted = image_root / candidate
        if rooted.exists():
            return str(rooted)
    sibling = input_path.parent / candidate
    return str(sibling if sibling.exists() else candidate)


def convert(
    records: Iterable[Dict[str, Any]],
    input_path: Path,
    image_root: Optional[Path],
    include_tasks: Optional[set[str]] = None,
    include_types: Optional[set[str]] = None,
    include_levels: Optional[set[str]] = None,
    include_sources: Optional[set[str]] = None,
) -> Dict[str, Any]:
    samples = []
    for idx, record in enumerate(records):
        question = _first(record, "question")
        answer = _first(record, "answer")
        image_path = _resolve_image(_first(record, "image_path"), input_path, image_root)
        if not question or answer in (None, "") or not image_path:
            continue

        sample_id = _first(record, "id") or f"sample_{idx}"
        task = _first(record, "task") or "transfer_function"
        task = str(task)
        if include_tasks and task not in include_tasks:
            continue
        source = str(_first(record, "source") or "CircuitSense")
        level = str(_first(record, "level") or "unknown")
        sample_type = str(_first(record, "type") or "unknown")
        if include_types and sample_type not in include_types:
            continue
        if include_levels and level not in include_levels:
            continue
        if include_sources and source not in include_sources:
            continue
        sample = {
            "id": str(sample_id),
            "question": str(question),
            "answer": str(answer),
            "image_path": image_path,
            "source": source,
            "level": level,
            "type": sample_type,
            "task": task,
        }
        samples.append(sample)

    return {
        "dataset_name": "CircuitSense_converted_for_AutoVSR",
        "version": "1.0",
        "description": "Converted CircuitSense-style records for AutoVSR batch evaluation.",
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CircuitSense JSON/JSONL/CSV export or Analysis directory")
    parser.add_argument("--image-root", type=Path, help="Optional root directory for relative image paths")
    parser.add_argument("--output", "-o", type=Path, required=True, help="AutoVSR batch JSON output")
    parser.add_argument("--max-samples", type=int, help="Optional cap after filtering valid samples")
    parser.add_argument(
        "--include-task",
        action="append",
        dest="include_tasks",
        help="Only keep samples with this inferred/provided task. Repeatable.",
    )
    parser.add_argument(
        "--include-type",
        action="append",
        dest="include_types",
        help="Only keep samples with this inferred/provided type, such as type1 or type5. Repeatable.",
    )
    parser.add_argument(
        "--include-level",
        action="append",
        dest="include_levels",
        help="Only keep samples with this level, such as level0 or level5_bd. Repeatable.",
    )
    parser.add_argument(
        "--include-source",
        action="append",
        dest="include_sources",
        help="Only keep samples with this source label. Repeatable.",
    )
    args = parser.parse_args()

    records = _load_records(args.input)
    converted = convert(
        records,
        args.input,
        args.image_root,
        set(args.include_tasks or []),
        set(args.include_types or []),
        set(args.include_levels or []),
        set(args.include_sources or []),
    )
    if args.max_samples:
        converted["samples"] = converted["samples"][:args.max_samples]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(converted, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(converted['samples'])} samples to {args.output}")


if __name__ == "__main__":
    main()
