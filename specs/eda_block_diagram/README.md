# EDA Block Diagram JSON Spec

This directory contains the working contract for recognizing real EDA/system
block diagrams in `eda-problem-6/datasets`.

Files:

- `standard.md`: human-readable representation standard.
- `survey_notes.md`: dataset survey notes and observed diagram classes.
- `schema.json`: machine-readable Draft 2020-12 JSON Schema.
- `validate_block_diagram_json.py`: zero-dependency structural validator for
  VLM outputs.
- `examples/hidden_eda_tests_015.partial.json`: partial annotation of
  `Hidden/images/EDA_TESTs_015.jpg`.

Validate an output:

```bash
python specs/eda_block_diagram/validate_block_diagram_json.py \
  specs/eda_block_diagram/examples/hidden_eda_tests_015.partial.json
```

Expected result:

```text
specs/eda_block_diagram/examples/hidden_eda_tests_015.partial.json: OK
```

The validator checks the invariants most likely to break in VLM output:

- required top-level sections
- globally unique IDs
- node-to-port references
- edge source/target references
- group containment references
- text and annotation references
- basic bbox and confidence formats

It intentionally does not require full coordinates or exhaustive annotations,
because early VLM passes may produce useful semantic graphs before exact layout
is stable.

Run VLM recognition on one real image:

```bash
python scripts/recognize_eda_block_diagram.py \
  --image eda-problem-6/datasets/Hidden/images/EDA_TESTs_015.jpg \
  --output-dir output/eda_block_diagram_probe_hidden015 \
  --mode graph \
  --min-nodes 8 \
  --min-edges 8
```

Each run creates a self-contained output directory:

- `<case_id>.json`: normalized diagram JSON in this standard.
- `<case_id>.graph.svg`: Graphviz-rendered semantic topology reconstructed
  from the JSON, useful for side-by-side visual audit against the source image.
- `<case_id>.graph.dot`: DOT source used to generate the graph SVG.
- `<case_id>.raw.txt`: raw VLM response, including audit response when
  `--audit` is enabled.
- `<case_id>.meta.json`: per-case runtime metadata, validation errors, and
  counts.
- `summary.json`: aggregate run summary.
- `index.html`: visual browsing report with the original image, validation
  state, nodes, edges, groups, warnings, links to the raw artifacts, and a
  draggable reconstructed graph editor. Drag nodes to correct the layout and
  use `Export edited SVG` to save the adjusted view.

Use `--mode graph` for dense figures or models with small output-token limits:
it emits nodes/groups and direct node-to-node edges without ports or bboxes. Use
`--mode compact` when you want some ports and major bboxes, and `--mode full`
only when the model has enough output budget.

Run a second visual audit/refinement pass:

```bash
python scripts/recognize_eda_block_diagram.py \
  --image eda-problem-6/datasets/Hidden/images/EDA_TESTs_015.jpg \
  --output-dir output/eda_block_diagram_probe_hidden015_graph_audit \
  --mode graph \
  --min-nodes 8 \
  --min-edges 8 \
  --audit
```

Smoke-tested commands:

```bash
python scripts/recognize_eda_block_diagram.py \
  --image eda-problem-6/datasets/Hidden/images/EDA_TESTs_015.jpg \
  --output-dir output/eda_block_diagram_probe_hidden015_graph \
  --mode graph \
  --min-nodes 8 \
  --min-edges 8

python scripts/recognize_eda_block_diagram.py \
  --image eda-problem-6/datasets/Public/benchmark/images/EDA_TEST_015.jpg \
  --output-dir output/eda_block_diagram_probe_public015_graph \
  --mode graph \
  --min-nodes 5 \
  --min-edges 5

python scripts/recognize_eda_block_diagram.py \
  --image eda-problem-6/datasets/Hidden/images/EDA_TESTs_015.jpg \
  --output-dir output/eda_block_diagram_probe_hidden015_graph_audit3 \
  --mode graph \
  --min-nodes 8 \
  --min-edges 8 \
  --audit

python scripts/recognize_eda_block_diagram.py \
  --input-dir eda-problem-6/datasets/Hidden/images \
  --start 0 \
  --limit 3 \
  --workers 1 \
  --output-dir output/eda_block_diagram_hidden_first3_graph \
  --mode graph \
  --min-nodes 3 \
  --min-edges 2
```

Run a larger batch with concurrency:

```bash
python scripts/recognize_eda_block_diagram.py \
  --input-dir eda-problem-6/datasets/Hidden/images \
  --limit 80 \
  --workers 4 \
  --output-dir output/eda_block_diagram_hidden_graph \
  --mode graph \
  --min-nodes 3 \
  --min-edges 2 \
  --quiet
```

Open `output/eda_block_diagram_hidden_graph/index.html` to review results. The
HTML report is an audit aid; `valid` means the JSON is structurally valid, not
that every semantic edge is correct.

To save dragged graph edits back to disk, serve the report through the local
writer:

```bash
python scripts/serve_eda_block_report.py \
  --output-dir output/eda_block_diagram_hidden_graph \
  --port 8765
```

Then open `http://127.0.0.1:8765/`. Drag nodes in the reconstructed graph and
click `Save`; the server writes `<case_id>.graph.layout.json` and
`<case_id>.graph.edited.svg` in that output directory.
The editor supports selecting nodes/edges, dragging nodes, adding nodes,
connecting two selected nodes, deleting the selected node or edge, and editing
the selected label/kind before saving.

Observed caveat: graph mode is structurally reliable under the current
`qwen3-vl-plus` 4096-token output budget, but it can still make semantic
endpoint mistakes on dense feedback diagrams. For example, it may recover all
major blocks in a control diagram while misassigning one feedback label. Use a
second visual audit/refinement pass before treating the graph as final. The
audit pass fixed the z/x/y versus phi/theta/psi feedback split on
`Hidden/images/EDA_TESTs_015.jpg`, but loop annotations in PLL-style diagrams
can still be over-interpreted as feedback edges and should be reviewed.

Current smoke-test status:

- Hidden first 3 graph mode: `3/3` structurally valid.
- Hidden `EDA_TESTs_015` graph+audit: structurally valid, feedback split
  corrected.
- Public benchmark `EDA_TEST_015` graph+audit: structurally valid; loop
  annotations are normalized away from signal feedback edges when labeled
  `Loop I`, `Loop II`, etc.
