# EDA Block Diagram JSON Standard

Version: draft-0.1

This document defines a JSON representation for real EDA/system block diagrams
such as `eda-problem-6/datasets/Hidden/images/EDA_TESTs_015.jpg` and the public
benchmark images. These figures are not ordinary circuit schematics: they mix
signal-flow topology, block semantics, mathematical operators, labels,
hierarchy, annotations, and sometimes screenshot or paper-figure artifacts.

The goal is to preserve all information needed for downstream reasoning while
remaining feasible for a VLM to emit.

## Design Goals

1. Preserve both structure and semantics.
   A box labeled `PID For altitude` is not only a rectangle; it is a functional
   block with text, ports, and signal-flow relations.

2. Separate signal connectivity from visual annotation.
   A solid arrow carrying `U1` is a signal edge. A dashed arrow labeled `Loop I`
   can be an explanatory annotation and must not be forced into the signal
   graph.

3. Support hierarchy.
   Dashed rectangles, colored regions, chip boundaries, subsystems, and named
   control loops should be represented as groups containing blocks and edges.

4. Preserve uncertainty.
   If a symbol is visible but ambiguous, the JSON should record the best guess,
   evidence text, and confidence instead of silently dropping it.

5. Keep coordinates optional but recommended.
   Bounding boxes are essential for auditing and correction, but the semantic
   graph should still be useful if some coordinates are missing.

## Top-Level Object

```json
{
  "schema": "eda_block_diagram.v0.1",
  "source": {
    "image_path": "eda-problem-6/datasets/Hidden/images/EDA_TESTs_015.jpg",
    "image_width": 1546,
    "image_height": 768
  },
  "diagram": {
    "title": null,
    "domain": ["control_system", "robotics"],
    "view_type": "block_diagram"
  },
  "nodes": [],
  "ports": [],
  "edges": [],
  "groups": [],
  "texts": [],
  "annotations": [],
  "legend": [],
  "warnings": []
}
```

Required top-level fields: `schema`, `source`, `diagram`, `nodes`, `ports`,
`edges`, `groups`.

Optional top-level fields: `texts`, `annotations`, `legend`, `warnings`.

## Coordinate Convention

Coordinates use image pixels with origin at the top-left corner.

```json
{"bbox": {"x": 226, "y": 20, "w": 145, "h": 52}}
```

If exact coordinates are unknown, omit `bbox` or provide an approximate box and
set `confidence` below `0.8`.

## Nodes

A node is any semantic object that participates in the diagram. Rectangular
blocks, operator circles, multipliers, triangular gains, delays, muxes, scopes,
ports, chips, memories, filters, ADCs, and plants are all nodes.

```json
{
  "id": "n_pid_altitude",
  "kind": "functional_block",
  "subtype": "controller.pid",
  "label": "PID",
  "text": {
    "main": "PID",
    "lines": ["PID", "For altitude"],
    "formula": null
  },
  "bbox": {"x": 226, "y": 20, "w": 145, "h": 52},
  "ports": ["p_pid_altitude_in", "p_pid_altitude_out"],
  "properties": {
    "role": "altitude controller"
  },
  "confidence": 0.93,
  "evidence": "rectangle labeled PID / For altitude"
}
```

### Node Kinds

Use these canonical `kind` values:

- `functional_block`: named processing block, controller, ADC, DAC, filter,
  decoder, plant, memory, interface, converter, detector, estimator.
- `operator`: mathematical or signal-flow operator such as sum, difference,
  product, integrator, derivative, delay, gain, saturation, switch, mux.
- `source_sink`: external input, output, probe, scope, display, antenna,
  reference, constant.
- `connector`: junction, split point, merge point, bus tap, terminal dot.
- `subsystem`: a visible component that is also a container; use when the
  boundary itself has ports and behavior.
- `unknown`: visible object whose type cannot be confidently classified.

Recommended `subtype` examples:

- `controller.pid`, `controller.pi`, `controller.smc`
- `filter.lpf`, `filter.hpf`, `filter.matched`
- `converter.adc`, `converter.dac`, `converter.sha`
- `math.sum`, `math.product`, `math.gain`, `math.delay`, `math.integrator`
- `signal.mux`, `signal.demux`, `signal.switch`
- `system.plant`, `system.dynamics`, `system.channel`
- `digital.register`, `digital.flip_flop`, `digital.counter`
- `rf.mixer`, `rf.vco`, `rf.pll`, `rf.phase_detector`

The subtype vocabulary is open. Prefer stable lowercase dotted names.

## Ports

Ports make edge attachment explicit. A port may be visually drawn as a pin, an
arrow endpoint, a small circle, or an implicit side of a block.

```json
{
  "id": "p_pid_altitude_out",
  "node": "n_pid_altitude",
  "name": "U1",
  "direction": "out",
  "side": "right",
  "bbox": {"x": 371, "y": 46, "w": 4, "h": 4},
  "signal": "U1",
  "confidence": 0.9
}
```

`direction` values:

- `in`
- `out`
- `inout`
- `unknown`

For source/sink labels that are not enclosed in a box, create a `source_sink`
node and a port if it has a visible connection.

## Edges

An edge represents directed or undirected connectivity. Most block diagrams use
directed signal flow.

```json
{
  "id": "e_u1_to_dynamics",
  "kind": "signal",
  "source": "p_pid_altitude_out",
  "target": "p_quad_u1_in",
  "direction": "directed",
  "label": "U1",
  "style": {
    "line": "solid",
    "arrow": "target",
    "color": "black"
  },
  "path": [
    {"x": 371, "y": 46},
    {"x": 897, "y": 46}
  ],
  "branches": [],
  "confidence": 0.92
}
```

### Edge Kinds

- `signal`: real signal or data/control flow.
- `feedback`: real signal edge that closes a feedback loop.
- `bus`: multi-bit or multi-signal connection.
- `power_or_clock`: named supply, clock, reset, or timing connection.
- `physical`: physical/mechanical relation, plant output, antenna/RF path.
- `annotation_link`: visual callout or explanatory arrow; not a signal.
- `unknown`: visible connection whose semantics are unclear.

### Edge Rules

1. A visible arrowhead determines direction when present.
2. A line without arrowhead can be `undirected` or inferred from labels and
   surrounding flow with lower confidence.
3. Branch points should be represented using either:
   - a `connector` node, if a junction dot or branch is visually important; or
   - one edge with `branches`, if the branch is simple and all destinations are
     clear.
4. Do not encode dashed explanatory loops as `signal` unless they physically
   connect functional ports.
5. Preserve signal names (`U1`, `D_in`, `f_REF`, `theta_d`) in `label` and
   preferably also in attached port `signal`.

## Operators

Operator symbols should be nodes, not edge attributes. This keeps signs and
multi-input logic explicit.

### Sum/Difference

```json
{
  "id": "n_sum_x",
  "kind": "operator",
  "subtype": "math.sum",
  "label": "+/-",
  "bbox": {"x": 78, "y": 86, "w": 31, "h": 31},
  "properties": {
    "inputs": [
      {"port": "p_sum_x_ref", "sign": "+"},
      {"port": "p_sum_x_feedback", "sign": "-"}
    ]
  },
  "confidence": 0.88
}
```

### Product/Mixer

Use `subtype: "math.product"` or `subtype: "rf.mixer"`. Preserve visible `x`,
`X`, or mixer symbols in `label`.

### Gain, Delay, Integrator, Transfer Function

Use `operator` for simple mathematical primitives:

```json
{
  "id": "n_delay_z1",
  "kind": "operator",
  "subtype": "math.delay",
  "label": "z^-1",
  "text": {"main": "z^-1", "formula": "z^-1"}
}
```

If the block is a named subsystem containing a formula, use
`kind: "functional_block"` and store the formula in `text.formula`.

## Groups and Hierarchy

Groups represent visible containment or conceptual regions: dashed boxes,
colored regions, chip boundaries, loop labels, "Position Control", "DSP
Circuit", "Analog Loop", etc.

```json
{
  "id": "g_position_control",
  "kind": "region",
  "label": "Position Control",
  "style": {
    "boundary": "dashed",
    "color": "black",
    "fill": null
  },
  "bbox": {"x": 211, "y": 3, "w": 168, "h": 178},
  "contains": {
    "nodes": ["n_pid_altitude", "n_pid_x", "n_pid_y"],
    "edges": []
  },
  "ports": [],
  "confidence": 0.92
}
```

Use `kind` values:

- `region`: visual grouping with no independent function.
- `subsystem_boundary`: boundary that also acts like a functional component.
- `chip_or_board`: IC, board, or hardware boundary.
- `loop_annotation`: visual loop region, often dashed.
- `lane_or_domain`: analog/digital/RF/mechanical/control domain separation.

Groups may be nested.

## Texts

All important text should be preserved, even if it is already attached to a
node, edge, or group. This allows OCR auditing.

```json
{
  "id": "t_position_control",
  "text": "Position Control",
  "bbox": {"x": 227, "y": 185, "w": 126, "h": 24},
  "role": "group_label",
  "attached_to": "g_position_control",
  "confidence": 0.96
}
```

Text roles:

- `node_label`
- `edge_label`
- `port_label`
- `group_label`
- `formula`
- `caption`
- `axis_or_tick`
- `note`
- `unknown`

## Annotations

Annotations are visible marks that are not primary connectivity:

- figure captions
- explanatory dashed arrows
- callout arrows
- loop direction arrows
- color highlights
- handwritten or scanned artifacts
- page crop boundaries

```json
{
  "id": "a_loop_ii",
  "kind": "loop_direction",
  "label": "Loop II",
  "style": {"line": "dashed", "arrow": "left"},
  "bbox": {"x": 317, "y": 42, "w": 285, "h": 66},
  "related_to": ["n_pd", "n_lpf", "n_vco"],
  "confidence": 0.76
}
```

## Confidence and Evidence

Every node, edge, group, and important text should include `confidence` when
generated by a model.

Recommended confidence bands:

- `0.90-1.00`: clearly visible and unambiguous.
- `0.70-0.89`: visible but small, low resolution, or some endpoint ambiguity.
- `0.40-0.69`: likely but uncertain.
- `<0.40`: weak guess; include in `warnings` unless needed for completeness.

Use `evidence` for short natural-language justification.

## Minimal Valid Output

A minimal valid output may omit coordinates and OCR trace, but must include
nodes, ports, edges, and groups:

```json
{
  "schema": "eda_block_diagram.v0.1",
  "source": {"image_path": "example.jpg"},
  "diagram": {"title": null, "domain": [], "view_type": "block_diagram"},
  "nodes": [
    {"id": "n1", "kind": "source_sink", "label": "input"},
    {"id": "n2", "kind": "functional_block", "subtype": "filter.lpf", "label": "LPF"},
    {"id": "n3", "kind": "source_sink", "label": "output"}
  ],
  "ports": [
    {"id": "p1", "node": "n1", "direction": "out"},
    {"id": "p2", "node": "n2", "direction": "in"},
    {"id": "p3", "node": "n2", "direction": "out"},
    {"id": "p4", "node": "n3", "direction": "in"}
  ],
  "edges": [
    {"id": "e1", "kind": "signal", "source": "p1", "target": "p2", "direction": "directed"},
    {"id": "e2", "kind": "signal", "source": "p3", "target": "p4", "direction": "directed"}
  ],
  "groups": []
}
```

## Example: Hidden EDA_TESTs_015

This is a partial representation, not a full annotation. It shows the intended
level of semantic detail.

```json
{
  "schema": "eda_block_diagram.v0.1",
  "source": {
    "image_path": "eda-problem-6/datasets/Hidden/images/EDA_TESTs_015.jpg",
    "image_width": 1116,
    "image_height": 372
  },
  "diagram": {
    "title": null,
    "domain": ["control_system", "robotics", "quadcopter"],
    "view_type": "block_diagram"
  },
  "nodes": [
    {
      "id": "n_zd",
      "kind": "source_sink",
      "label": "z_d",
      "bbox": {"x": 3, "y": 13, "w": 43, "h": 54},
      "ports": ["p_zd_out"],
      "confidence": 0.93
    },
    {
      "id": "n_sum_z",
      "kind": "operator",
      "subtype": "math.sum",
      "label": "+/-",
      "bbox": {"x": 78, "y": 20, "w": 31, "h": 31},
      "ports": ["p_sum_z_ref", "p_sum_z_feedback", "p_sum_z_out"],
      "properties": {
        "inputs": [
          {"port": "p_sum_z_ref", "sign": "+"},
          {"port": "p_sum_z_feedback", "sign": "-"}
        ]
      },
      "confidence": 0.9
    },
    {
      "id": "n_pid_altitude",
      "kind": "functional_block",
      "subtype": "controller.pid",
      "label": "PID",
      "text": {"main": "PID", "lines": ["PID", "For altitude"]},
      "bbox": {"x": 226, "y": 20, "w": 145, "h": 52},
      "ports": ["p_pid_alt_in", "p_pid_alt_out"],
      "properties": {"role": "altitude controller"},
      "confidence": 0.94
    },
    {
      "id": "n_conversion",
      "kind": "functional_block",
      "subtype": "control.conversion",
      "label": "Conversion",
      "text": {
        "main": "Conversion",
        "lines": ["Conversion", "Convert control", "signals T to", "attitude set-points"]
      },
      "bbox": {"x": 401, "y": 98, "w": 160, "h": 92},
      "confidence": 0.92
    },
    {
      "id": "n_pid_mfnn_phi",
      "kind": "functional_block",
      "subtype": "controller.pid_mfnn",
      "label": "PID - MFNN",
      "text": {"main": "PID - MFNN", "lines": ["PID - MFNN", "For phi"]},
      "bbox": {"x": 721, "y": 91, "w": 132, "h": 48},
      "confidence": 0.91
    },
    {
      "id": "n_quadcopter",
      "kind": "functional_block",
      "subtype": "system.dynamics",
      "label": "Quadcopter Dynamics",
      "bbox": {"x": 886, "y": 13, "w": 151, "h": 213},
      "confidence": 0.95
    }
  ],
  "ports": [
    {"id": "p_zd_out", "node": "n_zd", "direction": "out", "side": "right", "signal": "z_d"},
    {"id": "p_sum_z_ref", "node": "n_sum_z", "direction": "in", "side": "left", "signal": "z_d"},
    {"id": "p_sum_z_feedback", "node": "n_sum_z", "direction": "in", "side": "bottom", "signal": "z"},
    {"id": "p_sum_z_out", "node": "n_sum_z", "direction": "out", "side": "right"},
    {"id": "p_pid_alt_in", "node": "n_pid_altitude", "direction": "in", "side": "left"},
    {"id": "p_pid_alt_out", "node": "n_pid_altitude", "direction": "out", "side": "right", "signal": "U1"}
  ],
  "edges": [
    {
      "id": "e_zd_to_sum_z",
      "kind": "signal",
      "source": "p_zd_out",
      "target": "p_sum_z_ref",
      "direction": "directed",
      "label": "z_d",
      "style": {"line": "solid", "arrow": "target"},
      "confidence": 0.91
    },
    {
      "id": "e_sum_z_to_pid_alt",
      "kind": "signal",
      "source": "p_sum_z_out",
      "target": "p_pid_alt_in",
      "direction": "directed",
      "confidence": 0.9
    },
    {
      "id": "e_pid_alt_to_quadcopter",
      "kind": "signal",
      "source": "p_pid_alt_out",
      "target": "p_quadcopter_u1",
      "direction": "directed",
      "label": "U1",
      "confidence": 0.9
    }
  ],
  "groups": [
    {
      "id": "g_position_control",
      "kind": "region",
      "label": "Position Control",
      "style": {"boundary": "dashed", "color": "black"},
      "contains": {
        "nodes": ["n_pid_altitude", "n_pid_x", "n_pid_y"],
        "edges": []
      },
      "confidence": 0.9
    },
    {
      "id": "g_attitude_control",
      "kind": "region",
      "label": "Proposed Attitude Control",
      "style": {"boundary": "dashed", "color": "blue"},
      "contains": {
        "nodes": ["n_pid_mfnn_phi", "n_pid_mfnn_theta", "n_pid_mfnn_psi"],
        "edges": []
      },
      "confidence": 0.9
    }
  ],
  "texts": [],
  "annotations": [],
  "warnings": [
    "Partial example: not all feedback edges and ports are enumerated."
  ]
}
```

## Example: PLL Loop Diagram

For diagrams like `Public/benchmark/images/EDA_TEST_015.jpg`, dashed loop arrows
are annotations unless they connect real ports.

```json
{
  "schema": "eda_block_diagram.v0.1",
  "source": {"image_path": "eda-problem-6/datasets/Public/benchmark/images/EDA_TEST_015.jpg"},
  "diagram": {"title": null, "domain": ["pll", "rf"], "view_type": "block_diagram"},
  "nodes": [
    {"id": "n_din", "kind": "source_sink", "label": "D_in"},
    {"id": "n_pd", "kind": "functional_block", "subtype": "rf.phase_detector", "label": "PD"},
    {"id": "n_lpf", "kind": "functional_block", "subtype": "filter.lpf", "label": "LPF"},
    {"id": "n_vco", "kind": "functional_block", "subtype": "rf.vco", "label": "VCO"},
    {"id": "n_pfd", "kind": "functional_block", "subtype": "rf.phase_frequency_detector", "label": "PFD"},
    {"id": "n_cp", "kind": "functional_block", "subtype": "rf.charge_pump", "label": "CP"},
    {"id": "n_div_n", "kind": "operator", "subtype": "math.divide", "label": "÷N"},
    {"id": "n_fref", "kind": "source_sink", "label": "f_REF"},
    {"id": "n_fout", "kind": "source_sink", "label": "f_out"}
  ],
  "ports": [],
  "edges": [
    {"id": "e_din_pd", "kind": "signal", "source": "n_din", "target": "n_pd", "direction": "directed"},
    {"id": "e_pd_lpf", "kind": "signal", "source": "n_pd", "target": "n_lpf", "direction": "directed"},
    {"id": "e_lpf_vco", "kind": "signal", "source": "n_lpf", "target": "n_vco", "direction": "directed"},
    {"id": "e_vco_fout", "kind": "signal", "source": "n_vco", "target": "n_fout", "direction": "directed"},
    {"id": "e_vco_div_feedback", "kind": "feedback", "source": "n_vco", "target": "n_div_n", "direction": "directed"},
    {"id": "e_div_pfd", "kind": "feedback", "source": "n_div_n", "target": "n_pfd", "direction": "directed"},
    {"id": "e_pfd_cp", "kind": "signal", "source": "n_pfd", "target": "n_cp", "direction": "directed"},
    {"id": "e_cp_pd", "kind": "signal", "source": "n_cp", "target": "n_pd", "direction": "directed"}
  ],
  "groups": [],
  "annotations": [
    {
      "id": "a_loop_ii",
      "kind": "loop_direction",
      "label": "Loop II",
      "style": {"line": "dashed", "arrow": "left"},
      "related_to": ["n_pd", "n_lpf", "n_vco"],
      "confidence": 0.78
    },
    {
      "id": "a_loop_i",
      "kind": "loop_direction",
      "label": "Loop I",
      "style": {"line": "dashed", "arrow": "left"},
      "related_to": ["n_pfd", "n_cp", "n_div_n"],
      "confidence": 0.78
    }
  ],
  "warnings": [
    "Example omits exact port and bbox details for brevity."
  ]
}
```

## VLM Output Contract

When asking a VLM to transcribe a diagram, require:

1. Return only one JSON object.
2. Use stable IDs: `n_*`, `p_*`, `e_*`, `g_*`, `t_*`, `a_*`.
3. Do not invent blocks from surrounding paper text or the filename.
4. Include all visible blocks and operator symbols.
5. Preserve all visible signal names on edges or ports.
6. Represent visible grouping boxes as `groups`.
7. Distinguish real signal edges from annotations.
8. Use `unknown` and low confidence when uncertain.
9. Add `warnings` for cropped, illegible, or ambiguous regions.

## Common Failure Modes to Guard Against

- Treating dashed loop annotations as real signal edges.
- Dropping small operator nodes such as sum, product, delay, or gain.
- Losing signs on summing junctions.
- Flattening nested groups and chip/subsystem boundaries.
- Turning signal labels into nodes when they are only edge labels.
- Ignoring multi-output branches and junction dots.
- Hallucinating an ADC/DAC/PID block from the problem context rather than the
  image.
- Omitting formulas inside transfer-function blocks.
- Confusing repeated labels on a bus with separate nodes.
- Over-normalizing text; preserve original OCR text even if a normalized subtype
  is assigned.
