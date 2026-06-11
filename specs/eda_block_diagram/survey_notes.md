# EDA Block Diagram Dataset Survey Notes

Survey date: 2026-06-11

Dataset paths inspected:

- `eda-problem-6/datasets/Hidden/images`: 80 hidden test images.
- `eda-problem-6/datasets/Public/benchmark/images`: 20 public benchmark images.
- `eda-problem-6/datasets/Public/1000_images`: 978 public images from mixed paper,
  benchmark, screenshot, and synthetic sources.

Generated contact sheets:

- `output/block_schema_survey/hidden_first24.jpg`
- `output/block_schema_survey/hidden_spread24.jpg`
- `output/block_schema_survey/public_first24.jpg`
- `output/block_schema_survey/public_spread24.jpg`

## Observed Diagram Types

- Control-system block diagrams with feedback loops, plants, controllers,
  transfer functions, summing junctions, and set-point/reference signals.
- Communication and RF diagrams: PLLs, mixers, filters, ADC/DAC chains, carrier
  recovery, synthesizers, demodulators, antennas, and frequency selectors.
- Simulink-like diagrams with scopes, constants, gains, delays, integrators,
  muxes, products, and subsystem blocks.
- Digital/SoC interface diagrams with buses, registers, DMA/FIFO blocks,
  clocks, resets, status/control signals, and chip/board boundaries.
- Algorithm/dataflow diagrams with formulas, quantizers, decoders, estimators,
  memories, lookup tables, and multi-bit datapaths.
- Scanned paper figures with captions, low resolution, skew, dotted/dashed
  annotations, and non-signal explanatory arrows.

## Recurrent Visual/Semantic Elements

- Functional blocks: PID, ADC, DAC, LPF, VCO, PFD, CP, controller, plant,
  detector, decoder, estimator, memory, register, interface.
- Operators: summing junctions with visible signs, multipliers/mixers, gains,
  delays, integrators, differentiators, saturations, switches, muxes.
- Edges: directed signal arrows, feedback loops, buses, clock/reset/control
  lines, physical/RF/mechanical paths.
- Text: block labels, signal labels, port labels, formulas, captions, group
  labels, domain labels, explanatory notes.
- Hierarchy: dashed rectangles, colored regions, chip/subsystem boundaries,
  loop labels, analog/digital domain partitions.
- Ambiguity: dashed loop arrows that are annotations, repeated bus labels,
  tiny or cropped formulas, screenshot UI artifacts, and blocks whose subtype
  is domain-specific.

## Schema Implications

The representation should not be a flat list of boxes and arrows. It needs:

- `nodes` for functional objects and operators.
- `ports` for exact attachment points and signal names.
- `edges` for directed connectivity, feedback, buses, and non-signal links.
- `groups` for hierarchy and visual regions.
- `texts` for OCR preservation and audit.
- `annotations` for captions, callouts, and dashed loop arrows that are not
  actual signal edges.
- `confidence` and `warnings` for VLM uncertainty.

The proposed standard is documented in `specs/eda_block_diagram/standard.md`.
