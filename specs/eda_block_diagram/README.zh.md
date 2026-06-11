# EDA 框图 JSON 规范

这个目录保存用于识别真实 EDA/system 框图的工作契约，目标数据集是 `eda-problem-6/datasets`。

文件：

- `standard.md`：英文版、人类可读的表示标准。
- `standard.zh.md`：中文版、人类可读的表示标准。
- `survey_notes.md`：英文版数据集调研笔记和观察到的图类别。
- `survey_notes.zh.md`：中文版数据集调研笔记。
- `schema.json`：机器可读的 Draft 2020-12 JSON Schema。
- `validate_block_diagram_json.py`：零依赖结构校验器，用于检查 VLM 输出。
- `examples/hidden_eda_tests_015.partial.json`：`Hidden/images/EDA_TESTs_015.jpg` 的局部标注样例。

校验一个输出：

```bash
python specs/eda_block_diagram/validate_block_diagram_json.py \
  specs/eda_block_diagram/examples/hidden_eda_tests_015.partial.json
```

期望结果：

```text
specs/eda_block_diagram/examples/hidden_eda_tests_015.partial.json: OK
```

校验器会检查 VLM 输出中最容易出错的结构不变量：

- 必需的顶层字段
- 全局唯一 ID
- node 到 port 的引用
- edge 的 source/target 引用
- group containment 引用
- text 和 annotation 引用
- 基本 bbox 和 confidence 格式

它有意不强制要求完整坐标或穷尽所有 annotation，因为早期 VLM pass 即使还没有稳定的精确 layout，也可能已经能产出有用的语义图。

对单张真实图片运行 VLM 识别：

```bash
python scripts/recognize_eda_block_diagram.py \
  --image eda-problem-6/datasets/Hidden/images/EDA_TESTs_015.jpg \
  --output-dir output/eda_block_diagram_probe_hidden015 \
  --mode graph \
  --min-nodes 8 \
  --min-edges 8
```

每次运行都会创建一个自包含输出目录：

- `<case_id>.json`：按本标准归一化后的框图 JSON。
- `<case_id>.graph.svg`：根据 JSON 用 Graphviz 重建的语义拓扑图，方便和原始图片并排审计。
- `<case_id>.graph.dot`：生成 graph SVG 使用的 DOT 源文件。
- `<case_id>.raw.txt`：VLM 原始响应；启用 `--audit` 时也包含 audit 响应。
- `<case_id>.meta.json`：单样本运行元数据、validation errors 和计数。
- `summary.json`：整次运行的汇总。
- `index.html`：可视化浏览报告，包含原图、校验状态、nodes、edges、groups、warnings，以及 raw artifact 链接。
  报告中还包含可拖动的重建图编辑器；可以拖动节点修正布局，并用 `Export edited SVG` 导出修改后的 SVG。

对密集图或输出 token 预算较小的模型，优先用 `--mode graph`：它输出 nodes/groups 和直接 node-to-node 的 edges，不输出 ports 或 bboxes。需要部分 ports 和主要 bboxes 时用 `--mode compact`；只有模型输出预算充足时再用 `--mode full`。

运行第二次视觉 audit/refinement pass：

```bash
python scripts/recognize_eda_block_diagram.py \
  --image eda-problem-6/datasets/Hidden/images/EDA_TESTs_015.jpg \
  --output-dir output/eda_block_diagram_probe_hidden015_graph_audit \
  --mode graph \
  --min-nodes 8 \
  --min-edges 8 \
  --audit
```

已 smoke-test 的命令：

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
  --workers 3 \
  --output-dir output/eda_block_diagram_hidden_first3_graph \
  --mode graph \
  --min-nodes 3 \
  --min-edges 2
```

并发跑更大的 batch：

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

打开 `output/eda_block_diagram_hidden_graph/index.html` 查看结果。HTML 报告是审计工具；`valid` 只表示 JSON 结构合法，不表示每一条语义边都正确。

如果要把拖动后的图保存回磁盘，需要通过本地写盘服务打开报告：

```bash
python scripts/serve_eda_block_report.py \
  --output-dir output/eda_block_diagram_hidden_graph \
  --port 8765
```

然后打开 `http://127.0.0.1:8765/`。在重建图里拖动节点后点 `Save`，服务会把
`<case_id>.graph.layout.json` 和 `<case_id>.graph.edited.svg` 写回该输出目录。
编辑器支持选择节点/边、拖动节点、添加节点、连接两个节点、删除选中的节点或边、修改选中项的文字和类型，然后保存。

已观察到的 caveat：在当前 `qwen3-vl-plus` 4096-token 输出预算下，graph mode 的结构可靠性较好，但在密集反馈图上仍可能出现语义端点错误。例如，它可能恢复出控制图里的所有主要 block，却把某个反馈标签接错。把 graph 作为最终结果前，建议使用第二次视觉 audit/refinement pass。audit pass 已经修正了 `Hidden/images/EDA_TESTs_015.jpg` 上 z/x/y 与 phi/theta/psi 反馈拆分的问题；但 PLL 风格图里的 loop annotation 仍可能被过度解释成 feedback edge，需要人工 review。

当前 smoke-test 状态：

- Hidden first 3 graph mode：`3/3` 结构合法。
- Hidden `EDA_TESTs_015` graph+audit：结构合法，feedback split 已修正。
- Public benchmark `EDA_TEST_015` graph+audit：结构合法；当 loop annotation 标为 `Loop I`、`Loop II` 等时，会被归一化为非 signal feedback edge。
