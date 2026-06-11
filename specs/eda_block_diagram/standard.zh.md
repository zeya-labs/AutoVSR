# EDA 框图 JSON 标准

版本：draft-0.1

本文档定义一种 JSON 表示，用于真实 EDA/system 框图，例如 `eda-problem-6/datasets/Hidden/images/EDA_TESTs_015.jpg` 和 public benchmark 图片。这些图不是普通电路原理图：它们混合了信号流拓扑、block 语义、数学运算符、标签、层级、标注，有时还包含截图或论文图的 artifact。

目标是在 VLM 可输出的前提下，尽量保留下游推理需要的全部信息。

## 设计目标

1. 同时保留结构和语义。
   标有 `PID For altitude` 的框不只是一个矩形；它是带有文本、端口和信号流关系的功能块。

2. 区分信号连接和视觉标注。
   携带 `U1` 的实线箭头是信号边。标有 `Loop I` 的虚线箭头可能只是解释性标注，不能强行放进信号图。

3. 支持层级。
   虚线矩形、彩色区域、芯片边界、子系统、具名控制 loop 都应表示为包含 blocks 和 edges 的 groups。

4. 保留不确定性。
   如果某个符号可见但有歧义，JSON 应记录最佳猜测、证据文本和 confidence，而不是静默丢弃。

5. 坐标可选但推荐。
   Bounding boxes 对审计和修正很重要，但即使某些坐标缺失，语义图仍应有用。

## 顶层对象

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

必需顶层字段：`schema`、`source`、`diagram`、`nodes`、`ports`、`edges`、`groups`。

可选顶层字段：`texts`、`annotations`、`legend`、`warnings`。

## 坐标约定

坐标使用图片像素，原点在左上角。

```json
{"bbox": {"x": 226, "y": 20, "w": 145, "h": 52}}
```

如果精确坐标未知，可以省略 `bbox`，或给一个近似框并把 `confidence` 设为低于 `0.8`。

## Nodes

node 是图中参与语义表达的任何对象。矩形 block、运算圆圈、乘法器、三角增益、delay、mux、scope、port、chip、memory、filter、ADC、plant 都是 nodes。

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

使用以下规范 `kind` 值：

- `functional_block`：具名处理块、controller、ADC、DAC、filter、decoder、plant、memory、interface、converter、detector、estimator。
- `operator`：数学或信号流运算符，如 sum、difference、product、integrator、derivative、delay、gain、saturation、switch、mux。
- `source_sink`：外部输入、输出、probe、scope、display、antenna、reference、constant。
- `connector`：junction、split point、merge point、bus tap、terminal dot。
- `subsystem`：可见组件，同时也是容器；当边界本身有 ports 和行为时使用。
- `unknown`：可见但无法可靠分类的对象。

推荐的 `subtype` 示例：

- `controller.pid`、`controller.pi`、`controller.smc`
- `filter.lpf`、`filter.hpf`、`filter.matched`
- `converter.adc`、`converter.dac`、`converter.sha`
- `math.sum`、`math.product`、`math.gain`、`math.delay`、`math.integrator`
- `signal.mux`、`signal.demux`、`signal.switch`
- `system.plant`、`system.dynamics`、`system.channel`
- `digital.register`、`digital.flip_flop`、`digital.counter`
- `rf.mixer`、`rf.vco`、`rf.pll`、`rf.phase_detector`

`subtype` 词表是开放的。优先使用稳定的小写 dotted name。

## Ports

ports 让 edge 的连接点显式化。port 在视觉上可能是 pin、箭头端点、小圆点，也可能只是 block 某一侧的隐式连接点。

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

`direction` 取值：

- `in`
- `out`
- `inout`
- `unknown`

对于没有被框起来的 source/sink 标签，如果它有可见连接，应创建一个 `source_sink` node；必要时也创建 port。

## Edges

edge 表示有向或无向连接。大多数框图使用有向信号流。

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

- `signal`：真实信号或数据/控制流。
- `feedback`：闭合反馈环的真实信号边。
- `bus`：多 bit 或多信号连接。
- `power_or_clock`：具名 supply、clock、reset 或 timing connection。
- `physical`：物理/机械关系、plant output、antenna/RF path。
- `annotation_link`：视觉 callout 或解释箭头；不是信号。
- `unknown`：语义不清楚的可见连接。

### Edge 规则

1. 存在可见箭头时，箭头决定方向。
2. 没有箭头的线可以是 `undirected`，也可以根据标签和周围流向低置信度推断。
3. 分支点可以用两种方式表示：
   - 如果 junction dot 或 branch 在视觉上很重要，用 `connector` node；
   - 如果 branch 简单且所有 destination 清楚，用一个带 `branches` 的 edge。
4. 除非虚线解释性 loop 实际连接了功能端口，否则不要编码成 `signal`。
5. 在 `label` 中保留信号名（如 `U1`、`D_in`、`f_REF`、`theta_d`），并尽量也保留到关联 port 的 `signal` 中。

## Operators

运算符符号应作为 nodes，而不是 edge 属性。这样可以显式保留符号、正负号和多输入逻辑。

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

使用 `subtype: "math.product"` 或 `subtype: "rf.mixer"`。在 `label` 中保留可见的 `x`、`X` 或 mixer 符号。

### Gain、Delay、Integrator、Transfer Function

简单数学原语使用 `operator`：

```json
{
  "id": "n_delay_z1",
  "kind": "operator",
  "subtype": "math.delay",
  "label": "z^-1",
  "text": {"main": "z^-1", "formula": "z^-1"}
}
```

如果 block 是一个包含公式的具名子系统，使用 `kind: "functional_block"`，并把公式存入 `text.formula`。

## Groups 和层级

groups 表示可见 containment 或概念区域：虚线框、彩色区域、芯片边界、loop 标签、`Position Control`、`DSP Circuit`、`Analog Loop` 等。

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

使用这些 `kind` 值：

- `region`：没有独立功能的视觉分组。
- `subsystem_boundary`：边界本身也像功能组件一样起作用。
- `chip_or_board`：IC、板卡或硬件边界。
- `loop_annotation`：视觉 loop 区域，通常是虚线。
- `lane_or_domain`：analog/digital/RF/mechanical/control domain 分隔。

groups 可以嵌套。

## Texts

所有重要文本都应保留，即使它已经附着在 node、edge 或 group 上。这样可以进行 OCR 审计。

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

Text roles：

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

annotations 是可见但不是主要 connectivity 的标记：

- figure captions
- 解释性虚线箭头
- callout arrows
- loop direction arrows
- color highlights
- 手写或扫描 artifact
- 页面裁剪边界

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

## Confidence 和 Evidence

由模型生成时，每个 node、edge、group 和重要 text 都应包含 `confidence`。

推荐 confidence 区间：

- `0.90-1.00`：清晰可见且无歧义。
- `0.70-0.89`：可见，但尺寸小、分辨率低或端点有一定歧义。
- `0.40-0.69`：可能正确但不确定。
- `<0.40`：弱猜测；除非为了完整性必须保留，否则应写入 `warnings`。

`evidence` 用于保存简短的自然语言理由。

## 最小合法输出

最小合法输出可以省略坐标和 OCR trace，但必须包含 nodes、ports、edges 和 groups：

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

## 示例：Hidden EDA_TESTs_015

这是一个局部表示，不是完整标注。它展示预期的语义细节粒度。

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

## 示例：PLL Loop Diagram

对于 `Public/benchmark/images/EDA_TEST_015.jpg` 这类图，虚线 loop 箭头是 annotations，除非它们实际连接了真实 ports。

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

## VLM 输出契约

要求 VLM 转写框图时，应明确要求：

1. 只返回一个 JSON object。
2. 使用稳定 ID：`n_*`、`p_*`、`e_*`、`g_*`、`t_*`、`a_*`。
3. 不要根据论文上下文或文件名臆造 block。
4. 包含所有可见 block 和 operator symbols。
5. 在 edges 或 ports 上保留所有可见 signal names。
6. 把可见 grouping boxes 表示为 `groups`。
7. 区分真实 signal edges 和 annotations。
8. 不确定时使用 `unknown` 和低 confidence。
9. 对被裁剪、难以辨认或有歧义的区域添加 `warnings`。

## 常见失败模式

- 把虚线 loop annotation 当作真实 signal edge。
- 漏掉小的 operator nodes，例如 sum、product、delay 或 gain。
- 丢失 summing junction 上的正负号。
- 把 nested groups 和 chip/subsystem boundaries 扁平化。
- 把 signal labels 错当成 nodes，而它们其实只是 edge labels。
- 忽略多输出 branch 和 junction dots。
