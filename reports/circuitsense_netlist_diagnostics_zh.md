# CircuitSense Netlist Diagnostics

本文档记录当前围绕 CircuitSense synthetic 数据集做的诊断实验，用于持续保存结论、指标和后续方向。

## 目标

验证两个问题：

1. 如果已经有正确 netlist，后端 solver 是否还能成为主要瓶颈。
2. 当前 VLM 从 image 一步生成 netlist 的准确度到底如何。

## 数据范围

主要使用：

```text
data/CircuitSense/Analysis/synthetic/
```

目录规模：

```text
level0     1146
level1     2671
level2      464
level4      511
level5_bd   228
```

其中 `level5_bd` 是 block diagram / SFG 类型，没有 `q*_netlist.txt`，不适用于 netlist solver oracle 评测。

## 实验 1：Ground-Truth Netlist Solver Upper Bound

脚本：

```text
scripts/audit_gt_netlist_by_level.py
```

目的：

```text
直接使用 q*_netlist.txt，绕过 image-to-netlist 和 LLM planning，只测试 deterministic solver 能否推出 q*_ta.txt。
```

### 初始结果

使用较短超时和未对齐 level4 的 `Ad` 约定时：

```text
level0    1131/1146    98.69%
level1    2608/2671    97.64%
level2     454/464     97.84%
level4     347/511     67.91%
level5_bd    0/228      0.00%  # missing netlist
```

level4 初始较低不是因为 netlist 不足，而是求解口径未对齐。

### level4 修正

level4 全部含：

```text
Rint1
Cint1
Eint1 ... Ad
```

参考答案等价于对有限增益参数取理想极限：

```text
Ad -> oo
```

同时，`ieint1` 问题需要映射到 netlist 元件：

```text
ieint1 -> Eint1
```

修正后：

```text
level4    508/511    99.41%
```

剩余 3 个为 timeout。

### 加长超时后的结果

对旧失败样例重测：

```text
outer case timeout: 120s
SymPy compare timeout: 60s
```

结果：

```text
level0    1144/1146    99.83%
level1    2654/2671    99.36%
level2     458/464     98.71%
level4     511/511    100.00%
```

继续提高 Lcapy 内部超时：

```text
outer case timeout: 240s
SymPy compare timeout: 60s
LCAPY_TIMEOUT_SECONDS: 180s
```

只额外修复：

```text
level2 q610
```

最终当前结果：

```text
level0    1144/1146    99.83%
level1    2654/2671    99.36%
level2     459/464     98.92%
level4     511/511    100.00%
```

结果文件：

```text
output/gt_netlist_solver_by_level_fixed_retry_t120_c60_lcapy180.json
```

### 剩余失败

```text
level0:
q295 q515

level1:
q109 q438 q495 q541 q720 q1044 q1693 q2012 q2058
q2073 q2204 q2665 q2711 q2757 q2900 q3467 q3637

level2:
q174 q300 q422 q539 q544

level4:
无
```

剩余错误主要是：

```text
1. Lcapy transfer 超时，即使 LCAPY_TIMEOUT_SECONDS=180 仍未完成。
2. 少数 MNA matrix not invertible。
```

结论：

```text
对 synthetic netlist levels 而言，如果 netlist 正确且 solver 约定对齐，后端 symbolic solving 基本不是主要瓶颈。
主要瓶颈在 image-to-netlist / image-to-IR。
```

## 实验 2：VLM Image-to-Netlist 准确率

脚本：

```text
scripts/eval_vlm_netlist.py
```

目的：

```text
只运行 build_netlist_node，不运行 solver，评估从图片生成 netlist 的准确率。
```

使用模型配置：

```text
config/config.yaml
```

当前配置中：

```yaml
llm:
  provider: qwen
  model: qwen3-vl-plus
```

当前 `config/config.yaml` 和 `src/nodes/netlist/build.py` 中已经没有 `autovsr_plus_rerank` 逻辑；本实验调用的是当前 `build_netlist_node` 的 VLM 生成路径。它会做少量字符串级 source-domain 规范化（例如 transfer function 的电压源格式），但不再混入 AutoVSR++ visual parser/rerank 候选。

### 并发评测命令

该脚本可通过 `--level` 自由选择 synthetic `level0`、`level1`、`level2`、`level4`。如果不显式指定 `--output`，会自动写到：

```text
output/vlm_netlist_<level>_start<start>_limit<limit>_workers<workers>.json
```

示例：

```bash
python scripts/eval_vlm_netlist.py \
  --level level0 \
  --limit 100 \
  --workers 32 \
  --case-timeout 240
```

单独测一个 case：

```bash
python scripts/eval_vlm_netlist.py \
  --level level0 \
  --case-id q49 \
  --workers 1 \
  --print-prompt \
  --print-netlists
```

结果文件：

```text
output/vlm_netlist_level0_first100_parallel32.json
```

### 前 100 条结果

```text
total: 100
build_success: 100

component_multiset_match_ignore_nodes: 97/100
component_multiset_match_with_undirected_nodes: 57/100

avg_component_name_recall: 1.0000
avg_component_name_precision: 0.9970
avg_type_accuracy_on_common: 1.0000
avg_value_accuracy_on_common: 1.0000
avg_undirected_terminal_accuracy_on_common: 0.8398
```

### 指标解释

```text
build_success
```

成功生成可解析 netlist 的样例数。只表示格式可用，不表示网表正确。

```text
component_multiset_match_ignore_nodes
```

忽略节点连接，只看元件集合、类型和值是否一致。`97/100` 表明元件识别基本正确。

```text
component_multiset_match_with_undirected_nodes
```

比较元件集合以及每个元件连接的两个节点，忽略端点顺序。`57/100` 表明只有 57 个样例整张图连接关系完全正确。

```text
avg_undirected_terminal_accuracy_on_common
```

对同名元件逐个计算无向端点是否正确，再平均。`0.8398` 表明平均约 84% 的元件端点连接正确，但整图全对率明显更低。

### 观察

当前 VLM/build pipeline：

```text
1. 元件名、类型和值几乎全对。
2. 主要错误在节点编号、连线、端点连接。
3. image-to-netlist 的瓶颈不是“看出有哪些元件”，而是“恢复电气连接关系”。
```

较差样例包括：

```text
q68 terminal_acc=0.214
q64 terminal_acc=0.333
q66 terminal_acc=0.333
q16/q20/q21 terminal_acc=0.444
```

### 固定快速诊断集：level0 first64 / workers32

后续快速评测暂定使用：

```bash
python scripts/eval_vlm_netlist.py \
  --level level0 \
  --limit 64 \
  --workers 32 \
  --case-timeout 240
```

当前一次运行结果：

```text
total: 64
build_success: 64
component_multiset_match_ignore_nodes: 64/64
component_multiset_match_with_undirected_nodes: 64/64
avg_component_name_recall: 1.0000
avg_component_name_precision: 1.0000
avg_type_accuracy_on_common: 1.0000
avg_value_accuracy_on_common: 1.0000
avg_undirected_terminal_accuracy_on_common: 1.0000
```

注意：该结果显著好于前 100 条中的若干后段样例，后续比较方法时应固定同一 64 条集合、同一并发和同一模型配置，避免把不同样本范围或模型非确定性混在一起解释。

### level0 first256 / workers32

一次扩展到前 256 条后的结果：

```text
total: 256
build_success: 256
component_multiset_match_ignore_nodes: 245/256
component_multiset_match_with_undirected_nodes: 240/256
avg_component_name_recall: 0.9985
avg_component_name_precision: 0.9974
avg_type_accuracy_on_common: 1.0000
avg_value_accuracy_on_common: 1.0000
avg_undirected_terminal_accuracy_on_common: 0.9971
```

解释：

```text
1. first64 全对不代表 level0 前 256 条全对；样本范围扩大后覆盖到更复杂/更容易错的 case。
2. 245/256 的 ignore_nodes 表明有 11 个样例连元件集合都不完全一致，通常是漏元件、多元件或元件命名集合不一致。
3. 240/256 的 with_undirected_nodes 表明整图级 executable netlist 完全正确的样例是 240 个。
4. type/value 仍为 1.0，说明错误仍主要不是元件类型和值，而是少量元件集合和连接关系。
```

脚本曾经把 summary 最后一行误打印为 `saved: <level_dir> -> <limit> samples`，这里的 `<level_dir>` 不是输出文件路径。已修正为打印真实 `output_path`。

## 当前判断

现有证据支持：

```text
CircuitSense synthetic netlist levels 的主要困难是视觉到 netlist 的结构恢复。
一旦 executable netlist 正确，传递函数、节点电压、电流等符号结果基本可以由 deterministic solver 推出。
```

进一步工作应聚焦：

```text
1. 纯 VLM one-shot image-to-netlist 准确率。
2. 当前 rerank/heuristic 增强 pipeline 对准确率的提升。
3. 更细粒度的 netlist-level metric：
   - component detection
   - label OCR
   - terminal connection
   - node connectivity
   - polarity
   - exact executable netlist
```

## 建议下一步实验

1. 若要测更“原始”的 VLM 输出，可以在 `build_netlist_node` 中临时关闭 `_fix_netlist_sources` / `_fix_transient_source_domains` 这类后处理，另存 raw LLM response 与 parsed netlist。

2. 对比不同 prompt / 不同 VLM：

```text
component match
terminal accuracy
full undirected netlist match
final symbolic answer accuracy
```

3. 进一步拆分错误类型：

```text
component/label error
node numbering mismatch
terminal connection error
polarity error
source-domain formatting difference
```
