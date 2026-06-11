# EDA 框图数据集调研笔记

调研日期：2026-06-11

已检查的数据集路径：

- `eda-problem-6/datasets/Hidden/images`：80 张 hidden 测试图。
- `eda-problem-6/datasets/Public/benchmark/images`：20 张 public benchmark 图。
- `eda-problem-6/datasets/Public/1000_images`：978 张 public 图，来源混合了论文图、benchmark、截图和合成图。

已生成的 contact sheet：

- `output/block_schema_survey/hidden_first24.jpg`
- `output/block_schema_survey/hidden_spread24.jpg`
- `output/block_schema_survey/public_first24.jpg`
- `output/block_schema_survey/public_spread24.jpg`

## 观察到的图类型

- 控制系统框图：反馈环、被控对象、控制器、传递函数、求和节点、设定值/参考信号。
- 通信和 RF 图：PLL、混频器、滤波器、ADC/DAC 链路、载波恢复、频率合成器、解调器、天线、频率选择器。
- 类 Simulink 图：scope、常量、增益、延迟、积分器、mux、乘法器、子系统块。
- 数字/SoC 接口图：总线、寄存器、DMA/FIFO、时钟、复位、状态/控制信号、芯片/板级边界。
- 算法/数据流图：公式、量化器、解码器、估计器、存储器、查找表、多 bit 数据通路。
- 扫描论文图：caption、低分辨率、倾斜、点线/虚线标注、非信号解释箭头。

## 反复出现的视觉/语义元素

- 功能块：PID、ADC、DAC、LPF、VCO、PFD、CP、controller、plant、detector、decoder、estimator、memory、register、interface。
- 运算符：带正负号的求和节点、乘法器/混频器、增益、延迟、积分器、微分器、饱和、开关、mux。
- 边：有向信号箭头、反馈环、总线、clock/reset/control 线、物理/RF/机械路径。
- 文本：块标签、信号标签、端口标签、公式、caption、分组标签、domain 标签、解释性注释。
- 层级：虚线矩形、彩色区域、芯片/子系统边界、loop 标签、analog/digital/RF/mechanical/control domain 分区。
- 歧义：作为标注的虚线 loop 箭头、重复 bus 标签、很小或被裁剪的公式、截图 UI 干扰、领域相关且不易分类的 block subtype。

## 对 Schema 的影响

这个表示不应该只是扁平的 boxes 和 arrows 列表。它需要：

- `nodes`：表示功能对象和运算符。
- `ports`：表示精确连接点和信号名。
- `edges`：表示有向连接、反馈、总线和非信号链接。
- `groups`：表示层级和可视区域。
- `texts`：保留 OCR 文本，便于审计。
- `annotations`：表示 caption、callout、以及不是实际信号边的虚线 loop 箭头。
- `confidence` 和 `warnings`：表示 VLM 的不确定性。

拟定标准见 `specs/eda_block_diagram/standard.md`；中文版本见 `specs/eda_block_diagram/standard.zh.md`。
