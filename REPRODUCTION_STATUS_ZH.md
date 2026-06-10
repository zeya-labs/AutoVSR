# AutoVSR 当前可用状态

当前仓库已经可以跑 CircuitSense benchmark。推荐入口是：

```bash
python scripts/run_eval.py
```

现在支持：

- 样本级并发评测
- 单个 run 目录保存结果
- 断点续跑
- 失败样本重跑
- 基础设施错误自动重试
- 卡住样本超时重启
- 自动生成 `summary.md` 和 `paper_comparison.md`

旧的 shard runner 不再推荐使用。

## 怎么跑

先跑小规模测试：

```bash
python -m compileall scripts/run_eval.py main.py
python scripts/run_eval.py --start 0 --end 20 --jobs 2 --tag quick --stall-timeout 600
```

全量 5020 条：

```bash
python scripts/run_eval.py \
  --start 0 --end 5020 \
  --jobs 4 \
  --tag full \
  --min-interval 0 \
  --retry-attempts 8 \
  --retry-base-sleep 30 \
  --retry-max-sleep 240 \
  --stall-timeout 600 \
  --progress-interval 30
```

如果模型/API 稳定，可以尝试 `--jobs 6`；如果出现 429 或并发限制，降回 `--jobs 4`。

## 结果在哪

每次运行会生成独立目录：

```text
output/eval/YYYY-MM-DD/HHMMSS_tag/
```

主要文件：

```text
results.json
results.checkpoint.json
summary.md
paper_comparison.md
eval.log
logs/
samples/
```

## 怎么续跑

从已有 run 目录继续：

```bash
python scripts/run_eval.py \
  --run-dir output/eval/<date>/<run_name> \
  --start 0 --end 5020 \
  --jobs 4
```

只重跑失败样本：

```bash
python scripts/run_eval.py --run-dir <run_dir> --rerun-failed
```

只重新生成汇总：

```bash
python scripts/run_eval.py --run-dir <run_dir> --summarize
```

## 当前结论

项目现在已经可以正式跑 CircuitSense 自动评测。后续重点是选择稳定的模型/API，跑完整 5020 条，然后查看 `summary.md` 和 `paper_comparison.md`。
