# LCA UCIHAR Table 5 复现报告

## 当前状态

代码与协议修改已经完成，本轮未启动训练。现有 18→14 run 0、40 epoch、F1=1.0000 只属于冻结前的历史 smoke 证据，不进入正式三种子结果、汇总、完成计数或复现判断。

正式 18→14 将与其他任务一样，在选定协议下重新运行 seed 0、1、2。旧 checkpoint 若存在，只会额外评价到 `legacy_18_to_14_run0.csv`。

## 两条明确协议

### paper_code_protocol

- 显式启用公开代码的逐 epoch target test 评价。
- 公开 `evaluate/calculate_metrics` 路径调用 `algorithm.eval()`。
- 公开 epoch 循环在评价结束后没有调用 `train()`。
- 因此 epoch 1 评价后，后续训练会继续处于 eval 模式；仓库模型含 Dropout 和 BatchNorm，这会改变训练语义。
- 此轨用于判断公开代码能否产生论文表格。
- 对应 `metric_protocol=official_stateful_no_reset`。

### baseline_clean_protocol

- 训练期间不读取 target test 标签。
- 每个 epoch 开始显式调用 `train()`。
- 训练结束后只对 best/last checkpoint 各做一次无状态 clean 评价。
- 此轨用于决定后续研究应采用的 baseline 版本。
- 对应 `metric_protocol=stateless_current`；checkpoint clean 评价本身标记为 `clean_checkpoint`。

公开与 clean 两轨是否产生相同参数不能靠代码阅读断言。服务器必须先只运行 20→9 seed 0 双协议比较，核对 state_dict SHA256、best epoch、best/last clean macro-F1。若哈希完全一致，完整实验只需运行 clean 协议；若不同，则逐 epoch target 评价确实造成训练语义差异，Table 5 轨道需在审查比较文件后决定。

## 指标来源约束

- `results.csv` 与 checkpoint metadata 记录 training/metric protocol、运行时审计结果、best/last epoch 和代码指纹。
- 只有 `metric_protocol=official_stateful_no_reset` 才填写 `official_reported_f1`。
- 其他协议的训练期报告值写入 `current_reported_f1`，`official_reported_f1` 留空。
- `get_features_returns_z`、`metric_reset_fixed` 和 corrected-public 指纹由运行时审计得出，不硬编码。
- 当前代码不满足 corrected public implementation 审计或 checkpoint 指纹不匹配时，正式评价直接失败。

## 诊断与完整矩阵

诊断任务固定为：20→9（0.6946）、7→18（0.9108）、9→19（0.9692）。诊断论文均值为 0.8582，所有总体差异都只比较同一组三任务。十任务论文均值 0.94983 仅用于 full 模式。

诊断停止门禁检查：三任务是否完成、clean F1 是否有限、20→9 差异是否超过 0.15、三任务平均绝对误差是否超过 0.10、best/last 是否异常分离、是否使用 target 真实标签选 checkpoint。official/clean 差异超过 0.05 只记 warning。

`run_diagnostic.sh` 完成训练、评价和门禁后退出，不会自动调用 `run_full.sh`。通过时打印：

> Diagnostic passed. Review diagnostic files, then run run_full.sh manually.

## 服务器执行顺序

在 `/data/user/LCA/TSClassif` 中首先执行：

```bash
bash reproduction/UCIHAR/run_protocol_comparison.sh
```

审查 `protocol_comparison_20_to_9_seed0.json` 后再选择协议运行诊断，例如：

```bash
TRAINING_PROTOCOL=baseline_clean_protocol bash reproduction/UCIHAR/run_diagnostic.sh
```

诊断通过且人工检查文件后，才手动运行：

```bash
TRAINING_PROTOCOL=baseline_clean_protocol bash reproduction/UCIHAR/run_full.sh
```

不要在双协议比较完成前启动诊断或完整矩阵。
