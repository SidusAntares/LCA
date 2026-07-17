# LCA TSClassif/HAR 离线运行审计

## 审计范围与硬性边界

目标是在无法联网的四卡服务器上复用既有 conda 环境 `ti`，运行完整 LCA（不得传 `--No_prior`）。禁止安装/升级包、下载数据、修改 HAR 数据、使用目标域标签选模，以及在当前阶段引入 DDP/DataParallel。

当前代码审计发生在 Windows 工作区；本机 conda 不含 `ti`，只有一张 RTX 4060 Laptop GPU，且当前 Python 3.13 环境缺少 torch/numpy/pandas/sklearn/einops，工作区也没有 `TSClassif/dataset/HAR`。因此本机不能代替目标服务器报告环境、数据、单步、1 epoch 或 40 epoch 的 PASS。本次只报告实际获得的结果，所有服务器数值结果均保持“待运行”。

## Git 与历史证据

- 审计基线：`main`，`45c091f Update Install.md`。
- `git log --oneline --all -- TSClassif/algorithms/algorithms.py` 只显示首提交 `b456aed first commit`。
- `git blame`/首提交内容证明 `base_dist_mean` 原本即通过 `register_buffer` 注册；当前 checkout 也已注册，无需修改。
- 首提交和当前代码都先计算采样变量 `z`，却从 `get_features` 返回 `(z_mean, z_std, z_std)`。第三项随后被当作 `z` 送入 `q_dist.log_prob(z)`、`p_dist.log_prob(z)` 与 `transition_prior_fix.forward(z)`，因此这是有调用链和历史证据支持的明确笔误，已改为 `(z_mean, z_std, z)`。

## 环境兼容修改

1. 新增 `compat.load_torch_file`：检查 `torch.load` 函数签名，仅在支持时显式传 `weights_only=False`，同时保留旧版 PyTorch 兼容性。
2. 移除 torchvision 标准化依赖，使用纯 PyTorch 通道标准化 `(x - mean) / std.clamp(min=eps)`；输入输出保持 `[N,C,L]`。
3. 移除 torchmetrics 依赖，使用 sklearn 的无状态 `accuracy_score`、`f1_score`、`roc_auc_score`。AUROC 遇到缺失类别或 sklearn 拒绝输入时发出 warning 并返回 NaN，不跨 evaluate 调用累计状态。
4. wandb 改为可选导入；本地训练流程不初始化、不调用在线 wandb。四卡脚本显式设置 `WANDB_MODE=disabled`。
5. 新增环境检查、HAR 数据检查、checkpoint 完整性检查脚本。
6. `.gitignore` 增加 Python bytecode/cache 规则，避免在服务器审计和测试时继续产生未跟踪运行产物；仓库历史中已跟踪的旧 pyc 本轮不删除。

## 官方代码明确错误修复

1. `scripts/HAR.sh`：`--num_run 3` 改为 `--num_runs 3`，并显式加入 `--phase train --data_path dataset --device cuda:0`。
2. `get_features`：第三返回值由 `z_std` 改为真正的采样 latent `z`；修复前完整 prior 实际对 log-variance 张量计算概率和转移先验，修复后与重参数采样、重建和分类所用 latent 一致。
3. 训练 update 异常不再被吞掉：写入完整 traceback 到 `failed_runs.jsonl`，打印 stderr，并重新抛出；失败路径不会保存无效 checkpoint 或继续引用未赋值模型。
4. 小于官方 checkpoint 周期的调试运行没有 `best_model` 时，以最后一个纯源域训练状态作为唯一候选，保证 1 epoch checkpoint 的 `last`/`best` 均有效；没有使用目标指标选模。
5. PyTorch 2.3.1 在 Trainer 尚未选择 CUDA context 时直接执行 `reset_peak_memory_stats(torch.device("cuda:0"))` 会报 `Invalid device argument`。现先调用 `torch.cuda.set_device(self.device)`，再对当前设备无参数重置峰值计数；只修复显存审计初始化顺序，不改变训练流程。

## 实验语义与 CLI 修改

- 新增 `--scenario SRC,TGT`、`--num_epochs N`、`--run_ids 0,1,2`；未指定时分别使用官方全部 scenarios、40 epoch、`range(num_runs)`。
- 新增 `--eval_target_during_train`，默认 False。默认训练阶段不加载目标测试集，也不做 target 指标评估；训练和 checkpoint 选择结束后才统一加载并评价。显式开启时复现官方逐 epoch target F1 日志，并在回调后恢复 train mode。
- best checkpoint 使用单独计算的纯源域 `Src_selection_loss`；伪标签损失只影响训练总损失，不再污染选模指标，target 评估回调也不参与 checkpoint 条件。
- 保留官方 classifier 中 Softmax + CrossEntropyLoss。CrossEntropyLoss 通常期望未归一化 logits，这是待确认的算法/复现语义，本轮按要求不改。
- target 训练文件仍按官方格式包含 labels，但训练循环只消费 target samples 并丢弃 label；默认流程不会在训练期间加载 `test_<target>.pt`。
- `results.csv` 增加 status、运行时间、峰值 GPU 显存、checkpoint、git/Python/PyTorch/CUDA 元数据。

## 新增工具

- `tools/check_environment.py`：打印 Python、PyTorch/CUDA、GPU、依赖导入、torch.func/vmap/jacfwd、规避说明及 PASS/FAIL。
- `tools/check_har_dataset.py`：检查官方 HAR scenarios 的全部域与 train/test 文件、结构 `[N,9,128]`、labels、有限值、batch 下限和类别分布。
- `tools/smoke_test_lca.py`：使用完整 prior 做 source/target forward、五项 loss、backward、optimizer step、关键梯度和峰值显存检查。
- `tools/check_checkpoint.py`：只有 checkpoint 的 last/best 有效且 results.csv 中对应 run 为 success 时才判定完整。
- `tools/merge_har_results.py`：递归合并独立任务结果和 `failed_runs.jsonl` 到 `LCA_all_result/HAR/results_merged.csv`，显式标记 failed/missing/missing_checkpoint，保留要求字段并生成逐场景及全局 mean/std；只要预期任务不完整便返回非零。
- `scripts/run_HAR_4gpu.sh`：十个官方场景按四个 GPU 队列分配；每卡一个独立队列、每任务独立 Python 进程/exp/log/PID；任一失败最终非零；完整任务可跳过；默认仅 run 0。
- 仓库根目录 `sync_to_server.sh`：将代码归档上传并精确覆盖到 `user@10.150.10.38:/data/user/LCA`，排除 `.git`、缓存、`TSClassif/dataset`、结果、日志和 checkpoint；不删除远端项目目录，并在解压后检查三个审计工具确实存在，从而避免 `scp -r LCA ...:/data/user/LCA` 造成的 `/data/user/LCA/LCA` 嵌套问题。

## 算法级疑点（本轮未修改）

- classifier 输出经过 Softmax 后再传给 CrossEntropyLoss；按首次复现要求保留。
- 源域 best 模型仍沿用官方每 10 epoch 检查节奏 `(epoch + 1) % 10 == 0`；本轮只为不足 10 epoch 的调试运行增加无 target 信息的 fallback，没有重写官方选择频率。
- 重建损失额外除以 batch 数、软分位阈值等算法语义没有足够证据，本轮未修改。

## 本机实际验证结果

- 标准库合同测试：RED 阶段 8 failures + 1 error；完成实现和代码审查修复后 15/15 PASS。
- `python -m compileall -q TSClassif`：PASS；生成的 Python 3.13 cache 已清理。
- `tools/check_environment.py`：按设计 FAIL；本机缺少 torch/numpy/pandas/sklearn/einops，torch.func API 无法检查。torchmetrics/torchvision/wandb 也缺失，但已有项目内规避。
- Shell 语法：系统 `bash` 指向未安装 WSL；改用 `D:\Git\usr\bin\bash.exe -n` 后，`HAR.sh` 与 `run_HAR_4gpu.sh` 均 PASS。仍建议在目标 Linux 服务器再次执行同一 `bash -n` 门禁。
- HAR 数据检查、完整 prior 单步、18→14 1 epoch、40 epoch：本机无数据/训练环境，未运行，严禁标成 PASS。

## 服务器执行门禁与命令

在仓库根目录激活 `ti` 后：

```bash
cd TSClassif
python tools/check_environment.py
python tools/check_har_dataset.py --data-dir dataset/HAR --batch-size 32
python tools/smoke_test_lca.py --data-path dataset --source 18 --target 14 --device cuda:0
```

三项均 PASS 后才运行 1 epoch：

```bash
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0
python run.py --phase train --save_dir LCA_smoke_result \
  --exp_name HAR_18_to_14_smoke --da_method LCA --dataset HAR \
  --data_path dataset --scenario 18,14 --num_runs 1 --run_ids 0 \
  --num_epochs 1 --type type1 --lr 0.001 --device cuda:0
```

1 epoch 工件验证通过后才运行 40 epoch；单任务单种子通过后才运行四卡单种子：

```bash
python run.py --phase train --save_dir LCA_single_result \
  --exp_name HAR_18_to_14_type1 --da_method LCA --dataset HAR \
  --data_path dataset --scenario 18,14 --num_runs 1 --run_ids 0 \
  --num_epochs 40 --type type1 --lr 0.001 --device cuda:0

RUN_IDS=0 NUM_EPOCHS=40 bash scripts/run_HAR_4gpu.sh
python tools/merge_har_results.py \
  --input-root LCA_all_result --output LCA_all_result/HAR/results_merged.csv
```

单种子全部成功后才允许：

```bash
RUN_IDS=0,1,2 NUM_RUNS=3 NUM_EPOCHS=40 bash scripts/run_HAR_4gpu.sh
```
