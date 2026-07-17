# LCA Baseline Code Provenance

- Upstream public baseline: `45c091fca909ac13675c6ddac7e0464f0a186355`
- Starting repository commit: `a3cd02c73c646657b496281b8eb89df0d7bce60f` (`复现测试`)
- Working branch: `baseline/lca-corrected-clean`
- Current implementation: uncommitted until diagnostic results are reviewed

## Upstream code

The upstream baseline defines the LCA encoder/decoder, transition prior, Jacobian losses, CNN, UCIHAR scenarios, normalization, 40-epoch/32-batch budget, Adam optimizer, pseudo-label settings, and public checkpoint/evaluation behavior.

## Corrected-public release fixes

- `--num_run` corrected to `--num_runs`.
- `base_dist_mean` and `base_dist_var` are registered buffers.
- `get_features` returns sampled `z` as its third latent value.
- trusted torch/checkpoint loading is compatible with the available PyTorch runtime.
- full prior remains enabled.

## Clean-baseline training-semantic fixes

- classifier returns logits and CrossEntropyLoss consumes logits;
- pseudo-label probabilities are explicit softmax probabilities;
- every epoch starts in train mode and evaluation restores prior mode;
- target-test labels are not read during training;
- epoch meters reset each epoch;
- `last` is the primary checkpoint and `best_source` uses current-epoch source-supervised loss only.

## Engineering compatibility changes

- explicit scenario/run selection, resumable isolated outputs, failure records, CUDA memory accounting, optional online logging, deterministic metadata, and offline smoke/check tools.

## Evaluation changes

- TorchMetrics-1.3.2-compatible current-call forward F1 is separated from accumulated `compute()` audit;
- best/last clean checkpoint metrics use independent one-pass sklearn arrays;
- metadata/protocol mismatch fails before evaluation.

## Unchanged audited behavior

The encoder, decoder, latent dimension, transition prior, Jacobian computation,
sparsity/structure formulas, CNN, optimizer, data splits, normalization, epoch
budget, batch size, learning rate, loss weights, pseudo threshold, and pseudo
start epoch are shared by both protocols. The synthetic smoke observes a zero
gradient for the published scalar `threa`; this is recorded but not repaired in
either protocol because changing it would alter the frozen structure-loss formula.
