# Findings

## Protocol audit

- Public LCA calls target-test evaluation after every epoch.
- `AbstractTrainer.evaluate()` calls `self.algorithm.eval()`.
- The public epoch loop does not restore `train()` after evaluation.
- CNN contains BatchNorm1d and Dropout, so paper-code and clean training may produce different model parameters.
- Best checkpoint selection remains based on accumulated source classification loss and does not use target-test labels.
- The required corrected-public change is `get_features(): (z_mean, z_std, z)`; prior buffers, pseudo-label defaults, loss weights, normalization, and checkpoint selector remain public semantics.

## Result protocol

- Only `official_stateful_no_reset` may populate `official_reported_f1`.
- `stateless_current` populates `current_reported_f1`; clean checkpoint values are independent one-pass metrics.
- Runtime audit and protocol fingerprint metadata are required for formal checkpoint evaluation.
- Legacy 18→14 run 0 remains historical smoke evidence only and cannot enter formal CSVs, completion counts, or reproduction judgments.

## Required server order

1. Run only the 20→9 seed-0 protocol comparison.
2. Review parameter hashes, best epoch, and best/last clean F1.
3. Select one protocol and run the three-task diagnostic.
4. Review diagnostic files; launch the full matrix manually only if acceptable.

