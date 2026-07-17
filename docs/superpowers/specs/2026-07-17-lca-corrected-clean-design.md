# LCA Corrected-Public and Clean Baseline Design

## Goal

Build two fair, reproducible UCIHAR protocols around one LCA implementation:

- `corrected_public` (LCA-CP) fixes only confirmed release defects and preserves upstream training semantics.
- `clean_baseline` (LCA-Clean) adds explicit training-semantic corrections suitable for future paired comparisons.

Neither protocol may use target-test labels for checkpoint selection, hyperparameter tuning, or implementation selection.

## Protocol difference table

| Concern | `corrected_public` (LCA-CP) | `clean_baseline` (LCA-Clean) |
|---|---|---|
| Classifier output | Upstream probabilities | Logits |
| Cross-entropy input | Upstream probabilities | Logits |
| Pseudo-label confidence | Upstream softmax-on-probabilities | Softmax-on-logits |
| Target-test during training | Once per epoch | Never |
| Mode after evaluation | Upstream no-restore behavior | Restore previous mode |
| Epoch meters | Cumulative | Reset each epoch |
| Best-source selector | Upstream 10-epoch cadence, cumulative combined classification loss | Every epoch, current source-supervised loss only |
| Primary checkpoint | `best_source` | `last` |
| Formal checkpoint metric | One-pass sklearn for both states | One-pass sklearn for both states |
| Upstream metric audit | 1.3.2-compatible forward plus separate accumulated compute | Not used |
| Output namespace | `corrected_public/` | `clean_baseline/` |

## Architecture

`ProtocolPolicy` is the only authority for protocol-dependent behavior. Trainer, LCA, classifier construction, metric handling, checkpoint selection, metadata validation, output paths, and evaluation query the same immutable policy object. There is one LCA class, one Trainer, one CNN, and one classifier class.

Protocol-independent algorithm components remain unchanged: encoder, decoder, latent dimension, transition prior, Jacobian computation, sparsity/structure formulas, CNN, normalization, data splits, scenarios, optimizer, learning rate, epoch budget, batch size, loss weights, pseudo threshold, and pseudo start epoch.

## ProtocolPolicy contract

`corrected_public`:

- classifier output: probabilities produced by the upstream terminal Softmax;
- CrossEntropyLoss input: upstream probabilities;
- pseudo-label input: upstream behavior, including its explicit softmax call;
- target-test evaluation: once per epoch, persistent metric object;
- evaluation mode: preserve upstream behavior and do not restore train mode;
- epoch meters: cumulative across all epochs;
- checkpoint: upstream 10-epoch cadence and cumulative `Src_cls_loss` selector;
- primary comparison checkpoint: clean evaluation of upstream best and last are both reported;
- metric: `official_forward_f1` is the direct current-call return from TorchMetrics 1.3.2-compatible `forward`; `official_accumulated_compute_f1_audit` is a separate `compute()` audit.

`clean_baseline`:

- classifier output: logits;
- CrossEntropyLoss input: logits;
- pseudo labels/confidence: `softmax(logits, dim=1)`;
- target-test reads during training: zero;
- each epoch starts in train mode;
- any clean evaluation restores the previous module mode;
- epoch meters reset every epoch;
- checkpoint: `last` is primary; `best_source` is diagnostic and selected every epoch using only current-epoch source supervised classification loss;
- final metrics: one-pass stateless sklearn macro-F1 for last and best_source.

## Metric semantics

The compatibility metric stores an accumulated confusion matrix. `forward(preds, labels)` first updates persistent state, then returns F1 computed only from the current call. `compute()` returns F1 from accumulated state. Native TorchMetrics is used only when its version is exactly 1.3.2; otherwise the compatibility implementation is used and the actual installed version is recorded.

No concatenated-history reconstruction may populate `official_forward_f1`.

## State and metadata

Every checkpoint stores protocol, upstream/current commit, scenario, run/seed, epoch, training-mode trace, classifier/CE output types, target-test read count, checkpoint rule, z/buffer/prior audit, loss weights, pseudo settings, runtime versions, GPU, deterministic flags, metric backend/version, and protocol fingerprint.

Evaluation rejects missing metadata, protocol mismatch, classifier mismatch, primary-checkpoint mismatch, prior disabled, incorrect z/buffer audit, target-test reads in Clean, or fingerprint mismatch.

Output paths include the canonical protocol name. Cross-protocol resume or result merging is rejected.

## Checkpoints

Checkpoint payload keys are `last`, `best_source`, and `metadata`. For compatibility, LCA-CP may expose `best` as an alias only when loading legacy checkpoints; new checkpoints use canonical keys. LCA-Clean main tables always use `last_clean_macro_f1`; `best_source_clean_macro_f1` is diagnostic.

## Testing

Tests cover policy isolation, classifier contracts, CE and pseudo-label inputs, A/B/C metric semantics, train/eval behavior, epoch meter reset, checkpoint rules, target-test read count, metadata completeness/mismatch failure, deterministic metadata, stateless clean evaluation, and synthetic full-loss/backward/optimizer smoke for both protocols.

The synthetic smoke also records `threshold_gradient_nonzero_audit`. It is false
for both protocols because the published source/target straight-through masks
contain the same additive threshold term, which cancels in their difference.
This is an audit finding, not a protocol-specific correction; the published
Jacobian and structure-loss formulas remain unchanged.

No diagnostic or full UCIHAR training is run locally. The real 18→14 single-batch smoke is generated as a server command/tool.
