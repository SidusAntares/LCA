# LCA Corrected-Public and Clean Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement isolated corrected-public and clean-baseline UCIHAR protocols around one LCA/Trainer implementation.

**Architecture:** An immutable `ProtocolPolicy` centralizes every allowed behavioral difference. A TorchMetrics-1.3.2-compatible metric adapter separates current-call forward F1 from accumulated compute F1. Checkpoint metadata and protocol-qualified paths prevent cross-protocol reuse.

**Tech Stack:** Python 3.9, PyTorch, sklearn, unittest, Bash launchers.

---

### Task 1: Protocol policy and CLI

**Files:** create `TSClassif/protocol_policy.py`; modify `TSClassif/run.py`; test `TSClassif/tests/test_baseline_protocols.py`.

- [x] Write failing tests for the two canonical policies, invalid names, classifier output, target reads, meter behavior, checkpoint selector, and primary checkpoint.
- [x] Run the focused tests and confirm failures are caused by missing policy API.
- [x] Implement immutable policies and replace the two old CLI flags with required `--protocol`.
- [x] Re-run focused and existing tests.

### Task 2: TorchMetrics 1.3.2 forward compatibility

**Files:** create `TSClassif/metric_protocol.py`; test `TSClassif/tests/test_baseline_protocols.py`.

- [x] Add A/B/C fixtures where first-call, second-call, and accumulated macro-F1 differ.
- [x] Verify missing compatibility API fails.
- [x] Implement `forward` current-call return, persistent state update, `compute` accumulated return, native-1.3.2 factory, and version/backend metadata.
- [x] Verify official-forward and stateless-clean semantics.

### Task 3: Protocol-aware classifier and loss accounting

**Files:** modify `TSClassif/models/models.py`, `TSClassif/algorithms/algorithms.py`; test `TSClassif/tests/test_baseline_protocols.py`.

- [x] Add failing tests for CP probabilities, Clean logits, CE inputs, pseudo softmax, and unchanged LCA architecture.
- [x] Implement one classifier with policy-selected output and one LCA with separate source-supervised/pseudo loss accounting.
- [x] Preserve all mathematical losses, prior, weights, and CP behavior.
- [x] Run focused tests.

### Task 4: Training modes, meters, and checkpoint rules

**Files:** modify `TSClassif/algorithms/algorithms.py`, `TSClassif/trainers/abstract_trainer.py`, `TSClassif/trainers/train.py`.

- [x] Add failing tests for Clean train-mode start/restore, CP upstream mode behavior, meter reset, and source-only best selection.
- [x] Implement policy-driven epoch setup, mode traces, and `last`/`best_source` state capture.
- [x] Count target-test reads and assert Clean remains zero during training.
- [x] Run focused tests.

### Task 5: Metadata, checkpoint validation, and clean evaluation

**Files:** create `TSClassif/checkpoint_metadata.py`; modify `TSClassif/reproduction/UCIHAR/evaluate_checkpoints.py`, trainer files, and tests.

- [x] Add failing tests for required metadata, protocol mismatch, fingerprint mismatch, prior/output mismatch, and cross-protocol load rejection.
- [x] Implement metadata builder/validator and protocol-qualified output paths.
- [x] Rename fields to `official_forward_f1`, `official_accumulated_compute_f1_audit`, `last_clean_macro_f1`, and `best_source_clean_macro_f1`.
- [x] Preserve one-pass local-array sklearn evaluation.

### Task 6: Synthetic and real-HAR smoke tooling

**Files:** create `TSClassif/tools/smoke_baseline_protocol.py`; modify tests.

- [x] Add a failing smoke contract test covering every finite loss, gradient group, optimizer step, prior enabled, and classifier semantics.
- [x] Implement synthetic CPU/CUDA smoke for both protocols with JSON output.
- [x] Add `--data-path`, `--source`, `--target`, and `--real-har` for server 18鈫?4 single-batch JSON outputs.
- [x] Run local synthetic smoke only.

### Task 7: Isolated launchers, summaries, and provenance

**Files:** modify reproduction launchers/summarizers/report; create `TSClassif/baseline_audit/CODE_PROVENANCE.md` and diagnostic comparison tools.

- [ ] Add failing launcher/output-isolation and CSV-schema tests in the later diagnostic stage.
- [ ] Update diagnostic/full scripts only after the user reviews real-HAR smoke.
- [x] Document CP/Clean differences and current/upstream provenance.
- [x] Do not run diagnostic or full matrix.

### Task 8: Verification

- [x] Run all unit tests.
- [x] Run both local synthetic smokes and inspect JSON.
- [x] Run compileall and protocol audit.
- [x] Run `git diff --check` and inspect complete diff/status.
- [x] Report exact server real-HAR smoke commands and expected run count; do not commit.

