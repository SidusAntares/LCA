# Progress

## 2026-07-17 protocol revision

- Froze and audited the corrected public implementation; no training was started.
- Confirmed the public per-epoch target evaluation calls `eval()` and does not restore `train()`. The model contains Dropout and BatchNorm, so this can change subsequent epochs.
- Added explicit `paper_code_protocol` and `baseline_clean_protocol` paths and required CLI provenance.
- Added checkpoint/runtime audit metadata, metric-provenance routing, clean best/last evaluation, and state_dict SHA256 output.
- Added the 20→9 seed-0 two-protocol comparison entry point.
- Changed formal 18→14 to seeds 0/1/2; legacy run 0 is evaluated separately and excluded from all formal summaries and decisions.
- Changed diagnostics to stop after the three-task gate and require manual full-matrix launch.
- Corrected the diagnostic paper mean to 0.8582 and made official/clean gaps warnings only.
- Rewrote the static and generated reproduction reports to document the exact semantics.
- Verification: 28/28 unit tests pass; Python compileall passes; corrected-public runtime audit passes 8/8; `git diff --check` passes.
- Local Bash execution was unavailable because this Windows host has no installed WSL distribution; shell control flow is covered by contract tests and must receive `bash -n` on the server before training.

