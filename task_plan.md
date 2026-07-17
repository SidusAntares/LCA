# LCA UCIHAR Table 5 reproduction plan

## Objective

Reproduce the ten UCIHAR Table 5 tasks with three seeds and 40 epochs, using clean checkpoint macro-F1 to decide whether LCA is suitable as a research baseline.

## Constraints

- No internet, downloads, data changes, hyperparameter tuning, target-label checkpoint selection, No_prior, or new methods.
- Preserve public algorithm semantics except the required `z_std -> z` correction and compatibility fixes.
- Do not start training during the protocol-revision turn.

## Completed protocol work

1. Freeze and audit the corrected public implementation.
2. Separate `paper_code_protocol` from `baseline_clean_protocol`.
3. Record protocol/audit metadata in results and checkpoints.
4. Evaluate best/last checkpoints with fresh stateless arrays and SHA256 hashes.
5. Rerun formal 18→14 seeds 0/1/2; isolate legacy run 0.
6. Correct diagnostic three-task aggregation and stop/warning rules.
7. Prevent diagnostics from automatically starting the full matrix.
8. Add 20→9 seed-0 dual-protocol comparison and update the report.
9. Verify unit tests, Python compilation, runtime audit, and diff whitespace.

## Server sequence

1. Run the dual-protocol comparison only.
2. Review hashes, best epoch, and clean best/last F1.
3. Choose a protocol and run the three-task diagnostic.
4. Review diagnostic files manually.
5. Start the full matrix manually only after approval.

