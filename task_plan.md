# LCA baseline protocol implementation

## Goal

Implement one LCA/Trainer with centralized `corrected_public` and `clean_baseline` policies, then verify locally without starting diagnostic or full training.

## Phases

1. [completed] Freeze `main@a3cd02c` and create `baseline/lca-corrected-clean`.
2. [completed] Approve and write design, provenance, and TDD implementation plan.
3. [completed] Add failing protocol/metric/classifier tests.
4. [completed] Implement policy, metrics, classifier, training/checkpoint semantics, and metadata.
5. [completed] Implement synthetic and real-HAR single-batch smoke tooling.
6. [completed] Run full local verification and diff review.

## Hard constraints

- No duplicate LCA or Trainer.
- No target-test selection/tuning.
- No architecture, loss formula, data, budget, or hyperparameter changes outside declared policy corrections.
- No diagnostic/full matrix, no commit.

## Errors

| Error | Attempt | Resolution |
|---|---:|---|
| Sandbox could not create nested branch ref | 1 | Created the approved branch with scoped Git escalation |
| Synthetic smoke found zero `threa` gradient | 1 | Kept the frozen formula; separated it as an explicit non-gating audit field |
