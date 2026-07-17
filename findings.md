# Findings

- Local workspace has no HAR dataset; local smoke must be synthetic and real 18→14 smoke remains a server action.
- Upstream TorchMetrics 1.3.2 `forward` returns the current-call full-test metric while persistent state is updated; only explicit `compute()` is accumulated.
- Existing concatenated-history F1 must not populate the official forward field.
- Current code already has one LCA and Trainer; protocol differences can be centralized without model duplication.
- Current branch starts at committed reproduction work `a3cd02c`; upstream public reference is `45c091f`.
- Local default Python has a broken NumPy DLL; verification uses
  `D:\Miniconda\envs\book\python.exe` (PyTorch 2.6.0, NumPy 1.26.4,
  sklearn 1.6.1). The smoke provides a narrowly scoped offline `einops`
  fallback only when the dependency is absent.
- The upstream threshold parameter receives zero gradient in the synthetic
  full-loss step because the shared additive threshold cancels when the source
  and target straight-through masks are subtracted. This is reported as a
  non-gating audit and the frozen loss formula is not changed.
