# Validation result

- All 2,027 cached samples and 3,462,008 tokens passed scale application, exact token alignment, and finite-value checks.
- 3,113,637 tokens have a nonzero loss mask.
- Scan duration: 218.8 seconds; peak process RSS: 1.19 GiB (baseline peak before iteration: 0.85 GiB).
- All four one-sample, 512-token calibration runs succeeded; checkpoints contain finite positive scales for all 36 calibrated modules.
- 14 regression tests passed. Shell syntax, focused Ruff checks, Python compilation, and actual-revision dry-run checks passed.
- Compatible legacy cache reuse succeeds; Qwen3-4B reuse of the Qwen3-8B cache fails before server launch.

These checks resolve input-path usability for the existing Qwen3-8B/DFlash cache.
They do not choose experimental budgets, prove historical target weight identity,
or validate serving performance or acceptance.
