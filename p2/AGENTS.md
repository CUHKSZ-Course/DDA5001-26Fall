# Part II exercise completion contract

Complete only the two marked TODOs in `analysis_exercises.py`:

- `math_by_level(questions)`: group MATH-500 records by `level`, compute per-level
  question counts and `pass_at_1` fractions for all four arms (`original`, `teacher`,
  `sft`, `opd`), and count `sft_wins`, `opd_wins`, and `ties` (more correct draws on
  the same question wins).
- `ifeval_transitions(questions)`: compare `sft` and `opd` against `original` at the
  `pass@4` threshold (at least one passing draw counts as a pass), classifying each
  prompt once into `retained`, `lost`, `gained`, or `still_failing`.

Preserve the supplied function signatures, return-dict keys, and surrounding files:

- Do not modify `check_analysis.py`, `analyze_results.py`, `run.sh`, `schema_sft.py`,
  `schema_opd.py`, `evaluate.py`, `booking_report.py`, or files under `data/`.
- Run `python3 check_analysis.py` after any edit and fix failures without weakening
  the tests.
- You may write separate plotting or inspection scripts for your report.
