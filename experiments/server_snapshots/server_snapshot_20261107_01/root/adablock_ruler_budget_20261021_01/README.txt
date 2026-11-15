RULER equal-useful-budget evaluation, 2026-10-21

Remote: /root/adablock_ruler_budget_20261021_01
Frozen source: /root/adablock_sab90_20261019_01/branches/{S,A,B}/best_pair.pt
Tasks: niah_single_1, niah_multivalue, vt; 50 examples/task/length; 450 examples.
Lengths: 8192, 16384, 32768 total RULER sequence limits (generation reserve included).
Read ratios: 0.10, 0.15, 0.25, 0.40; Full baseline once per example.
Fixed-region tokens count toward each layer quota. Native budget q=.95 head counts become proportional weights (floor 1); capped largest-remainder allocation enforces the exact integer middle-block total. No answer-based calibration. This is a budget-renormalized intervention, not native deployment.
Equal useful slots apply at equal sequence positions; padded costs and generated lengths are reported separately.
Full causal prefill shared; first output token shared. Sparse selection starts from the second output token.
Official RULER answer recall and supplementary all-required-answers hit are both reported. The latter does not penalize extra wrong answers.
Data uses upstream generation templates and limits, with no post-generation truncation. Official essay URL contents are cached; HTTPS is used for Paul Graham pages and the upstream Git checkout supplies GitHub raw text files. No original experiment files changed.

Outputs on completion: comparison.json, score_vs_logical_fraction.{png,pdf}, score_vs_padded_fraction.{png,pdf}, all_hit_vs_logical_fraction.{png,pdf}, all_hit_vs_padded_fraction.{png,pdf}, and per-question qa_0..3.jsonl.
