Frozen ranker: 90% versus 95% attention coverage

Remote run directory: /root/adablock_budget_rho_20261019_01

This experiment reuses the original gpu027 WikiHop QA budget training
features and per-budget coverage curves (48 training, 12 calibration QA).
Both classifiers start with exactly the same random parameters and feature
normalization. Only the minimum-coverage label threshold differs.
The ranker and base Llama model are frozen. The original 588-128-64-9
classifier, CE objective, AdamW settings, 40 epochs, budget buckets,
maximum 32 generated tokens, prompt, EOS and attention implementation
are reused. Both deployment quantiles are fixed at 0.95; there is no
quantile search and no retraining of the ranker.

This is a controlled comparison on the original gpu027 CE implementation,
not a continuation of gpu026's cost/quality constrained training run.

A new 32-question validation set is sampled from official HotpotQA training
data before budget training. Selection uses a fixed random seed, requires
2048 to 8160 prompt tokens, and applies no truncation. Existing questions
and context overlaps with the earlier budget/pilot data and LongBench are
excluded. Answers are used only for final F1 scoring. New QA results do
not select a checkpoint, deployment quantile, or coverage threshold.

Full attention, rho90 and rho95 share prefill caches and first-token
logits. Generation is independent after that point. Execution order is
rotated across questions. Read slots include all fixed tokens. Padded
slots use the largest selected length across KV heads in each layer.
No full attention teacher is used to select deployed budgets.

Main timings are measured without component instrumentation. Separate
seven-step runs measure ranker, feature construction, budget forward and
selected attention. Their synchronization overhead prevents interpreting
them as main end-to-end latency. Token slots are not memory-traffic
measurements or proof of acceleration.

Artifacts:
  protocol.json: frozen experiment settings and source hashes
  qa_selected.json: frozen new question IDs, prompts and answers
  rho90/, rho95/: supervision, initial/best model and training history
  qa_0.jsonl ... qa_3.jsonl: per-question predictions and measured costs
  preflight_*.json: full/custom attention parity checks
  profile_*.json: separate component timing diagnostics
  comparison.json: paired F1, bootstrap interval, costs and timing
  protected_hashes.json: original input/model hashes
  verification.json: completed-run source preservation checks

The new set is a development validation set, not a final benchmark.
An F1 difference within one point is a practical pilot criterion, not
statistical evidence of noninferiority when the paired interval is wide.
Do not tune further settings against these 32 answers.
