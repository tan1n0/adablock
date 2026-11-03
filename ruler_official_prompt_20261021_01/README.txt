RULER official-prompt rerun
Remote: /root/adablock_ruler_official_prompt_20261021_01
Source: /root/adablock_ruler_budget_20261021_01

The old run omitted the separate answer_prefix field. Its 357 completed records
and logs are preserved, but it was stopped before finishing because all tasks
require the prompt correction. Old VT-only queued corrections are superseded.

All 450 original contexts/questions/answers are reused in source line order.
Upstream retrieval indices are not necessarily unique. Each old token sequence
is checked before replacing it with encode(input + answer_prefix), without a
second chat template. Encode/decode round trips are asserted on all prompts.

Pilot: first original row in each of 9 task/length cells, selected without scores.
Each row runs Full and S/A/B at 10/15/25/40 percent logical read quota (13 variants).
Initial generation caps: NIAH 128, VT 30. If a task truncates, its cap is raised
uniformly across all methods and lengths through 128, 512, 1024. Pilot cases
are rerun at each attempt; outputs remain in separate pilot directories.
Gate: all 117 responses nonempty and ended with EOS without reaching the cap;
Full must emit numeric/variable-like answers. No correctness threshold is used.
Pilot cannot guarantee that every future evaluation response will fit the cap;
final metrics include truncation rate and all raw predictions are retained.

After a passing pilot, the driver runs all 450 rows on four GPUs, then writes
comparison.json and quality-vs-logical/padded-read curves under evaluation/.
Checkpoint identities, rho=0.90, quantile=0.95, allocation rule, greedy decode,
full causal prefill and shared first output token remain unchanged.
Generation caps beyond official defaults are explicitly recorded if needed.
Nominal 8K/16K/32K bins refer to original generated examples; actual corrected
prompt lengths are recorded. Increased output caps can exceed nominal totals.

status.json is authoritative. Earlier preparation errors remain in logs.
