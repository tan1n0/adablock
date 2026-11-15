Prepared but NOT launched: gpu027 SSH unavailable as of preparation.

Three harder candidates from the unchanged official task configuration:
- niah_multikey_3: UUID keys and values amid distractor key-value records.
- cwe: identify ten common words across a long list; official one-shot example retained.
- fwe: identify three frequent coded words amid noise.
No promise that these tasks will separate S/A/B; inspect Full and sparse pilot outputs.

Plan: 8K/16K/32K, 50 samples per task/length, 450 total, Full and S/A/B at
10/15/25/40 percent logical read budgets. All existing checkpoints/configuration
remain frozen. New task outputs and official-copied source stay under this directory.
Use generation seed 42 and official generation caps 128/120/50, respectively.
Unlike the old run, max input tokens plus generation cap must fit its nominal bin.

Alignment findings:
- Official prompts are input + answer_prefix. Do not re-wrap the chat template.
- Official CWE includes one worked example with another BOS/chat segment.
- Official defaults are greedy, no string stops, per-task generation caps.
- Metrics and control-character cleanup are loaded from official source functions.
- Prior 5850 predictions rescored with official cleanup: no text or score changes.
- Prior VT maximum generation length 21: its larger cap did not affect termination.
- This remains a 50-sample subset, not official default 500 samples or full suite.
- Official Llama recipe uses vLLM. Our custom HF sparse backend is not an exact
  vLLM reproduction. EOS matching and native HF generation parity remain untested
  until GPU access returns; do not describe them as already passing.
- Full prefill/shared first output token and sparse decode only remain deliberate
  method choices. Slot counts are not measured bandwidth or speedup.

Entry point on gpu027: driver.py using the existing ada Python environment.
Driver calls prepare.py, verifies source bytes against the pinned commit,
checks actual prompt tokenization/lengths, runs native HF-vs-custom decoding
and full/all-block parity checks, then nine-task/length smoke cases (117 outputs).
Truncation, empty outputs, failed native parity or zero-score Full smoke cells
stop before the full evaluation. No task selection by B-A score. Only after the
checks pass does the full run start. All smoke outputs are retained.

Local validation: Python syntax compilation only. Remote data generation, deps,
EOS matching, GPU parity and smoke execution are pending connectivity.

Dispatch: local hidden dispatcher retries connectivity/GPU availability for up to 24 hours.
It uploads and starts the detached audit-and-run driver once ready. This requires
the local computer to remain running. dispatch_status.json records the outcome.
No remote experiment has been launched while status is waiting_for_connection.

Restart update: GPU access restored, 450 inputs prepared and audited. Official
LFS English dictionary hydrated and SHA verified; SciPy installed in extra_deps.
Native decoder and identical-prefill buffer parity passed. Chunked BF16 prefill
can break greedy ties, so its diagnostic is separate from decoder correctness;
longest Full/all-block parity passed. Smoke is running; full run remains gated.
