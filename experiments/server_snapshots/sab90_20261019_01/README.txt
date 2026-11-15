Controlled S/A/B at 90% attention-mass coverage

Remote directory: /root/adablock_sab90_20261019_01

All branches use the original pilot ranker and the previous rho90 budget
checkpoint as the same warm start. Networks, normalization, data, budget
CE objective, optimizers, update counts, deployment quantile and QA
protocol are identical across branches.

S: rank0 -> rank1 -> budget0 -> budget1
A: rank0 -> budget0 -> rank1 -> budget1
B: same as A, adding actual-K boundary supervision for ranker training.

Each rank phase uses one pass over 1536 groups (12 documents x 32 layers x
4 positions), with all 8 KV heads kept together and 8 groups per optimizer
step: 192 steps per phase, 384 total. Ranking positives use rho=.90.
B additionally samples up to 32 inverted teacher-mass pairs across the
hard selected/unselected boundary. K is fixed for a rank phase, while
top-K membership is computed from current scores and detached. The
original RankLoss and boundary loss have equal coefficient. The common
pair cap is 192 per scene; actual pair numbers are recorded, not assumed
equal. No gradient passes through hard top-K or the frozen K.

Budget data use the same 48 WikiHop training and 12 held-out calibration
questions as the preceding frozen-ranker experiment. Original full and
old-adaptive trajectories are reproduced. Query, per-block teacher mass
and immutable block summaries are added so both features and labels can
be refreshed after ranker updates. Complete future blocks in a saved
digest are never accessible to an earlier scene; candidate IDs must be
causal and completed. Teacher attention is used only offline.

Every budget phase refreshes features, coverage curves and minimum-budget
labels with the current ranker. Features use identical frozen initial
normalization. Training uses original unweighted CE, AdamW lr=.001,
weight decay=.01, batch size=512, gradient clip=1, and 20 epochs per phase
(40 total). The deployed pair has the final ranker and minimum held-out
CE budget checkpoint in the last 20 epochs. No new QA score selects a
checkpoint. Both optimizers preserve state across phases.

The budget label target is rho=.90. The separate deployment probability
quantile remains q=.95 for every branch. There is no quantile sweep,
new cost penalty, change to budget buckets, or shared-model redesign.

The exact same frozen 32 HotpotQA training questions from the previous
90/95 run are used. This is an already-inspected development set, not a
new independent test. All branches share prompt arrays, full prefill,
first-token logits, BF16 Llama-3.1-8B-Instruct, max 32 generated tokens,
greedy decoding, original EOS set, block16, sink32 and local32 blocks.
Full attention is rerun and checked against previous full token outputs.

QA reports F1, effective token slots, padded token slots, decode time and
prefill+index+decode time. Separate instrumentation diagnoses ranker,
feature construction, budget prediction and selected attention overhead.
Slots are proxies, not measured memory traffic. Runtime includes the
current unoptimized Python/index implementation, with concurrent workers.

Files:
  protocol.json, protected_hashes.json: fixed plan and source identities
  raw_qa/: refreshable offline data (shared by S/A/B)
  collection_verification.json, refresh_parity.json: reconstruction checks
  branches/S, branches/A, branches/B: paired checkpoints and histories
  branches/*/supervision_*.pt: rank-version-bound regenerated supervision
  qa_*.jsonl, comparison.json: paired question results and aggregate report

Original files are read-only inputs. The results archive contains code,
models, manifests and outputs; large raw/derived training tensors remain
in the independent remote directory and are identified by hashes.
