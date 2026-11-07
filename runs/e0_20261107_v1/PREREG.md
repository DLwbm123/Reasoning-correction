# E0: evidence-conditioned context reconstruction, development mechanism pilot

Frozen before the first model inference on 2026-11-07 (Asia/Shanghai).

## Question and intervention

Given the same medical image, question, frozen initial answer and newly acquired observations,
does selective context reconstruction improve the next answer compared with raw context,
fresh reset, count/length-matched random filtering, and evidence-only summarization?
The last control tests whether any gain merely comes from restating the same evidence.

This is an untrained, one-round mechanism experiment, not the full CLEAR-Med method.
No gate, RL, candidate acceptance audit, crop, expert oracle evidence, or clinical deployment claim.

## Data and permissions

200 questions from existing PMC-VQA `train.csv`, seed 20261107; one question per source PMC article.
Exclude articles occurring in `test.csv`, `test_clean.csv`, and `test_2.csv` before sampling.
Select using the frozen question-text regexes in `prepare_data.py`, not model correctness or answer values.
This is a broad observable-question proxy, not an expert-certified visual-only subset.
No test generation or test-driven tuning. Global near-duplicate auditing and pretraining contamination
are not established by article exclusion. All samples are development exposures after this run.
The inference manifest contains questions, options, image locations, source groups, fingerprints only.
Reference letters/text are in a separate private file, opened only by a separate offline evaluator.
Original source data are read-only. Raw images, labels, and case-level content are excluded from Git.

## Generator and observation

Frozen Qwen/Qwen2.5-VL-7B-Instruct, revision cc594898137f460bfe9f0759e9844b3ce807cfb5.
Existing transformers 4.49.0 / torch 2.2.1 environment; BF16, SDPA, batch size 1.
Image pixel range 256*28*28 to 768*28*28; identical preprocessing across all branches.
Greedy decoding; fixed per-role maximum output tokens recorded in `run_e0.py`.
Up to 6 initial claims, up to 2 neutral observation questions. Full-image-only observation.
Planner cannot see image or answer options; observer cannot see initial claims/answer, options, or references.
Binder uses frozen observations and no image; unsupported evidence IDs cannot establish contradiction/support.
Model-derived observations are fallible and correlated with generator errors; blinding is not independence.
One fresh-message format retry at most per role; raw failure outputs and all costs are saved.
No old turns or KV cache are carried across generation calls.

## Branches and mechanism accounting

All branches use the EXACT SAME repair instruction/template; only `historical_facts` differs.
Raw: initial facts/inferences without final answer; Selective: canonical supported visual/context facts
after unreliable-node descendant isolation; Reset: empty history; Random-drop: same retained fact count,
closest token length with randomized ties, without seeing error identities; Evidence-summary:
question/evidence-only summary with maximum fact count matched to Selective, without old claim content.
Actual context token counts are recorded; equal fact counts do not imply exactly equal lengths.
Randomize branch execution order, and reuse one result whenever deterministic inputs are identical.

Generate the mechanism branches for ALL selected cases, including originally correct cases, rather
than conditioning collection on detected contradiction. Separately log the proposed deployment trigger.
This removes trigger and learned-gate effects from the primary comparison. Forced-answer A/B/C/D;
uncertainty flags do not remove cases. Do not use automatic audit as a ground-truth process label.

## Outcomes, inference and stopping

Exact answer-letter accuracy; initial and final accuracy, paired delta, W->C, C->W,
repair rate conditional on initial wrong, answer harm conditional on initial correct,
population harm, changed-answer coverage, harm/edit, parse failures, evidence adequacy,
K-empty frequency, graph changes to K, trigger frequency and deterministic branch collapse.
Invalid candidate replacements count as errors (and as changed answers if initial answer was valid).
All collected cases, including pipeline failures, remain in the evaluation denominator.
Unrun cases are explicitly reported; incomplete runs are descriptive, not completed 200-case evidence.
Paired article bootstrap, 2000 replicates, seed 20261107; one case per article.
Main contrasts: Selective minus Raw/Reset/Random-drop/Evidence-summary; report all, no winner picking.
200 cases are a pilot and may be insufficient to distinguish small benefits or rare harmful revisions.
Model prompt outputs and seeds are not modified after examining correctness; hypotheses from results
belong in a separately preregistered successor experiment, with these exposures retained.

GPU: jiangsuiyang/zmic44 physical GPU 6; no other process is stopped or modified.
Preflight minimum free memory 19500 MiB; per-process allocator fraction .78. An OOM stops the run,
preserves evidence, and is not automatically rerun as the same frozen experiment.
Existing shared interpreter is read-only; new weights/cache under this experiment's NAS root.
Expected new disk footprint ~18 GB, conservative stop at 40,000,000,000 bytes,
strict actual limit <50,000,000,000 bytes including small neutral launch files.
23-hour total deadline from setup start, shared by setup/inference/retries/evaluation.
Independent GNU timeout and supervising process enforce deadlines; storage checks every 60 seconds
plus case boundaries. Model download is bounded to the pinned known model files (~16.6 GB).
No downloads or caches fall back to home/root. No costly automatic retry of failed inference.

## Reporting and next decisions

No performance results exist at protocol freeze. Inspect the first real trace for interface/data errors;
such failures are retained, and a corrected successor receives its own run identity if model cost occurred.
Without Selective-vs-Reset/Evidence-summary evidence, do not attribute gains to selective memory.
If most K are empty or graph never changes K, report collapse rather than claiming structural benefit.
If observation errors dominate, evaluate observation quality before adding training or RL.
Expert/oracle observations require additional reviewed annotations and are NOT claimed to be run here.
One 30-minute heartbeat for this chat follows analysis, fault handling, successors and eventual delivery.
Results and reproducible non-data code are published according to global project rules after completion;
private/raw data and model weights are never pushed.
