# Reasoning-correction mechanism pilot

Development-only medical VQA context-reconstruction experiment, requested for GPU 6.
See the completed [v2 frozen protocol](runs/e0_20261107_v2/PREREG.md).

E0 and the [32-case evidence-interface feasibility diagnosis](runs/e0_interface_20261107_v1/REPORT_ZH.md) are complete.
D1 registered 42/42 observations with zero initial-format failures, but Selective/Reset/Summary/Random-drop
gave identical answers in all 32 cases. The current context-state mechanism remains unsupported.
D1 changed initial answer formatting and per-test observation execution, not the completed E0 result.
The [24-image annotation/history positive control](runs/oracle_replay_20261107_v1/REPORT_ZH.md) is also
complete: all answer-revealing oracle conditions reached 24/24, leaving no state-reconstruction gap
on that narrow synthetic task. No large gate/RL training is started.
The [training-data feasibility audit](runs/data_audit_20261107_v1/REPORT_ZH.md) is complete: original
ClinHallu image mapping was validated, known held-out image groups were excluded, and a private
16-case packet was reviewed preliminarily. Original SLAKE masks provide a limited multi-label source;
these annotations and model-generated traces are not new expert adjudications.
The [multi-fact state design](STATE_REVISION_DESIGN.md) now has a small declared-support CPU prototype:
six checks pass, including 256 evidence-state transitions. This is not model-performance evidence.
The user's next instruction prioritizes running a trainable baseline without a novelty gate:
[G1](runs/gate_baseline_20261107_v1/REPORT_ZH.md) completed on physical GPU 6 of PRO5000.
It reused the frozen E0 candidate pool, extracted frozen features, and trained six four-class MLPs.
All calibrated gates chose KEEP on the 40-case retrospective diagnostic subset (35%, no improvement);
the 20-case calibration subset contained no WC event. The training pipeline works, but a gate benefit
is unsupported. No GPU job from G1 remains active. Independent clinical verification is unavailable;
this is not a full-method or untouched-test evaluation.
The [fresh-pool G2 result](runs/gate_pool_20261107_v1/REPORT_ZH.md) is complete:
1,600 new article groups, six trained gates, 43.17% mean diagnostic accuracy per training configuration
versus 43% KEEP and 44.5% always-repair. No stable gate gain was established.
All G2 GPU work has ended. The [G3 result](runs/action_gate_20261107_v1/REPORT_ZH.md) is complete:
conditional heads averaged 41.83% versus 42% KEEP and 43–43.33% frozen-head rule controls.
This branch did not improve the baseline. All G3 jobs have exited. The
[evidence-audit protocol](runs/evidence_audit_20261108_v1/PREREG.md) targets the original missing
candidate evidence-audit module. The [G4 result](runs/evidence_audit_20261108_v1/REPORT_ZH.md)
completed all 64 calls but failed its frozen engineering gates (24/64 valid requests, 22/64 effective
paired agreements). Full audit expansion and associated training stopped. Wrapper-only offline
normalization still failed; no clinical correctness or permutation instability is inferred.
The [CPU capacity diagnosis](runs/capacity_probe_20261108_v1/REPORT_ZH.md) also completed:
six smaller four-class heads did not improve development utility ranking versus paired frozen MLPs.
No old diagnostic rows were used, and no new generator or feature calls were made. This capacity
branch is closed. The [G6 frozen protocol](runs/evidence_score_20261108_v1/PREREG.md) now implements evidence-conditioned
forced-letter support features. Its [launch receipt](runs/evidence_score_20261108_v1/LAUNCH.json)
records old-cache scoring in progress on PRO5000 GPU 6; fitting and new diagnostic evaluation have
not yet completed. The extra scoring cost and software-stack difference from cached G2 vectors
are explicit experimental limitations.
The [hardware timing check](runs/throughput_20261107_v1/REPORT_ZH.md) found comparable throughput in
the currently shared environments; no migration speedup or hardware-only attribution is claimed.

Implemented: grouped PMC-VQA sample preparation; reference-free inference DTOs; role isolation;
blind observations; evidence binding; graph/flat-state accounting; five same-template mechanism branches;
deterministic branch reuse; token/forward/memory/latency traces; separate exact-letter evaluator;
paired bootstrap; detached supervisor with deadline and storage guards.

Executed locally: 15 permission/state/format/execution-binding/outcome-accounting unit tests and Python compilation. E0 manifest: 200 unique
source articles, no article overlap with the three excluded official test metadata files.
Historical launch: jiangsuiyang/zmic44 physical GPU 6, experiment `e0_20261107_v2` (completed).
The first v1 interface attempt was stopped after the first trace exposed non-neutral observation
questions. Its two completed cases / 33 calls and raw failures are preserved; no accuracy labels were
opened and it is not scientific effectiveness evidence. V2 uses program-owned neutral questions and
separates valid answer letters from explanation schema failures. It shares the original sample,
weights, overall time deadline and storage budget, without overwriting v1.
V2 completed 200/200 cases and offline evaluation successfully. Initial accuracy was 41%;
Raw/Selective/Reset reached 45%, random-drop 44%, evidence-summary 46%. No additional Selective
benefit was established. Graph closure changed K in zero cases and K was empty in 109/200.
See the [full result and limitations](runs/e0_20261107_v2/REPORT_ZH.md).
Twelve initial-format failures materially confound apparent improvement; primary results are preserved.
Runtime progress and results are under the experiment's NAS outputs, not implied by unit tests.

Not implemented/run: expert oracle observations, crop tools, independent expert process audit,
clinical evaluation, final held-out generalization study, or exhaustive image near-duplicate auditing.
This pilot does not establish the full method's effectiveness or clinical safety.

Local checks: `python -m unittest -v test_mechanism.py test_evaluation.py test_format.py`.
`prepare_data.py` reads the already downloaded dataset from its frozen local source path.
On the server, set `RUN_ROOT` and execute `run_e0.py`; after collection exits, execute
`evaluate_e0.py` in a separate process. `supervise.py` is the configured neutral-entry wrapper.
Do not rerun preparation or collection over existing frozen outputs.

Raw images, private labels, generated case contents and model weights are intentionally ignored by Git.
