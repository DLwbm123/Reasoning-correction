# Reasoning-correction mechanism pilot

Development-only medical VQA context-reconstruction experiment, requested for GPU 6.
See [frozen protocol](runs/e0_20261107_v1/PREREG.md).

Implemented: grouped PMC-VQA sample preparation; reference-free inference DTOs; role isolation;
blind observations; evidence binding; graph/flat-state accounting; five same-template mechanism branches;
deterministic branch reuse; token/forward/memory/latency traces; separate exact-letter evaluator;
paired bootstrap; detached supervisor with deadline and storage guards.

Executed locally: 9 permission/state/outcome-accounting unit tests and Python compilation. Data manifest: 200 unique
source articles, no article overlap with the three excluded official test metadata files.
Remote launch: jiangsuiyang/zmic44 physical GPU 6, experiment `e0_20261107_v1`.
Model download and adapter import succeeded; checkpoint loading has started. At this snapshot,
real model generation and medical performance evaluation are not yet confirmed complete.
Runtime progress and results are under the experiment's NAS outputs, not implied by unit tests.

Not implemented/run: learned gate, expert oracle observations, crop tools, independent expert process audit,
clinical evaluation, final held-out generalization study, or exhaustive image near-duplicate auditing.
This pilot does not establish the full method's effectiveness or clinical safety.

Local checks: `python -m unittest -v test_mechanism.py test_evaluation.py`.
`prepare_data.py` reads the already downloaded dataset from its frozen local source path.
On the server, set `RUN_ROOT` and execute `run_e0.py`; after collection exits, execute
`evaluate_e0.py` in a separate process. `supervise.py` is the configured neutral-entry wrapper.
Do not rerun preparation or collection over existing frozen outputs.

Raw images, private labels, generated case contents and model weights are intentionally ignored by Git.
