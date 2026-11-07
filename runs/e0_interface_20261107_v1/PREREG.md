# D1: small evidence-interface feasibility diagnosis

Frozen before D1 inference on 2026-11-07, after the completed E0 negative result was exposed.
This is a successor development experiment, not a confirmation on untouched evaluation data.
The 200-case E0 result and both old execution roots remain unchanged.

## Motivation and scope

E0 had 12 terminal initial format failures, and only 153/239 planned tests had registered observations.
Its apparent net accuracy gain was largely associated with the initial-format-failure subset.
Selective and Reset agreed on 196/200 answers and graph closure changed no K.
Do not start gate training. First test whether usable evidence/state construction can be executed
reliably enough to interpret a later method experiment. This does not establish correct observations.

## Data, endpoints and comparisons

32 fresh PMC-VQA training-source questions, seed 2026110703, one per article, excluding every E0
article and all three previously excluded official test article sets. Same lexical visual-question
proxy and limitations; no expert-confirmed visual subset or near-duplicate guarantee.
Keep references in the offline label file only, with unchanged exact-letter evaluation.

Primary feasibility targets: at least 90% of requested public tests produce a registered observation;
no more than 1/32 initial outputs remains unparseable after the single permitted format retry.
These are engineering targets, not correctness or medical safety guarantees. Report all failures.
Secondary descriptive results: all five same-template branches and original WC/CW definitions,
K coverage, Selective/Reset disagreement, graph changes, paired bootstrap intervals. 32 cases do
not establish clinical effectiveness or power a claim of superiority. No winner-based prompt tuning.

## Exactly what changes from E0

1. Initial JSON requests answer first, followed by the existing claim-list/parent interface.
2. Invoke Observer separately for each fixed neutral test, with one full image and one question.
3. Test/source provenance is assigned from that actual execution context, not from generated IDs.
   Preserve the complete original model output and the execution binding. Bind only exactly one
   observation with nonempty text and a valid adequate/limited/unavailable status. Never invent or
   semantically repair the observation, quality, factual value or physical measurements.
4. Observer prompt explicitly requests one-item JSON. All other evidence qualification and repair
   semantics remain unchanged, including two maximum tests and one retry per role invocation.

Every candidate sees exactly the same newly acquired E. Raw/Selective/Reset/Random-drop/Summary
repair prompts are identical. Old answer remains hidden from every repair and observer.
No gate, RL, crop, external model or expert oracle is introduced. Answers/observations remain fallible.
More Observer calls count in all costs; higher field coverage is not evidence of better perception.
The generator/model revision remains fixed. Existing server environments may differ as measured in
the separate timing study; any cross-E0 accuracy difference is therefore descriptive and confounded.
Within D1, all branches use the same frozen stack and inputs.

## Execution, budgets and decisions

Use the operational throughput check to choose the available server, without changing old E0.
Read-only reuse of existing weights; new D1 input/log footprint expected below 0.1 GB.
Actual per-server added storage must remain <50 GB, stop at 40 GB including conservative project
budget accounting. Setup/read-write probe must succeed. No other process is stopped or modified.
Choose sufficient free GPU memory; neutral visible paths apply to main and child processes.
Hard wall time: 55 minutes from D1 setup start, GNU timeout plus detached supervisor;
leave 120 seconds for termination and evaluation. One bounded 32-case run, no automatic costly rerun.

If feasibility targets fail, inspect logs and prepare a small audit package instead of repeatedly
regenerating larger datasets. If targets pass but Selective still lacks a distinct state effect,
redesign the state intervention/hypothesis before large training. This is not the final method and
does not reduce the research objective to an interface fix. Keep negative evidence and all exposures.

The same chat heartbeat handles progress, analysis and delivery. Publish only code/protocol/aggregate
results after completion; raw images, references and generated case content stay private.
