# D2: annotation-conditioned history headroom positive control

Frozen before inference on 2026-11-07 after E0 and D1 were exposed. This is a bounded diagnostic
of whether reliable corrective information and active assistant history create a recoverable gap.
It is NOT a natural-error benchmark, a clinical performance result, a replication of a whole paper,
or evidence that the Selective mechanism works. No trained gate or generator update.

## Data and deliberate answer disclosure

SLAKE English TRAIN only; 24 distinct images, eight each of CT/MRI/XRAY, seed 2026110704.
Choose modality questions whose canonical reference agrees with the image's modality metadata;
exclude all English validation/test image groups. Pick by seeded sampling, not model correctness.
The oracle observation states the annotated modality and therefore DIRECTLY REVEALS THE ANSWER.
Supported history likewise states that modality. Contradicted history is synthetic: the other two
modalities are counterbalanced within each class. It is not a naturally generated model error.
Metadata consistency is not an independent expert image audit. No deployment claim is possible.

The sole purpose of this deliberately easy upper-bound control is to distinguish absence of a
context/evidence-use gap from a gap masked by unreliable automatic observations or by the old
user-payload presentation. It does not replace a later annotated medical multi-step task.

## Ten fixed conditions for every image

- No history, without/with oracle E (two conditions).
- Supported/contradicted historical premise quoted in the current user payload, without/with E
  (four conditions).
- The same supported/contradicted premise in an earlier assistant turn, without/with E
  (four conditions).

Actual image, question, final instruction, model/revision, image budget, greedy decoding and answer
format are constant within each comparison. Randomize execution order per image. The selected pair's
other premise, condition label and correctness label are never serialized to the model.
Only oracle-E conditions receive the metadata observation. No reference field enters ordinary
inference; labels are opened by a separate evaluator after collection completes.

Within the assistant condition, supported versus contradicted changes only the premise text.
Quoted versus assistant presentation also changes conversation layout/token count, so its difference
is an operational presentation contrast, not an isolated causal estimate of role tokens alone.
All histories are imposed replay contexts; do not imply the model generated those observations.

## Outcomes and decision rule

Closed answer classes CT/MRI/XRAY, deterministic normalization of case/spaces/hyphen only.
One fresh-message schema retry at most, logged and costed; final invalid output counts wrong.
Keep all 24 collected cases and report unrun cases explicitly if a budget stop occurs.
Report all ten accuracies, counts, and predeclared paired image bootstrap intervals (2000 replicates).
Primary diagnostic contrasts: no-history minus contradicted-assistant, with/without E; oracle gain
under contradicted assistant; supported minus contradicted with E; quoted minus assistant with E.
These 24 cases are a sanity check, not a significance-powered medical study.

If reset-with-E is high but contradicted-assistant-with-E remains lower, active-history interference
has a recoverable upper-bound gap worth studying. If E closes the gap entirely, prioritize reliable
evidence and harder multi-step tasks before another state mechanism; do not fabricate a Selective
benefit. If false history has little impact even without E, this simple task does not exercise the
target phenomenon. All outcomes remain limited by synthetic and answer-revealing construction.

## Execution and limits

Frozen Qwen2.5-VL-7B-Instruct revision cc594898137f460bfe9f0759e9844b3ce807cfb5;
pro5000 existing torch2.7.1+cu128/transformers4.51.3, BF16, SDPA, batch1.
Image budget 256*28*28 to 768*28*28; max48 generated tokens, ten conditions/image (240 calls plus retries).
Only read existing staged weights; expected new input/log footprint <0.1 GB. Strict actual added
storage <50 GB/server, stop at40 GB with conservative project accounting. Read/write probe required.
Physical GPU6 if free memory>=19500 MiB. No other process modified. Main/child visible paths neutral.
20-minute wall budget from setup, external timeout and detached supervisor; reserve120 seconds.
One frozen run, no automatic costly retry, no test-based tuning and no escalating sample search.
The single existing chat heartbeat follows completion, analysis and public aggregate delivery.
