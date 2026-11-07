# Same-workload timing before successor development experiments

Frozen before benchmark generation on 2026-11-07. This implements the user's accepted plan to
check pro5000 before deciding where to run successor experiments. It is not a medical experiment.

Replay initial and Reset requests for the first three predeclared E0 case indices 0, 10, 20.
The selection is by position, without using correctness. The inputs are the original question,
choices, image and frozen observations. No reference labels are copied into the timing inputs.
Both machines use identical model weights/revision, processor files, prompts, BF16, SDPA,
single-request execution, 256*28*28 to 768*28*28 pixels, greedy decoding, fixed 128 output tokens.
Fixed length intentionally ignores natural stopping to avoid conflating token count with speed.
Raw generated content is discarded; no accuracy or scientific improvement is claimed.
One warmup request, then three repeats of six requests (18 measured requests).

Compare aggregate tokens/second and each request's token lengths. Current shared-GPU utilization
is recorded. Existing usable environments are read-only: the 3090 has torch 2.2.1 / transformers
4.49.0; the Blackwell has torch 2.7.1+cu128 / transformers 4.51.3. This is an operational comparison
of available stacks; any difference cannot be attributed solely to GPU silicon. No quantization,
FlashAttention installation, package upgrade or tuning search is performed.

The project weights copied to pro5000 are reusable read-only resources for subsequent development;
their new actual footprint (~16.6 GB) counts for this experiment. A NAS/disk write-read probe and
capacity check are required. No other process is modified. Choose a GPU with >=19500 MiB free;
nonzero utilization alone is not a blocker. Visible interpreter and script paths remain neutral.

Setup plus execution is bounded to two hours, with copy timeout 20 minutes and standalone inference
timeout at most 10 minutes, and a hard external timeout. Expected new footprint <18 GB per server,
stop before 40 GB, strict limit <50 GB. No full-run retries after failure; preserve failure records.

This is an execution prerequisite for a small new evidence-interface diagnosis. It does not replace
method development, overturn E0's negative result or authorize large gate training.
