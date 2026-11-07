# Read-only training-data feasibility audit

No model generation, training or reference regeneration. Preserve E0/D1/D2.
Inspect local ClinHallu VQA-RAD/PathVQA train references and their original HF source images.
Reconstruct the official download converter's first-occurrence image naming from decoded pixels;
validate original_index, question, answer and image name before linking a record.
Read held-out source Parquet files with the image column ONLY to exclude shared images.
Pixel fingerprints are scientific grouping keys for split exclusion, not file-transfer verification;
no model/archive checksum or claim of near-duplicate exclusion.

After the first pass, the planned VGMED/SLAKE ROI stratum had zero usable boxes or masks (971
training-linked rows). Preserve that finding and the initial eight-case packet. A separately marked
extension may add eight ORIGINAL SLAKE mask/detection cases, seed2026110706, with at least two
registered nonzero labels and matching image/mask dimensions, still excluding English held-out
images. This is a source-availability adaptation before any model performance, not filling VGMED
with invented labels. All source types remain explicit; maximum total remains16 private examples.

Seed 2026110705. Select at most 16 training examples without using any new model correctness:
two yes/no and two open-ended examples from each ClinHallu source (eight total), and up to eight
distinct English-SLAKE-training images with locally available VGMED boxes/masks (eight total).
If a stratum cannot be linked safely, report the shortfall rather than substituting test examples.
Persist images/reference text/individual source keys only in gitignored private_analysis.
Publish aggregate mapping, split and annotation-availability findings, with explicit uncertainty.

Image inspection is a preliminary model-assisted audit, not expert clinical adjudication. Distinguish
multiple written claims from independently verified visual facts, source ROI annotations from
answers about abnormality, and lexical answer overlap from semantic answer disclosure.
Do not infer physical measurements/patient laterality or fabricate longitudinal clinical records.
