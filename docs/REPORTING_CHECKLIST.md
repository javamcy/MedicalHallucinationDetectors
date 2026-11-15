# Six-item reporting checklist for paired hallucination benchmarks

Fill in the right column when reporting a paired-benchmark result. Each item is
satisfiable with data a study already has; `code/reporting_checklist.py` computes all six
from a pair file and a per-sample score file and emits a pass/incomplete report card.

| # | Item | What to report | Why it changes the number |
|---|---|---|---|
| 1 | **Negative-construction rule** | The sampling pool, whether negatives were length- and format-matched, and the fraction of items where matching failed. | Unmatched negatives let a detector win on answer length or answer format alone. On VQA-RAD, format matching alone moves AUROC by 0.111. |
| 2 | **Image-level split** | The split unit, and how many test items have their source image in training (should be 0). | A pair-level split raises VQA-RAD by 0.105 with the detector unchanged: the head memorises images, not grounding. |
| 3 | **Text-only control** | The identical detector run on question and answer only. | In the benchmarks studied here this control recovers 91-97% of the full detector's AUROC. If the image is doing the work, this number must be lower. |
| 4 | **Leakage control** | One model trained on permuted labels through the same pipeline; report its test AUROC. | Should return 0.50. If it does not, the pipeline itself is generating signal. |
| 5 | **Operating characteristics** | Sensitivity, specificity and their ratio at the chosen threshold; precision at a stated prevalence where a deployment claim is made. | AUROC is invariant to the direction of the error. In these benchmarks missed hallucinations outnumber false alarms by up to 5.9:1 at the Youden point. |
| 6 | **Confidence interval and full-set results** | A CI (bootstrap over items is enough) and the AUROC on the full test set, not a subset. | A 100-pair subset has an interval of roughly +-0.10 and can reverse rankings; a reproduced detector scored 0.669 on a subset and 0.494 on the full set. |

## Filled example: VQA-RAD, matched rung

| # | Value |
|---|---|
| 1 | Negatives drawn from the answer pool, length (<=2 tokens) and format matched; 100% of 2,244 pairs matched (`type+len2`). |
| 2 | Image-level (source image grouped); 0 of 452 test pairs share an image with training. |
| 3 | Question+answer only: AUROC 0.665 versus 0.682 full (97%). |
| 4 | Permuted-label retrain: 0.508 (VQA-RAD), 0.500 (Path-VQA), 0.473 (OmniMed). |
| 5 | Youden threshold: sensitivity 0.471, specificity 0.803; 240 missed hallucinations vs 89 false alarms (2.7:1); precision 16% at 10% prevalence. |
| 6 | AUROC 0.682, 95% bootstrap CI [0.649, 0.717], n = 904 rows / 452 pairs. |

## Minimal machine-readable inputs

* **Pair file** (JSON Lines, one object per pair): `pid`, `gid`, `question`, `pos_answer`,
  `neg_answer`, `question_type`, `match`, `split` in {train, val, test}.
* **Score file** (CSV, one line per row): `pid`, `gid`, `question`, `answer`, `label`
  (0 grounded, 1 hallucinated), `question_type`, `match`, `split`, then one column per
  detector, e.g. `score_full`, `score_noimage`.

Both formats are the ones used in `results/pairs/` and `results/matched_scores/` of this
repository.
