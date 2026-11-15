# Protocol-ladder audit for hallucination-detection benchmarks

Code and item-level results for auditing how much of a reported hallucination-detection
AUROC is attributable to the evaluation protocol rather than to the image. The
repository provides three things: a six-rung protocol ladder that holds a detector,
its features and its training fixed while varying only negative construction and
split; the per-item scores, pair constructions and control results behind every
released number; and `benchaudit`, a training-free tool that screens a paired
benchmark for answer-pool leakage using question and answer strings alone.

No benchmark images are redistributed.

---

## 1. Repository layout

```
code/                          analysis, figure and table generators (Python)
benchaudit/                    installable package: protocol sensitivity from text alone
results/
  ladder/                      protocol-ladder runs, five medical benchmarks
  detectors/                   detector reproductions and benchmark runs
  judges/                      frozen-judge score arrays (NPZ = per-item scores)
  vase/                        VASE caches (NPZ = per-item semantic entropy / RadFlag)
  pairs/                       matched-protocol pair constructions, one JSONL per benchmark
  matched_scores/              per-sample detector scores on the matched test sets (CSV)
  audit/                       negative-construction audit, per benchmark
  seeds/                       five-seed retraining of the three heads (JSON)
  controls/                    image-substitution and pixel-permutation controls (JSON/NPZ)
  clinical/                    clinical error types, adjudication sheet and decision curve
  crossdomain_results.json     protocol-level results, eleven benchmarks
  rank_stability.json          detector rankings at the leaky and matched rungs
  prospective_validation_results.json   three held-out benchmarks
  example/                     one item carried through every rung
  archive/                     earlier batch of per-configuration summaries
figures/                       main figures 1-6 and supplementary figures 1-5 (vector PDF)
tables/                        LaTeX sources of the released tables
docs/                          literature coding instrument, benchaudit manual, report card
pyproject.toml                 makes `benchaudit` installable
```

Two levels of data are shipped:

* **Per-item scores** exist for the five medical benchmarks (VQA-RAD, Path-VQA,
  OmniMedVQA, SLAKE, POPE), in `results/judges/*.npz`, `results/vase/*.npz`,
  `results/matched_scores/*.csv`, `results/pairs/*.jsonl` and the ladder JSONs.
* **Protocol-level results** exist for all eleven benchmarks in
  `results/crossdomain_results.json`, which records `null` wherever a cell was not
  measured. No value in that file is inferred, interpolated or reconstructed.

`code/recompute_tables.py` recomputes the headline numbers from the files in
`results/` and needs no GPU:

```bash
python code/recompute_tables.py
```

---

## 2. The protocol ladder

A *rung* is a pair-construction rule plus a split rule, nothing else. A negative is
drawn from the answer pool of the same benchmark, always from a different image,
never equal to the positive.

| rung | length match | format match | split | role |
|---|---|---|---|---|
| `L0_leaky` | no | no | pair-level | unconstrained answer pool |
| `L1_imgsplit` | no | no | image-level | isolates the split rule |
| `L2_len` | yes (<=2 tokens, widening 5, 10) | no | image-level | isolates length |
| `L3_type` | no | yes (yes/no, numeric, single, short, long) | image-level | isolates format |
| `L4_fair` | yes | yes | image-level | **the matched rung used throughout** |
| `L5_fair_row` | yes | yes | pair-level | L4 pairing, leaky split |
| `L6_native` | - | - | image-level | the benchmark's own authored wrong options |

Matching is defined on the **answer only**, never on the image or the question.
Splits are 60/20/20. The image-level split is keyed on a content hash of the decoded
image, so no image contributes pairs to more than one split. Sampling uses a fixed
seed (42).

---

## 3. benchaudit

`benchaudit` estimates how much of a benchmark's AUROC is exposed to the answer-pool
leak using only the question and answer strings: no GPU, no model weights.

```bash
pip install -e .
benchaudit --leaky leaky.jsonl --matched matched.jsonl --json audit.json
```

Each input is JSONL with `question`, `answer` and `label` (1 = supported, 0 = not).
Build the benchmark twice, once with unconstrained negatives and once with negatives
matched to the positives in length and answer type. The tool reports the AUROC at
each construction and the drop between them. See `docs/benchaudit.md`.

The 20-feature gradient-boosted screen is `code/audit_benchmark.py`; its outputs on
three held-out benchmarks are in `results/prospective_validation_results.json`.
A six-item report card that runs on a pair file and a score file is
`code/reporting_checklist.py`, described in `docs/REPORTING_CHECKLIST.md`.

---

## 4. Reproducing figures and tables

```bash
python code/make_figs_main.py           # main figures 1-4
python code/make_figs_supp.py           # supplementary figures 2-4
python code/make_fig_example.py         # supplementary figure 5 (one item through the ladder)
python code/make_figs_crossdomain.py    # cross-domain figures 5 and 6
python code/make_tables_crossdomain.py  # tables/supp_tab_*.tex
python code/rank_stability.py           # results/rank_stability_recomputed.json
```

Generators write into `figures/` and `tables/` in place, from the files in
`results/`. `code/audit_benchmark.py` and `code/answer_conditioned_entropy.py`
additionally import `xdomain.py`, the loader for the non-medical corpora. That module
is not redistributed because those benchmark images are not ours to redistribute;
`benchaudit` covers the same screen from plain JSONL and needs no loader.

---

## 5. Running the experiments

`DATA_DIR` points at the benchmark images (VQA-RAD, Path-VQA, OmniMed-VQA-mini,
SLAKE, POPE + COCO `val2014`). `BENCH_BASE` overrides the default cache location.

```bash
python code/e1_protocol.py --ds vqa_rad --levels L0_leaky,L1_imgsplit,L2_len,L3_type,L4_fair,L5_fair_row
python code/e1_protocol.py --ds path_vqa
python code/e1_protocol.py --ds omni            # adds L6_native
python code/bench2.py     --ds slake
python code/bench2.py     --ds pope_random
python code/e1_protocol.py --ds <bench> --train-mlp      # 4-layer MLP on frozen CLIP-L/14
python code/llava_lora.py  --ds pope_random --seed 0     # LoRA on the protocol
```

The detector families are driven by `code/e6_family.py` (`gen` / `score` / `eval`);
`--tag` selects the backbone. The frozen groundedness judge scores
`logit(grounded) - logit(ungrounded)` at the first scoring position and is stored
per item in `results/judges/*.npz`. `code/vase_reproduction.py` implements the
sampling-based detector (10 samples, Gaussian input noise, DeBERTa entailment
clustering, token-level semantic entropy and noise-shifted visual entropy).
`code/official_scorers.py` runs CLIPScore and maximum patch-answer alignment.

Clinical error types, the adjudication sheet and the decision curve are produced by
`code/clinical_adjudication.py` and `code/adjudication_risk.py` from the released
per-item matched scores; the image-substitution and pixel-permutation controls are
produced by `code/e12_controls.py` and `code/lora_pixelshuffle.py`.

---

## 6. Statistics

* AUROC is computed once per configuration on the test split; hallucinated answer is
  the positive class.
* Confidence intervals: percentile bootstrap, 1,000 resamples over items.
* Differences between configurations: paired permutation test, 2,000 permutations.
* Rank agreement between the leaky and matched rung: Kendall tau-a and Spearman rho
  over detector families, per dataset and backbone (`code/rank_stability.py`).
* Configurations are always compared at the same rung and on the same rows.

---

## 7. Environment

Python 3.10, `torch` 2.x, `transformers`, `bitsandbytes` (4-bit nf4 for the 32B
arms), `scikit-learn`, `numpy`, `matplotlib`. `benchaudit` needs only Python >= 3.9,
`numpy` and `scikit-learn`.

---

## 8. Data notes

* The five medical benchmarks ship with per-item scores; every number computed from
  them can be recomputed from this repository.
* The cross-domain study ships at protocol level. `results/crossdomain_results.json`
  records the AUROC of every measured cell; cells that were not measured are `null`.
* `results/rank_stability.json` is the artifact of record and covers fourteen
  benchmark-backbone settings. `code/rank_stability.py` recomputes the three medical
  settings from `results/ladder/` and writes them to
  `results/rank_stability_recomputed.json`, so the artifact of record is never
  overwritten.
* The literature coding (`docs/coding_instrument.tex`) is manual; every row links to
  a full text and values are marked *unclear* rather than guessed when a source does
  not state its construction.

---

## 9. License

Code: MIT. Aggregated numeric results: CC BY 4.0. Benchmark images remain under the
licences of VQA-RAD, Path-VQA, SLAKE, OmniMedVQA, POPE, MMMU, ScienceQA, MMBench,
CV-Bench, ChartQA, HallusionBench, DocumentVQA and RSVQA-HR, and are not
redistributed here.
