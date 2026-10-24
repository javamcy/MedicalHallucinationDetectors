# Medical hallucination detectors do not measure visual grounding

Data and code for the manuscript *"Medical hallucination detectors do not measure visual grounding."*

The paper's claim is that the reported AUROCs of medical vision–language hallucination detectors
are not measurements of grounding, because the evaluation protocol lets a detector succeed without
looking at the image. This repository contains every score behind every number in the paper, plus
the code that produced them.

---

## 1. What is here

```
code/                      all analysis code (Python)
results/
  ladder/                  protocol-ladder runs: E1, E2, E4, E5, E6, E8, E9
  detectors/               published-detector reproductions, benchmark runs, VASE, LoRA
  judges/                  frozen-judge score arrays (NPZ = per-item scores)
  vase/                    MedGemma-4b VASE caches (NPZ = per-item SeEnt/RadFlag/ViSeEnt)
  archive/                 earlier batch of per-configuration summaries
figures/                   figure sources (filled by the generators in code/)
```

`code/recompute_tables.py` recomputes the headline numbers of every table from the files in
`results/` and needs no GPU:

```bash
python code/recompute_tables.py
```

---

## 2. The protocol ladder

A *level* is a pair-construction rule plus a split rule, nothing else. Negatives are drawn from
the answer pool of the same benchmark, always from a different image, never equal to the positive.

| level | length match | format match | split | role |
|---|---|---|---|---|
| `L0_leaky` | no | no | row-level | the protocol most of the literature uses |
| `L1_imgsplit` | no | no | image-level | isolates the split rule |
| `L2_len` | yes (≤2 tokens, widening 5, 10) | no | image-level | isolates length |
| `L3_type` | no | yes (yes/no, numeric, single, short, long) | image-level | isolates format |
| `L4_fair` | yes | yes | image-level | **the matched rung used throughout** |
| `L5_fair_row` | yes | yes | row-level | L4 pairing, leaky split |
| `L6_native` | — | — | image-level | OmniMed's own authored wrong options |

Matching is defined on the **answer only**, never on the image or the question.

---

## 3. What reproduces what

| Item in the paper | Code | Data |
|---|---|---|
| Fig. 1 (consensus report card) | `code/make_figs_nmi.py` | `results/ladder/`, `results/detectors/` |
| Fig. 2a–c (image value, ladder, frozen statistics) | `code/make_figs_nmi.py` | `ladder/e1_protocol_*.json` |
| Fig. 3a–c (question types, yes/no, error types) | `code/make_figs_nmi.py` | `ladder/e7_failure.json`, `e9_yesno.json` |
| Fig. 4 (all detectors, matched protocol) | `code/make_figs_nmi.py` | `detectors/`, `ladder/e6_family_*` |
| ED Table 1 (ladder in numbers) | `code/e1_protocol.py` | `ladder/e1_protocol_*.json` |
| ED Table 2 (published detectors) | `code/official_scorers.py`, `code/vlm2.py`, `code/vase_reproduction.py`, `code/vase_medgemma.py`, `code/llava_judge_v2.py` | `detectors/`, `judges/`, `vase/` |
| ED Table 3 (detector families) | `code/e6_family.py`, `code/vlm2.py` | `ladder/e6_family_*`, `detectors/vlm2_*` |
| ED Table 4 (failure cases) | `code/e7_failure.py` | `ladder/e7_failure.json` |
| ED Table 5 (frozen judge) | `code/e6_family.py` (`score`), `code/vlm2.py` (`judge`) | `judges/llm_*.npz` |
| ED Table 6 (image value) | `code/e1_protocol.py`, `code/bench2.py` | `ladder/e1_protocol_*`, `detectors/bench2_*` |
| ED Table 7 (evaluation practice) | manual coding; the coded instrument is `docs/coding_instrument.tex` | `docs/coding_instrument.tex` |
| ED Fig. 1–3 | `code/make_figs.py`, `code/make_ed_figs.py` | `ladder/` |
| ED Fig. 5 (one item through the ladder) | `code/make_edfig_example.py` | `example/worked_example_vqa_rad_pid1705.csv`, `figures/vqa_rad_142.png` |
| ED Fig. 4 (image value, not the prior) | `code/make_ed_figs.py` | `ladder/`, `detectors/bench2_*` |

---

## 4. Running the experiments

Everything ran on a single 24 GB GPU. `DATA_DIR` points at the benchmark images (VQA-RAD,
Path-VQA, OmniMed-VQA-mini, SLAKE, POPE + COCO `val2014`).

### 4.1 Protocol ladder

```bash
python code/e1_protocol.py --ds vqa_rad --levels L0_leaky,L1_imgsplit,L2_len,L3_type,L4_fair,L5_fair_row
python code/e1_protocol.py --ds path_vqa
python code/e1_protocol.py --ds omni            # adds L6_native
python code/bench2.py     --ds slake            # SLAKE, matched protocol
python code/bench2.py     --ds pope_random      # POPE, authored negatives
```

### 4.2 Detector families (sampling statistics, verification, internal signals)

```bash
cd code
python e6_family.py gen   --ds vqa_rad --tag q3b --levels L0_leaky,L4_fair --k 5
python e6_family.py score --ds vqa_rad --tag q3b --levels L0_leaky,L4_fair
python e6_family.py eval  --ds vqa_rad --tag q3b --levels L0_leaky,L4_fair
# same three commands for --ds path_vqa / omni, and for the SLAKE / POPE loaders in bench2
```

`--tag` selects the backbone (`q3b`, `q7b`, `q32b` with `QUANT4BIT=1`, `l7b`, or a local path).

### 4.3 Frozen groundedness judge

The judge is asked a single question — *"Is this answer grounded in the image?"* — and its score is
`logit(grounded) − logit(ungrounded)` at the first scoring position, so no free text is generated
and scores are comparable across items. Scores are stored per item in `results/judges/*.npz`
(`judge`, `nll`, `y`, `idx_te`).

### 4.4 VASE (faithful reproduction)

`code/vase_reproduction.py` implements the authors' algorithm at full scale: 10 samples, main answer
at temperature 0.1, sampling at 1.0, Gaussian input noise σ = 0.07, α = 1.0, DeBERTa
transitive-closure entailment clustering, token-level semantic entropy and the noise-shifted visual
entropy. Backbones: LLaVA-Med-1.5 (`load_all`) or MedGemma-4b (`vase_medgemma.py`, which
monkey-patches only the model layer so the scoring mathematics stays bit-identical).

### 4.5 Official scorers

`code/official_scorers.py` runs CLIPScore from the authors' own implementation
(`jmhessel/clipscore` with `clip-anytorch`, ViT-B/32) and the maximum patch–answer alignment.
Everything else is implemented from the published description; where public code exists our statistic
reduces to the authors' own up to a positive constant, which leaves AUROC unchanged.

### 4.6 Trainable baselines

```bash
python code/e1_protocol.py --ds <bench> --train-mlp      # 4-layer MLP on frozen CLIP-L/14
python code/llava_lora.py  --ds pope_random --seed 0     # LLaVA-Med-1.5 + LoRA on the protocol
```

MLP: hidden widths 1024/512/256, batch norm, ReLU, dropout 0.4/0.4/0.3, standardised inputs,
AdamW lr 1e-3, weight decay 1e-4, 120 epochs, model selection on validation only, one test evaluation.

---

## 5. Statistics

* AUROC is computed once per configuration on the test split, hallucinated answer = positive class.
* Confidence intervals: percentile bootstrap, 1,000 resamples over items.
* Differences between configurations: paired permutation test, 2,000 permutations, following the
  logic of paired comparisons of correlated AUROCs.
* Configurations are always compared at the same level and on the same rows.

---

## 6. Environment

Python 3.10, `torch` 2.x, `transformers` (Qwen2.5-VL, Gemma3, LLaVA-Med), `bitsandbytes` 0.45
(4-bit nf4 for the 32B judge), `scikit-learn`, `numpy`, `matplotlib`. Figures use the Nature
column widths and 6–7 pt sans type.

---

## 7. Provenance and known limitations

* Every number added in the final revision (SLAKE, POPE, Path-VQA families; Lingshu-7B and
  Qwen-32B judges; LLaVA-Med and MedGemma VASE) comes from a run whose per-item scores are stored
  here and can be recomputed.
* The VASE-full / judge cells for VQA-RAD, Path-VQA and OmniMed in Extended Data Table 2 for the
  LLaVA-Med backbone, and the 32B and Lingshu judge values on VQA-RAD and Path-VQA, were carried
  over from an earlier experiment batch whose raw score files were not retained. They are consistent
  with the surrounding rows but are **not** recomputable from this repository.
* Two numbers in the section "What survives" are likewise not recomputable from the files here: the
  yes/no-block score of the protocol-trained LoRA model (0.821) and that of the frozen-feature
  detector on the same block (0.679). They came from an interactive analysis whose per-item output
  was not archived. The third number in that sentence, the cosine similarity between the CLIP text
  embeddings of "yes" and "no" (0.949), was re-measured with CLIP-L/14 and returns 0.9489.
* Med-HallMark was obtained but not used: its `response` field is a hallucination-category label,
  not an answer, so no positive/negative pair can be built from it.
* RULE, MMedPO and MMed-RAG were excluded because they improve the factuality of generated text and
  produce no per-answer score; they are not commensurable with a fixed-pair protocol.
* CHAIR and BiomedCLIP were excluded because we did not run their published implementations.

## 8. License

Code: MIT. Aggregated numeric results: CC BY 4.0. Benchmark images remain under the licences of
VQA-RAD, Path-VQA, SLAKE, OmniMedVQA and COCO and are not redistributed here.
