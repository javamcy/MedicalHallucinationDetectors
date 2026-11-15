# benchaudit

Screen a vision--language benchmark for evaluation-protocol sensitivity, without
training a model on the images.

A benchmark can look hard while its negatives are trivially separable. If the
wrong answers are longer than the right ones, or of a different answer type, then a
detector can score highly without ever looking at the image. `benchaudit` estimates
how much of a benchmark's measured AUROC is exposed to that leak, using nothing but
the question and the answer strings.

## Install

```bash
pip install -e .
```

Requires Python 3.9 or newer, numpy and scikit-learn. No GPU, no model weights.

## Use

Build the same benchmark twice: once with unconstrained negatives, once with
negatives that are matched to the positive answers in length and answer type.
Each file is JSONL with `question`, `answer` and `label` (1 = supported, 0 = not).

```jsonl
{"question": "is there a pneumothorax?", "answer": "yes", "label": 1}
{"question": "is there a pneumothorax?", "answer": "a large left-sided pneumothorax with mediastinal shift", "label": 0}
```

Then:

```bash
benchaudit --leaky leaky.jsonl --matched matched.jsonl --json audit.json
```

```
leaky   rung: AUROC 0.663  (4000 rows, 800 test)
matched rung: AUROC 0.764  (4000 rows, 800 test)
oracle drop : -0.101
predicted detector drop: +0.002   (interval -0.062 .. +0.066)
```

`oracle drop` is how much a purely lexical classifier loses when the negatives are
matched. `predicted detector drop` maps that onto the AUROC loss a trained
vision--language detector would show between the image-split rung and the matched
rung.

## What the number means, and what it does not

The regression behind `predicted detector drop` was fitted on eleven benchmarks:
nine image-VQA benchmarks and two document and remote-sensing ones. Its
leave-one-out mean absolute error within that set is 0.029.

It was then tested prospectively on three benchmarks it had never seen. One of the
three predictions landed within 0.03, all three within 0.06, and two of three
pairwise orderings were correct. The mean absolute error was 0.038.

So: use it to rank benchmarks, and read the interval rather than the point. The
tool prints a warning when the oracle drop falls outside the range seen in the
fitting set (0.000 to 0.135), because that is where its error roughly doubles.

## Why the two rungs

- **Leaky.** Negatives are whatever the benchmark happens to offer, so they may
  differ from positives in length, answer type or frequency. A lexical classifier
  can exploit that, and so can a detector that ignores the image.
- **Matched.** Negatives are chosen to match the positives in length and answer
  type. The lexical classifier has little left to exploit.

The gap between the two is the part of the benchmark's difficulty that is an
artefact of construction.

## Programmatic use

```python
from benchaudit import audit

res = audit(leaky_rows, matched_rows)
print(res.oracle_drop, res.predicted_detector_drop, res.interval)
```

`leaky_rows` and `matched_rows` are lists of dicts with the same three keys as the
JSONL files.

## Limitations

- The feature set is lexical. It cannot see whether an answer is *true*.
- The regression is calibrated on eleven benchmarks; do not read it as a law.
- The tool says nothing about whether a detector conditions on the candidate
  answer. Two widely used families, semantic entropy and RadFlag, do not, and on
  any paired benchmark they therefore score exactly 0.500 whatever the data. A
  benchmark can pass this audit and still be unable to distinguish those families
  from chance.

## Citation

If this is useful, please cite this repository.
