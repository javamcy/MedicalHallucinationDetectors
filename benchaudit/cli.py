"""Command line entry point for benchaudit."""
from __future__ import annotations

import argparse
import json
import sys

from .core import audit


def _read(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            missing = {"question", "answer", "label"} - set(r)
            if missing:
                raise SystemExit("%s line %d: missing %s" % (path, i + 1, sorted(missing)))
            rows.append(r)
    if not rows:
        raise SystemExit("%s: no rows" % path)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="benchaudit",
        description="Screen a vision--language benchmark for protocol sensitivity "
                    "without training any model on the images.")
    ap.add_argument("--leaky", required=True,
                    help="JSONL rows from the unconstrained (leaky) construction")
    ap.add_argument("--matched", required=True,
                    help="JSONL rows from the length- and type-matched construction")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", metavar="PATH", help="also write the full result here")
    a = ap.parse_args(argv)

    res = audit(_read(a.leaky), _read(a.matched), seed=a.seed)
    print("leaky   rung: AUROC %.3f  (%d rows, %d test)" %
          (res.leaky.auroc, res.leaky.n_rows, res.leaky.n_test))
    print("matched rung: AUROC %.3f  (%d rows, %d test)" %
          (res.matched.auroc, res.matched.n_rows, res.matched.n_test))
    print("oracle drop : %+.3f" % res.oracle_drop)
    print("predicted detector drop: %+.3f   (interval %+.3f .. %+.3f)" %
          (res.predicted_detector_drop, res.interval[0], res.interval[1]))
    if res.extrapolating:
        print()
        print("WARNING: the oracle drop %.3f lies outside the range seen in the eleven" % res.oracle_drop)
        print("         fitted benchmarks (0.000 to 0.135). The regression is extrapolating;")
        print("         in the prospective test its absolute error roughly doubled there.")
    print()
    print("Note: the regression is fitted on eleven benchmarks. It ranks benchmarks")
    print("      reasonably but its absolute error on unseen benchmarks was 0.038 in")
    print("      mean absolute terms; treat the interval, not the point, as the answer.")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            f.write(res.to_json())
        print("written: %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
