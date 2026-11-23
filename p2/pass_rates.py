"""Aggregate four stochastic responses per question with the unbiased pass@k estimator."""

import argparse
from collections import Counter
import json
from math import comb
from pathlib import Path


def pass_rates(correct, n=4):
    if n < 4 or not 0 <= correct <= n:
        raise ValueError("Need at least four draws and a valid number of successes")
    # comb(n-c, k) is zero when fewer than k incorrect draws exist.
    return {f"pass@{k}": 1 - comb(n - correct, k) / comb(n, k) for k in (1, 2, 4)}


def summarize(rows, expected=None):
    groups = {}
    run = None
    for r in rows:
        signature = (r["variant"], r["benchmark"], r["budget"], r["sampling"])
        if run is None:
            run = signature
        if run != signature or type(r["correct"]) is not bool:
            raise ValueError("Mixed runs or invalid correctness label")
        group = groups.setdefault(r["id"], {})
        if r["draw"] in group:
            raise ValueError(f"Duplicate draw for {r['id']}")
        group[r["draw"]] = r
    if not groups or (expected is not None and len(groups) != expected):
        raise ValueError(f"Unexpected number of questions: {len(groups)}")
    hist = Counter()
    for key, draws in groups.items():
        if (
            set(draws) != set(range(4))
            or len({r["seed"] for r in draws.values()}) != 4
            or len({r["prompt"] for r in draws.values()}) != 1
        ):
            raise ValueError(f"Incomplete or mismatched draws for {key}")
        hist[sum(r["correct"] for r in draws.values())] += 1
    scores = {
        metric: sum(n * pass_rates(c)[metric] for c, n in hist.items()) / len(groups)
        for metric in ("pass@1", "pass@2", "pass@4")
    }
    return {
        "questions": len(groups),
        "n": 4,
        "correct_draw_histogram": dict(sorted(hist.items())),
        **scores,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scores", type=Path)
    parser.add_argument("--expected", type=int, help="number of questions in the evaluated split")
    args = parser.parse_args()
    with args.scores.open(encoding="utf8") as source:
        print(json.dumps(summarize(map(json.loads, source), args.expected), indent=2))
