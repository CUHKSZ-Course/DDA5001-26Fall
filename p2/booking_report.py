"""Validate matched prompts/draws across all eight score files and write summary.csv."""

import argparse
import csv
import json
from pathlib import Path

from pass_rates import summarize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scores", type=Path)
    p.add_argument(
        "--assets",
        type=Path,
        default=Path("assets"),
        help="directory containing math.jsonl and ifeval.jsonl",
    )
    args = p.parse_args()
    panel = {}
    for task, key, size in (("math", "unique_id", 500), ("ifeval", "key", 541)):
        with (args.assets / f"{task}.jsonl").open(encoding="utf-8") as source:
            panel[task] = {json.loads(line)[key] for line in source}
        if len(panel[task]) != size:
            raise ValueError(f"Expected {size} distinct benchmark IDs for {task}")
    report = []
    for task in ("math", "ifeval"):
        base = None
        for arm in ("original", "teacher", "sft", "opd"):
            name = f"{arm}-{task}.jsonl"
            dest = args.scores / name
            with dest.open(encoding="utf-8") as source:
                rows = [json.loads(line) for line in source]
            if any(r["benchmark"] != task or r["variant"] != arm for r in rows):
                raise ValueError(f"Mislabeled rows in {name}")
            stats = summarize(rows, len(panel[task]))
            if {r["id"] for r in rows} != panel[task]:
                raise ValueError(f"Question IDs differ in {name}")
            paired = {
                (r["id"], r["draw"]): (r["prompt"], r["seed"], r["sampling"], r["budget"])
                for r in rows
            }
            if base is None:
                base = paired
            if paired != base:
                raise ValueError(f"Evaluation protocol differs in {name}")
            report.append(
                dict(
                    variant=arm,
                    benchmark=task,
                    questions=stats["questions"],
                    **{f"pass@{k}": stats[f"pass@{k}"] for k in (1, 2, 4)},
                )
            )
    with (args.scores / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=report[0].keys())
        w.writeheader()
        w.writerows(report)


if __name__ == "__main__":
    main()
