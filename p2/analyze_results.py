"""Prepare matched benchmark outcomes for the two report exercises."""

import argparse
import json
from pathlib import Path

from analysis_exercises import ifeval_transitions, math_by_level
from pass_rates import summarize

ARMS = ("original", "teacher", "sft", "opd")
PROTOCOL = ("prompt", "seed", "sampling", "budget", "level")


def read_questions(scores, benchmark, expected):
    by_arm = {}
    for arm in ARMS:
        path = scores / f"{arm}-{benchmark}.jsonl"
        with path.open(encoding="utf-8") as source:
            rows = [json.loads(line) for line in source]
        if any(r["benchmark"] != benchmark or r["variant"] != arm for r in rows):
            raise ValueError(f"Mislabeled results in {path}")
        summarize(rows, expected)
        by_draw = {(r["id"], r["draw"]): r for r in rows}
        if by_arm:
            reference = by_arm["original"]
            if by_draw.keys() != reference.keys():
                raise ValueError(f"Question IDs or draws differ in {path}")
            if any(any(row.get(key) != reference[draw].get(key) for key in PROTOCOL)
                   for draw, row in by_draw.items()):
                raise ValueError(f"Evaluation conditions differ in {path}")
        by_arm[arm] = by_draw

    questions = []
    for (question_id, draw), row in by_arm["original"].items():
        if draw == 0:
            questions.append({
                "id": question_id,
                "level": row.get("level"),
                "correct_draws": {
                    arm: sum(by_arm[arm][question_id, i]["correct"] for i in range(4))
                    for arm in ARMS
                },
            })
    return questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scores", type=Path)
    args = parser.parse_args()
    math = read_questions(args.scores, "math", 500)
    instructions = read_questions(args.scores, "ifeval", 541)
    print(json.dumps({
        "math_by_level": math_by_level(math),
        "ifeval_transitions": ifeval_transitions(instructions),
    }, indent=2))


if __name__ == "__main__":
    main()
