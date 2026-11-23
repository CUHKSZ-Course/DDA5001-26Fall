"""Read matched training or test responses as text on a personal computer."""

import argparse
import json
from pathlib import Path


def find_row(path, question_id=None, draw=None):
    if not path.is_file():
        raise SystemExit(f"Missing file {path}")
    with path.open(encoding="utf-8") as source:
        for row in map(json.loads, source):
            if (question_id is None or str(row["id"]) == str(question_id)) and (
                draw is None or row.get("draw") == draw
            ):
                return row
    raise SystemExit(f"No matching question and draw in {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", choices=("training", "math", "ifeval"))
    parser.add_argument("question_id", nargs="?")
    parser.add_argument("--draw", type=int, choices=range(4), default=0)
    parser.add_argument("--run", type=Path, default=Path("booking-run"))
    args = parser.parse_args()
    if args.benchmark == "training":
        rollout = find_row(args.run / "opd/rollouts.jsonl", args.question_id)
        teacher = find_row(args.run / "data/math-panel.jsonl", rollout["id"])
        print(f"Training question {teacher['id']}\n{teacher['question']}")
        print(f"\nReference answer\n{teacher['reference']}")
        print(f"\nOffline teacher response\n{teacher['response']}")
        print(f"\nStudent rollout {rollout['rollout_index']} ({rollout['finish']})\n{rollout['response']}")
        return

    if args.question_id is None:
        parser.error(f"{args.benchmark} requires a question ID")
    scores = args.run / "scores"
    rows = {
        arm: find_row(scores / f"{arm}-{args.benchmark}.jsonl", args.question_id, args.draw)
        for arm in ("original", "sft", "opd")
    }
    original = rows["original"]
    if args.benchmark == "math":
        print(f"MATH-500 question {args.question_id}, draw {args.draw}\n{original['prompt']}")
        print(f"\nReference answer\n{original['answer']}")
        for arm, row in rows.items():
            grade = "correct" if row["correct"] else "incorrect"
            print(f"\n{arm.upper()} ({grade}, {row['finish_reason']})\n{row['response']}")
        return

    print(f"IFEval prompt {args.question_id}, draw {args.draw}\n{original['prompt']}")
    for arm, row in rows.items():
        grade = "pass" if row["correct"] else "fail"
        checks = ", ".join(
            f"{inst}={'pass' if ok else 'fail'}"
            for inst, ok in zip(row["instruction_ids"], row["instruction_pass"])
        )
        print(f"\n{arm.upper()} ({grade}, {row['finish_reason']})\nInstructions: {checks}\n{row['response']}")


if __name__ == "__main__":
    main()
