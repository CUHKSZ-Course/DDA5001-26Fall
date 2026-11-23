"""Create a portable copy of the Part II analysis inputs and code."""

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

CODE = ("analysis_exercises.py", "analyze_results.py", "check_analysis.py", "pass_rates.py",
        "inspect_trajectory.py", "data/MATH-LICENSE", "data/LICENSE")
RUN_FILES = (
    "data/math-panel.jsonl", "sft/summary.json", "opd/summary.json", "opd/rollouts.jsonl",
    "timing.tsv", "elapsed-seconds.txt", "scores/summary.csv",
    *(f"scores/{arm}-{task}.jsonl"
      for arm in ("original", "teacher", "sft", "opd") for task in ("math", "ifeval")),
)
INSTRUCTIONS = """# Part II offline analysis

From the extracted folder, use Python 3 to run the analysis. On Windows,
use `python` in the commands below if that is your Python 3 command.

```bash
python3 check_analysis.py
python3 analyze_results.py booking-run/scores > analysis.json
python3 inspect_trajectory.py training
python3 inspect_trajectory.py math QUESTION_ID --draw 0
python3 inspect_trajectory.py ifeval PROMPT_ID --draw 0
```

Complete the two TODOs in `analysis_exercises.py` before running the checks.
The first training trajectory is displayed by default. Give its question ID
after `training` to inspect another. The `math` and `ifeval` viewers take a
question or prompt ID from the score files and a draw number from 0 to 3.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("booking-run"))
    parser.add_argument("--output", type=Path, default=Path("part2-analysis.zip"))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} already exists. Remove it or pass --output for a new archive.")
    starter = Path(__file__).resolve().parent
    files = {name: starter / name for name in CODE}
    files.update({f"booking-run/{name}": args.run / name for name in RUN_FILES})
    for path in files.values():
        if not path.is_file():
            raise SystemExit(f"Missing analysis input {path}. Complete the workstation run before exporting.")
    with ZipFile(args.output, "x", ZIP_DEFLATED) as archive:
        archive.writestr("README.md", INSTRUCTIONS)
        for name, path in files.items():
            archive.write(path, name)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
