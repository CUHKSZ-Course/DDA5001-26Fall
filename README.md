# DDA5001 — 2026 Fall Final Project

Student starter code for the DDA5001 final project at The Chinese University of
Hong Kong, Shenzhen. Follow the project handbook and Blackboard announcements for
deadlines, experiment settings, and submission requirements.

## Project parts

| Part | Topic | Materials |
| --- | --- | --- |
| I | Pre-train a character-level GPT; compare optimizers and learning rates | [Instructions and code](p1/README.md) |
| II | To be announced | Added when released |
| III | To be announced | Added when released |

Each part has its own directory and README. Only Part I is included currently.

```text
.
├── README.md
├── LICENSE
└── p1/
    ├── README.md              # Part I instructions
    ├── train.py               # training and experiment budgets
    ├── model.py               # GPT and optimizer construction
    ├── evaluate.py            # language-model scores
    ├── plot.py                # individual and comparison curves
    ├── sample.py              # generate text from your checkpoint
    ├── run_experiments.py     # the six-run experiment menu
    ├── export_results.py      # submission without large checkpoints
    ├── check_solution.py      # small correctness checks
    ├── checkpoint_io.py       # supplied checkpoint helpers
    ├── configurator.py        # configuration loading
    ├── requirements.txt
    ├── RELEASE.json           # starter-release metadata
    ├── config/                # course and smoke-test settings
    ├── data/shakespeare_char/ # bundled corpus and prepared token data
    └── setup/                 # personal environment setup, Linux/WSL
```

Future `p2/` and `p3/` directories will follow the same per-part layout.

## Start Part I

From the repository root:

```bash
cd p1
```

Complete the marked TODOs before running the experiments. There are five exercise
categories and eight short TODO blocks. The optimizer implementations and checkpoint
helpers are provided; unfinished TODOs intentionally raise `NotImplementedError`.

**Classroom workstation:** use the preinstalled environment. When working with
this cloned repository, run commands from `p1/` with `lab-python`:

```bash
lab-python check_solution.py
lab-python train.py config/smoke.py --device=cpu --out_dir=runs/cpu-check
```

If using the workstation's preinstalled starter instead of this clone, follow
`lab-init` and the directory instructions in the [Part I guide](p1/README.md).

**Your own GPU:** install [uv](https://docs.astral.sh/uv/getting-started/installation/),
then create a personal environment from `p1/` on Linux or WSL:

```bash
bash setup/setup-local.sh cuda
source .venv/bin/activate
python check_solution.py
```

See the [Part I guide](p1/README.md) for the six required experiments, smaller-GPU
adaptations, logging, plotting, sampling, and submission. Run outputs stay under
`p1/runs/` and are excluded from Git. Keep login credentials and API keys private.

## Acknowledgements and license

Part I builds on [nanoGPT](https://github.com/karpathy/nanoGPT) and the
[DDA5001 2025 course code](https://github.com/CUHKSZ-Course/DDA5001-25Fall).
Optimizers use PyTorch's built-in implementations, including Muon.
The original nanoGPT MIT license is retained in [LICENSE](LICENSE).
