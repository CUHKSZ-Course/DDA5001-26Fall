# DDA5001 2026 — Project Part I

Train a classic character-level GPT **from random initialization** on Shakespeare.
Complete the TODOs, then compare AdamW, native PyTorch Muon + auxiliary AdamW, and SGDM.
No pretrained weights are downloaded, and no `pip install .` is needed.

## Your code

| File | Work |
| --- | --- |
| model.py | Causal mask and the three optimizer-construction calls |
| train.py | One-token-shifted labels and optimizer updates |
| evaluate.py | Token-mean NLL, perplexity and bits per character |
| plot.py | Loss curves |

There are five exercise categories and eight short TODO blocks. Unfinished blocks
raise NotImplementedError. Both attention paths use the causal mask. Optimizer
internals and checkpoint I/O are provided; do not implement Newton–Schulz yourself.

The tokenizer has 65 characters. `train.bin` has 1,003,854 tokens and `val.bin` has 111,540.
Training budgets count every prediction token processed, including repeated samples.
Random-window sampling does not mean each corpus position is visited once per pass.

## Classroom setup

Sign in as the account assigned to your group (user1, user2, etc.) and change its
initial password. Run `lab-init`, then `cd ~/work/project1`. Edit your own files in
VS Code. Prepared data is shared read-only; do not run prepare.py against it.
Use **lab-python** in place of python in all commands below. The shared environment
already contains the requirements; personal code/results stay in your home.

First complete the TODOs and run:

```bash
python check_solution.py
python train.py config/smoke.py --device=cpu --out_dir=runs/cpu-check
```

Use a fresh output directory. These are correctness checks, not the six formal runs.

## Fixed experiment configuration

| Setting | Value |
| --- | --- |
| nanoGPT preset | gpt2-medium:24 layers,16 heads,width1024 |
| Course adaptation |65-character vocabulary, context256, random initialization |
| Microbatch / accumulation |72 /1 (effective batch72) |
| Tokens / updates |92,160,000 /5000 |
| Schedule |100-step linear warmup, then cosine to0.1 of peak LR; horizon5000 |
| Evaluation |Every500 updates,576 fixed windows per split |
| Precision / compilation |Native BF16 on classroom GPUs; torch.compile enabled |
| Dropout / weight decay / clipping |0.2 /0 /1.0 |
| Training / evaluation seed |42 /1729 |

The preset selects nanoGPT's existing model dimensions, not pretrained weights.
The character vocabulary/context differ from the original GPT-2 BPE setup, so the
course model has about302M parameters. Always pass the supplied config file.

## Run the six required experiments

```bash
python run_experiments.py --protocol=tokens --multipliers=0.3,1 --output=runs/sweep01
```

This runs three optimizers ×two LR factors sequentially. The six completed classroom
trials took about142 minutes in total, including sampling and collection overhead;
this is summed per-run timing from the calibration machines, not an extra manual-time
allowance. A five-hour room slot leaves time to learn the environment and troubleshoot.
Do not run another GPU workload simultaneously: this preset uses about20GiB.

| Method | Factor0.3 peak LR | Factor1 peak LR |
| --- | --- | --- |
| AdamW |0.00009 |0.0003 |
| Muon hidden matrices |0.003 |0.01 |
| Muon's auxiliary AdamW |0.00009 |0.0003 |
| SGDM |0.03 |0.1 |

The multiplier scales each optimizer's own base LR and the entire common schedule.
Muon uses `adjust_lr_fn='original'` and momentum0.95; other parameters use auxiliary
AdamW. SGDM uses momentum0.9, no Nesterov, and no auxiliary optimizer. All methods
use weight_decay=0; do not add extra model/seed/budget searches to the required menu.

The goal is to observe learning-rate effects, convergence and generalization.
Overfitting is a valid observation. Report endpoint results at the common budget;
best.pt may come from an earlier, better validation point. Do not train longer to
force a crossover or a particular optimizer winner. These results are not a general
ranking of optimizers on all language-model tasks.

## Inspect, plot, sample and export

Each run writes config.json, environment.json, metrics.csv, summary.json, run.log,
last.pt and best.pt. CSV rows are flushed at each evaluation and include actual
instantaneous optimizer LRs. Training time excludes evaluation/checkpoint I/O;
total process time is a different measurement. First-time compilation is included
in the measured workflow.

For example:

```bash
python plot.py --run-dir=runs/sweep01/00-adamw-tokens-lr0.3
python sample.py --checkpoint=runs/sweep01/00-adamw-tokens-lr0.3/last.pt --tokens=400
python plot.py --sweep-dir=runs/sweep01
python export_results.py --sweep-dir=runs/sweep01 --output=results01.zip
```

Generate individual plots for all six run folders and representative text from one endpoint checkpoint, as required by the handbook.
The sweep plot combines existing curves without retraining. Lines connect recorded
points; shared initialization is not a crossover, and small crossings are not proof
of a universal ranking. If no crossing appears, say so.

Sampling/resume loads your own trained checkpoint and its tokenizer, not a downloaded
pretrained model. To resume an interruption, keep the same config/schedule horizon,
pass `--resume=path/to/last.pt`, and use a new output directory.

The six best/last checkpoint pairs occupy about32GiB. Before a new sweep, check
`quota -s` and leave room for temporary checkpoint writes. Replotting or resampling
does not require repeating training. The submission ZIP excludes checkpoints and
data; keep your code/config/CSV/logs/figures/text and follow the announcement on what
to retain or submit. Export work before leaving a workstation that may be reset.

## If the whole group is off campus

Use your own GPU resources and the same released code. The accompanying
`setup/setup-local.sh cuda` creates a personal uv environment with Python3.12.16
and the validated requirements. PyTorch2.9.1 provides native Muon; select a CUDA build
compatible with your host driver. Install uv first following https://docs.astral.sh/uv/getting-started/installation/ .
From the repository root, enter `p1/`. On Linux or WSL, run:

```bash
bash setup/setup-local.sh cuda
source .venv/bin/activate
```

Use that environment's python. The setup subdirectory in this repository
contains only the personal setup helper and its dependencies; it does not create
classroom accounts or require sudo. Native Windows users should use WSL for this shell helper.

On a smaller GPU, keep effective batch72 by reducing microbatch and increasing
accumulation. For example, batch8, accumulation9 and eval_iters72 preserve both
tokens/update and576 evaluation windows per split. If that still does not fit, use
a smaller divisor of72 and recompute accumulation/eval_iters. Use the same settings
for all six comparisons and report them. CPU smoke tests do not establish formal
training speed; an older GPU is not guaranteed to finish in2–3 hours.

If torch.compile is unavailable (for example, an unsupported native Windows compiler
setup), set compile=False consistently for all six runs. On your own copy, save such
hardware adaptations to a separate config and pass it with `--config=...` to the
runner. Keep the model, seed and total token budget unless the TA approves an
alternative. Record precision, hardware and configuration; do not rank groups by
absolute training time across different devices.

Do not disable kernel/storage error checks or ignore a checksum failure. Stop and
contact the TA if the system reports a host fault; repeatedly launching new jobs
does not repair a failed host.
