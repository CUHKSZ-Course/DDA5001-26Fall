# Part I: GPT Pre-training

Train a GPT from scratch on Shakespeare. Complete the TODOs, then compare
**AdamW, Muon, and SGDM**, with two learning rates each.

## Setup

Work in this repository's p1/ folder.

- **Classroom:** use lab-python instead of python below. The environment is ready.
  If using the preinstalled starter, run lab-init, then cd ~/work/project1.
- **Your own GPU:** follow [setup/README.md](setup/README.md).

## Complete the code

Fill the TODOs in:

| File | Task |
| --- | --- |
| model.py | Causal mask and optimizer calls |
| train.py | Shifted labels and parameter updates |
| evaluate.py | Evaluation scores |
| plot.py | Loss curves |

Then check your code:

    python check_solution.py
    python train.py config/smoke.py --device=cpu --out_dir=runs/cpu-check

## Run the experiments

Use the supplied [configuration](config/train_shakespeare_char.py):

    python run_experiments.py --output=runs/sweep01

This runs **3 optimizers × 2 learning rates = 6 experiments**. Keep the other
settings fixed. Use a new output directory for another sweep.

The classroom preset needs about 20 GiB of GPU memory. Its six runs produce about
32 GiB of checkpoints, so check quota -s before starting.

## Results

Each run saves its configuration, CSV metrics, summary, logs and checkpoints.
Use last.pt for endpoint results. You may write your own analysis scripts.

For example:

    python plot.py --run-dir=runs/sweep01/00-adamw-tokens-lr0.3
    python sample.py --checkpoint=runs/sweep01/00-adamw-tokens-lr0.3/last.pt --tokens=400
    python plot.py --sweep-dir=runs/sweep01
    python export_results.py --sweep-dir=runs/sweep01 --output=results01.zip

Make individual plots for all six runs and generate text from one representative
endpoint. Discuss learning-rate effects and any crossover visible in these curves;
no extra training is required.

Add your report PDF to the submission ZIP. Checkpoints and data are excluded;
keep checkpoints locally. See the handbook for the report requirements.
