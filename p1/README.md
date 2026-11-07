# Part I: GPT Pre-training

Train a GPT from scratch on Shakespeare. Complete the TODOs, then compare **AdamW, Muon, and SGDM**, with two learning rates each.

## Setup

Run the commands from p1/.

- **Classroom:** use lab-python instead of python. For the preinstalled copy, run lab-init, then cd ~/work/project1.
- **Your own GPU:** follow [setup/README.md](setup/README.md).

## Complete the code

Fill the TODOs in:

| File | Task |
| --- | --- |
| model.py | Causal mask and optimizer calls |
| train.py | Shifted labels and parameter updates |
| evaluate.py | Evaluation scores |
| plot.py | Loss curves |

    python check_solution.py
    python train.py config/smoke.py --device=cpu --out_dir=runs/cpu-check

## Run the experiments

Use the supplied [configuration](config/train_shakespeare_char.py):

    python run_experiments.py --output=runs/sweep01

This runs **3 optimizers × 2 learning rates = 6 experiments**. Keep other settings fixed and use a fresh output directory. Allow about 32 GiB for checkpoints.

## Results

Each run saves CSV metrics and logs for your plots and analysis. Use last.pt for endpoint results. For example:

    python plot.py --run-dir=runs/sweep01/00-adamw-tokens-lr0.3
    python sample.py --checkpoint=runs/sweep01/00-adamw-tokens-lr0.3/last.pt --tokens=400
    python plot.py --sweep-dir=runs/sweep01
    python export_results.py --sweep-dir=runs/sweep01 --output=results01.zip

Plot all six runs and generate one text sample. Discuss learning-rate effects and any crossover in these runs.

Add your report PDF to the exported ZIP. Keep checkpoints locally.
