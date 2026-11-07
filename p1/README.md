# Part I: GPT Pre-training

Train a GPT on Shakespeare and compare **AdamW, Muon, and SGDM**. For each optimizer, run the provided default learning rate and one other rate you choose.

## 1. Open the project

Download and extract this repository. In the file manager, open the p1 folder, right-click an empty area, and select **Open in Terminal**.

On a classroom machine, enter these commands one at a time:

    source /opt/dda5001/venv/bin/activate
    code .

The first command selects the installed Python environment. The second opens the project in VS Code. Keep this terminal open to run the commands below.

On your own computer, follow [setup/README.md](setup/README.md) to prepare Python first.

## 2. Complete the code

Fill the TODOs in model.py, train.py, evaluate.py and plot.py. Then run:

    python check_solution.py
    python train.py config/smoke.py --device=cpu --out_dir=checks/cpu

The second command is a small test of your code.

## 3. Run the default experiment

Open [config/train_shakespeare_char.py](config/train_shakespeare_char.py) in VS Code. It selects one optimizer and gives one default learning rate for each:

| optimizer_name | Learning-rate field | Default |
| --- | --- | --- |
| adamw | learning_rate | 0.0003 |
| muon | muon_learning_rate | 0.01 |
| sgdm | sgdm_learning_rate | 0.1 |

Run the unchanged configuration:

    python train.py config/train_shakespeare_char.py

This runs **one AdamW experiment** and saves its results in runs/adamw-default. Wait for it to finish before starting the next run.

## 4. Choose another learning rate

Copy the original configuration in VS Code and save it as `config/adamw-other.py`. In the copy, change `learning_rate` to your chosen value and set `out_dir = 'runs/adamw-other'`. Then run:

    python train.py config/adamw-other.py

If loss decreases slowly and smoothly, try a larger learning rate. If it fluctuates strongly or rises, try a smaller one. Halving or doubling the default is a starting point.

For Muon and SGDM, start from fresh copies of the original configuration. Set `optimizer_name`, edit the corresponding learning-rate field, and give each run its own `out_dir`. Keep the other settings fixed.

You will have **six separate runs**: a default and a chosen learning rate for each optimizer.

## 5. Plot and submit

For one run:

    python plot.py --run-dir=runs/adamw-default
    python sample.py --checkpoint=runs/adamw-default/last.pt --tokens=400

Make individual plots for all six runs and generate one representative text sample. Once they are complete, compare and export the results:

    python plot.py --runs-dir=runs
    python export_results.py --runs-dir=runs --output=results01.zip

Discuss the learning-rate effects and any crossover visible in these curves. Add your report PDF to the exported ZIP. Keep checkpoints locally.
