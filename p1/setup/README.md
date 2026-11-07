# Personal GPU setup

Run these commands from p1/ on Linux or WSL. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/) first.

    bash setup/setup-local.sh cuda
    source .venv/bin/activate

Use this environment's python for the project.

## Smaller GPUs

The default configuration uses about 20 GiB of GPU memory. To reduce memory usage,
copy the configuration first:

    cp config/train_shakespeare_char.py config/personal.py

Then edit the copy, keeping effective batch size 72. For example:

    batch_size = 8
    gradient_accumulation_steps = 9
    eval_iters = 72

This preserves the training batch and evaluation size. Run with your copy:

    python run_experiments.py --config=config/personal.py --output=runs/sweep01

If your system cannot run torch.compile, set compile = False. Use the same
configuration for all six runs and report your changes.
