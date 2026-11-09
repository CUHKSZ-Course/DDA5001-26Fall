# Personal GPU setup

Open the p1 folder in VS Code and select **Terminal → New Terminal**. On Linux or WSL, install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

    bash setup/setup-local.sh cuda
    source .venv/bin/activate

The first command installs the Python environment; the second selects it. Use this terminal for the project commands.

## Smaller GPUs

The default model uses about 20 GiB of GPU memory. Copy config/train_shakespeare_char.py to config/personal.py before changing the hardware settings. For example:

    batch_size = 8
    gradient_accumulation_steps = 9
    eval_iters = 72

This keeps the training batch and evaluation size unchanged. If your system cannot run torch.compile, set compile = False.

Use this adapted file as the starting point for each optimizer's experiments. Set optimizer_name, its learning rate and out_dir, then run one experiment with:

    python train.py config/personal.py

Use the same hardware settings for all six runs and report your changes. Allow at least 37 GiB of free disk space for the six runs: best/last checkpoints use about 31.6 GiB, and saving a checkpoint temporarily requires up to another 3.4 GiB. Keep additional space for your own files.
