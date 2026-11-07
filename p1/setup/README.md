# Personal GPU setup

Run these commands from p1/ on Linux or WSL. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/) first.

    bash setup/setup-local.sh cuda
    source .venv/bin/activate

Use this environment's python for the project. CPU checks are available with
bash setup/setup-local.sh cpu, but formal training needs a GPU.

## Smaller GPUs

Keep effective batch size 72 by reducing batch_size and increasing
gradient_accumulation_steps. For example:

    batch_size = 8
    gradient_accumulation_steps = 9
    eval_iters = 72

This also preserves 576 evaluation windows per split. If necessary, use a smaller
divisor of 72 and adjust accumulation and evaluation accordingly.

Save the adapted configuration as a separate file and pass it to the runner:

    python run_experiments.py --config=config/personal.py --output=runs/sweep01

If compilation is unsupported, set compile = False. Use the same settings for
all six runs and report the adaptations. Keep the model, seed and token budget
unless the TA approves a change. Runtime depends on your GPU.

If the model still does not fit, or the program reports a kernel/storage error,
contact the TA.
