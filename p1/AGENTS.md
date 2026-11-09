# Part I exercise completion contract

Complete only the seven marked TODO blocks in `model.py`, `train.py` and `plot.py`.
The provided scaffold deliberately fixes the API and control flow. Replace each
`todo("...")` call using its adjacent instructions; preserve the surrounding code.

- Use native `torch.optim.AdamW`, `torch.optim.SGD` and `torch.optim.Muon` only.
  Do not write an optimizer, use a custom optimizer module/subclass, substitute
  another method, or implement Newton-Schulz. PyTorch >=2.9 is required.
- Preserve all supplied parameter groups, LR/betas/momentum/Nesterov arguments,
  fused selection, Muon `adjust_lr_fn='original'`, and `[auxiliary, muon]` order.
  Do not hard-code or additionally scale the supplied hyperparameters.
- Each provided optimizer steps exactly once per completed accumulated update.
  The loop is supplied. Do not add backward, zero_grad, clipping or closures to
  `step_optimizers`; these are already handled by the training loop.
- Shift sequence positions by one, not token IDs. The causal mask includes self
  and past positions and excludes every future position.
- Plot the supplied xs/ys with the supplied labels/styles. Evaluation aggregation,
  axis labels, legends, file saving and experiment management are provided.
- `evaluate.py` is complete: loss is total NLL divided by total prediction tokens.
  Do not replace it with an average of sequence losses or change the evaluator,
  fixed model/data/token budget, scheduler, initialization or checkpoint code.

For experiments, copies of the config may change optimizer_name, its learning-rate
field and out_dir as instructed by the README. Use only the documented hardware
adaptations when working on a smaller GPU.

Run `python check_solution.py`, then a CPU smoke run in a fresh output directory.
Fix the indicated contract violation rather than weakening the checks. You may
write separate analysis scripts as permitted by the project handbook.
