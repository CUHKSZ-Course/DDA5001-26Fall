"""Run upstream TRL teacher-top-128 on-policy distillation."""

import argparse
from collections import Counter
import hashlib
import json
from math import isfinite
from pathlib import Path
import time

from datasets import Dataset
from peft import LoraConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from trl.experimental.distillation import DistillationConfig, DistillationTrainer


class FiniteTraining(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        for key in ("loss", "on_policy_loss", "off_policy_loss", "grad_norm"):
            if key in (logs or {}) and not isfinite(logs[key]):
                raise FloatingPointError(f"Non-finite {key} at step {state.global_step}")


def collate_fixed_prefix(features):
    """Supply exact teacher token IDs to TRL's documented collator interface."""
    (row,) = features
    prompt, answer = row["prompt_ids"], row["response_ids"]
    return {
        "input_ids": torch.tensor([prompt + answer]),
        "attention_mask": torch.ones((1, len(prompt) + len(answer)), dtype=torch.long),
        "labels": torch.tensor([[-100] * len(prompt) + answer]),
        "prompts": torch.tensor([prompt]),
        "prompt_attention_mask": torch.ones((1, len(prompt)), dtype=torch.long),
    }


def main():
    parser = argparse.ArgumentParser()
    for name in ("panel", "student", "teacher", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--cap", type=int, default=3072)
    parser.add_argument("--prompt-cap", type=int, default=768)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--fixed-teacher-prefixes", action="store_true")
    parser.add_argument("--save-rollouts", action="store_true", help="write student responses to rollouts.jsonl")
    args = parser.parse_args()
    if args.save_rollouts and args.fixed_teacher_prefixes:
        parser.error("--save-rollouts requires student-generated prefixes")
    args.out.mkdir(parents=True, exist_ok=False)
    rollout_path = args.out / "rollouts.jsonl" if args.save_rollouts else None
    panel = args.panel.read_bytes()
    tok = AutoTokenizer.from_pretrained(args.student)
    tok.chat_template = "{% set enable_thinking = false %}" + tok.chat_template
    examples = []
    prompts = {}
    for row in map(json.loads, panel.splitlines()):
        text = row.get("student_text") or (
            row["question"] + "\n" + r"Show your work and put the final answer in \boxed{}."
        )
        messages = [{"role": "user", "content": text}]
        tokens = tuple(tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True
        )["input_ids"])
        if len(tokens) > args.prompt_cap or tokens in prompts:
            raise ValueError(f"Expected unique prompts of at most {args.prompt_cap} tokens")
        prompts[tokens] = row["id"]
        if args.fixed_teacher_prefixes:
            target = row["response_ids"]
            if not 0 < len(target) <= args.cap or target[-1] != tok.eos_token_id:
                raise ValueError(f"Invalid fixed demonstration for {row['id']}")
            examples.append({"prompt_ids": list(tokens), "response_ids": target})
        else:
            examples.append({"messages": messages})
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=torch.bfloat16)
    endings = Counter()
    rollouts = []
    rollout_example = None
    stop_ids = teacher.generation_config.eos_token_id
    stop_ids = {tok.eos_token_id, *(stop_ids if isinstance(stop_ids, list) else [stop_ids])}

    # Observe the exact sequences reaching the frozen teacher.
    def observe(_module, _args, kwargs):
        nonlocal rollout_example
        for ids, mask in zip(kwargs["input_ids"], kwargs["attention_mask"], strict=True):
            sequence = ids[mask.bool()].tolist()
            matches = [(question, len(prompt)) for prompt, question in prompts.items()
                       if tuple(sequence[:len(prompt)]) == prompt]
            if len(matches) != 1:
                raise ValueError("Teacher input does not start with an identified panel prompt")
            question, prompt_length = matches[0]
            end = "stopped" if sequence[-1] in stop_ids else "unfinished"
            endings[end] += 1
            rollouts.append({
                "id": question, "response_tokens": len(sequence) - prompt_length, "finish": end
            })
            if not args.fixed_teacher_prefixes and (rollout_example is None or rollout_path is not None):
                example = {
                    **rollouts[-1],
                    "rollout_index": len(rollouts),
                    "response": tok.decode(
                        sequence[prompt_length:], skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    ),
                }
                if rollout_example is None:
                    rollout_example = example
                if rollout_path is not None:
                    with rollout_path.open("a", encoding="utf-8") as output:
                        print(json.dumps(example, ensure_ascii=False), file=output)

    hook = teacher.register_forward_pre_hook(observe, with_kwargs=True)
    set_seed(args.seed)
    config = DistillationConfig(
        output_dir=str(args.out),
        model_init_kwargs={"dtype": "bfloat16"},
        max_steps=args.steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_length=args.prompt_cap + args.cap,
        max_prompt_length=args.prompt_cap,
        max_completion_length=args.cap,
        lmbda=0.0 if args.fixed_teacher_prefixes else 1.0,
        beta=0.0,
        loss_top_k=128,
        loss_add_tail=False,
        temperature=1.0,
        top_p=0.8,
        top_k=20,
        bf16=True,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=min(16, args.steps // 10),
        gradient_checkpointing=not args.no_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        logging_nan_inf_filter=False,
        save_strategy="no",
        report_to="none",
        log_completions=not args.fixed_teacher_prefixes,
        log_completions_steps=min(20, args.steps),
        num_completions_to_print=1,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = DistillationTrainer(
        model=str(args.student),
        teacher_model=teacher,
        args=config,
        processing_class=tok,
        data_collator=collate_fixed_prefix if args.fixed_teacher_prefixes else None,
        train_dataset=Dataset.from_list(examples),
        callbacks=[FiniteTraining],
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    # Prompt-only raw batches contain no completion labels before rollout.
    # With one response per update, use the native post-rollout batch mean.
    trainer.model_accepts_loss_kwargs = False
    if config.gradient_checkpointing:
        trainer.model.enable_input_require_grads()
    start = time.monotonic()
    try:
        trainer.train()
    finally:
        hook.remove()
        (args.out / "summary.json").write_text(json.dumps({
            "steps": trainer.state.global_step,
            "seed": args.seed,
            "prefix_source": "teacher" if args.fixed_teacher_prefixes else "student",
            "training_seconds": time.monotonic() - start,
            "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
            "endings": dict(endings),
            "rollouts": rollouts,
            "rollout_example": rollout_example,
            "panel_sha256": hashlib.sha256(panel).hexdigest(),
            "history": trainer.state.log_history,
        }, indent=2))
    if args.require_complete and endings["unfinished"]:
        raise RuntimeError("Unfinished rollouts reached the native loss. No adapter saved.")
    trainer.save_model(str(args.out / "adapter"))


if __name__ == "__main__":
    main()
