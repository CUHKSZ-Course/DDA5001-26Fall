"""Run stock completion-only TRL SFT on a preverified schema panel."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import set_seed
from trl import SFTConfig, SFTTrainer

from common import MATH_HINT, STUDENT, model, records, tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("panel", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--save-steps", type=int, default=32)
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--cap", type=int, default=3072)
    a = p.parse_args()
    tok = tokenizer(STUDENT)
    panel = records(a.panel)
    assert len({r["id"] for r in panel}) == len(panel)
    examples = []
    for row in panel:
        text = row.get("student_text") or (row["question"] + "\n" + MATH_HINT)
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=a.thinking,
        )
        target = row["response_ids"]
        assert 0 < len(target) <= a.cap and target[-1] == tok.eos_token_id
        examples.append(
            dict(input_ids=prompt + target, completion_mask=[0] * len(prompt) + [1] * len(target))
        )
    a.out.mkdir(parents=True, exist_ok=False)
    set_seed(a.seed)
    cfg = SFTConfig(
        output_dir=str(a.out),
        max_steps=a.steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=a.batch,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_steps=min(16, a.steps // 10),
        max_grad_norm=1,
        seed=a.seed,
        data_seed=a.seed,
        report_to="none",
        save_strategy="steps",
        save_steps=a.save_steps,
        save_only_model=True,
        logging_steps=8,
        max_length=None,
        completion_only_loss=True,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    tr = SFTTrainer(
        model=model(STUDENT),
        args=cfg,
        processing_class=tok,
        train_dataset=Dataset.from_list(examples),
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    begin = time.monotonic()
    result = tr.train()
    tr.save_model(str(a.out / "adapter"))
    assert tr.state.global_step == a.steps
    (a.out / "summary.json").write_text(
        json.dumps(
            dict(
                questions=len(panel),
                panel_sha256=hashlib.sha256(a.panel.read_bytes()).hexdigest(),
                steps=a.steps,
                lr=a.lr,
                batch=a.batch,
                seed=a.seed,
                thinking=a.thinking,
                peak_cuda_gib=torch.cuda.max_memory_allocated() / 2**30,
                wall_seconds=time.monotonic() - begin,
                trainer_metrics=result.metrics,
                history=tr.state.log_history,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
