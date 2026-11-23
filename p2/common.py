import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

STUDENT = "Qwen/Qwen3-1.7B"
TEACHER = "Qwen/Qwen3-4B-Instruct-2507"
REVISIONS = {
    STUDENT: "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    TEACHER: "cdbee75f17c01a7cc42f958dc650907174af0554",
}
MATH_HINT = r"Show your work and put the final answer in \boxed{}."
QWEN_NONTHINKING = dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0)


def location(name):
    key = "P2_TEACHER" if name == TEACHER else "P2_STUDENT"
    path = os.environ.get(key)
    return (path, {}) if path else (name, {"revision": REVISIONS[name]})


def tokenizer(name):
    path, options = location(name)
    if name == STUDENT and os.environ.get("P2_STUDENT_TOKENIZER"):
        return AutoTokenizer.from_pretrained(os.environ["P2_STUDENT_TOKENIZER"])
    return AutoTokenizer.from_pretrained(path, **options)


def model(name):
    path, options = location(name)
    return AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, device_map={"": 0}, **options
    ).eval()


def prompt_ids(tok, text):
    messages = [{"role": "user", "content": text}]
    return tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_tensors="pt", return_dict=False,
    )


def records(path):
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]
