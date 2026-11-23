import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
from pathlib import Path

from transformers import set_seed

from common import (MATH_HINT, QWEN_NONTHINKING, STUDENT, TEACHER, location, prompt_ids, records,
                    tokenizer)


def math_score(row, response):
    from math_verify import LatexExtractionConfig, parse, verify
    gold = parse("$" + row["answer"] + "$")
    start = response.rfind(r"\boxed{")
    end = start + len(r"\boxed{")
    depth = 1
    if start >= 0:
        while depth and end < len(response):
            depth += (response[end] == "{") - (response[end] == "}")
            end += 1
    boxed = start >= 0 and depth == 0
    final = "$" + response[start:end] + "$" if boxed else ""
    predicted = parse(final, extraction_config=[LatexExtractionConfig(
        boxed_match_priority=0, try_extract_without_anchor=False)], fallback_mode="no_fallback")
    return {"correct": bool(boxed and predicted and gold and verify(gold, predicted)),
            "parsed": str(predicted), "boxed": boxed}


def main(merged_dir_factory=tempfile.TemporaryDirectory):
    p = argparse.ArgumentParser()
    p.add_argument("variant", choices=["original", "teacher", "sft", "opd"])
    p.add_argument("benchmark", choices=["math", "ifeval"])
    p.add_argument("--adapter")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tokens", type=int, help="default: 4096 for math, 3072 for IFEval")
    p.add_argument("--limit", type=int)
    p.add_argument("--panel", type=Path, help="JSON with the exact question IDs for each benchmark")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--samples", type=int, choices=[1, 4], default=1,
                   help="one greedy completion or four Qwen non-thinking model-card draws")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float)
    p.add_argument("--thinking", action="store_true",
                   help="Use the model's native thinking template for this experiment")
    p.add_argument("--top-p", type=float)
    p.add_argument("--top-k", type=int)
    p.add_argument("--min-p", type=float)
    args = p.parse_args()
    overrides = {name: getattr(args, name) for name in QWEN_NONTHINKING
                 if getattr(args, name) is not None}
    if overrides and args.samples != 4:
        p.error("sampling parameter overrides require --samples 4")
    policy = dict(QWEN_NONTHINKING, presence_penalty=0.0, frequency_penalty=0.0,
                  repetition_penalty=1.0)
    if args.thinking:
        policy.update(temperature=0.6, top_p=0.95)
    policy.update(overrides)
    if (not 0 < policy["temperature"] < float("inf") or not 0 < policy["top_p"] <= 1
            or not 0 <= policy["min_p"] <= 1 or not (policy["top_k"] == -1 or policy["top_k"] >= 1)):
        p.error("use positive temperature, top-p in (0,1], min-p in [0,1], and top-k=-1 or >=1")
    if args.tokens is None:
        args.tokens = 4096 if args.benchmark == "math" else 3072
    if bool(args.adapter) != (args.variant in {"sft", "opd"}):
        p.error("sft and opd each require their own --adapter")
    if min(args.tokens, args.batch) < 1 or (args.limit is not None and args.limit < 1):
        p.error("limits must be positive")
    if args.out.exists() or args.out.with_suffix(".partial").exists():
        p.error("output or partial file already exists; choose a fresh path")
    set_seed(args.seed)
    name = TEACHER if args.variant == "teacher" else STUDENT
    tok = tokenizer(name)
    if args.adapter:
        # Start a clean engine worker after PEFT's threaded CPU merge.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM, SamplingParams
    path, opts = location(name)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Merging once per evaluation lets all four variants use the same dense kernels.
    # Keep the transient model on the run filesystem until decoding completes.
    merged = None
    if args.adapter:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        merged = merged_dir_factory(prefix="eval-weights-", dir=args.out.parent)
        base = AutoModelForCausalLM.from_pretrained(path, **opts, dtype=torch.bfloat16,
                                                   device_map=None)
        weights = PeftModel.from_pretrained(base, args.adapter, torch_device="cpu").merge_and_unload(safe_merge=True)
        if any(p.device.type != "cpu" for p in weights.parameters()):
            raise RuntimeError("Adapter merge must finish on CPU before the inference engine starts")
        weights.save_pretrained(merged.name)
        tok.save_pretrained(merged.name)
        del weights, base
        path, opts = merged.name, {}
    net = LLM(model=path, **opts, dtype="bfloat16", max_model_len=8192,
              max_num_seqs=args.batch, enforce_eager=True, gpu_memory_utilization=0.88)
    source = records("assets/" + args.benchmark + ".jsonl")
    indices = list(range(len(source)))
    if args.panel:
        keys = json.loads(args.panel.read_text(encoding="utf8"))[args.benchmark]
        field = "unique_id" if args.benchmark == "math" else "key"
        wanted = set(keys)
        indices = [i for i,r in enumerate(source) if r[field] in wanted]
        if not keys or len(wanted) != len(keys) or len(indices) != len(keys):
            p.error("panel contains missing or duplicate benchmark IDs")
    indices = indices[:args.limit]
    data = [source[i] for i in indices]
    if args.benchmark == "ifeval":
        sys.path.insert(0, str(Path("assets").resolve()))
        from instruction_following_eval import evaluation_lib as grader
        import langdetect
        import nltk
        nltk.data.path.insert(0, str(Path("assets/nltk").resolve()))
        langdetect.DetectorFactory.seed = 0
    partial = args.out.with_suffix(".partial")
    with partial.open("x", encoding="utf8") as out:
        math = args.benchmark == "math"
        texts = [r["problem"] + "\n" + MATH_HINT if math else r["prompt"] for r in data]
        if args.thinking:
            prompts = [{"prompt_token_ids": tok.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=True,
                add_generation_prompt=True, enable_thinking=True, return_dict=False)} for text in texts]
        else:
            prompts = [{"prompt_token_ids": prompt_ids(tok, text)[0].tolist()} for text in texts]
        params = SamplingParams(temperature=0, max_tokens=args.tokens)
        for start in range(0, len(data), 128):
            batch = data[start:start + 128]
            batch_prompts = prompts[start:start + 128]
            for draw in range(args.samples):
                seeds = [int.from_bytes(hashlib.sha256(
                    f"{args.seed}/{args.benchmark}/{indices[start+i]}/{draw}".encode()
                ).digest()[:4], "big") for i in range(len(batch))]
                settings = ([SamplingParams(**policy, seed=seed, max_tokens=args.tokens)
                             for seed in seeds] if args.samples == 4 else params)
                outputs = net.generate(batch_prompts, settings)
                for i, (row, text, output) in enumerate(zip(batch, texts[start:start+128], outputs, strict=True)):
                    answer = output.outputs[0]
                    response, n = answer.text, len(answer.token_ids)
                    if math:
                        details = dict(level=row["level"], answer=row["answer"], **math_score(row, response))
                    else:
                        inp = grader.InputExample(**row)
                        if args.samples == 4:
                            random.seed(str(row["key"]))
                        score = grader.test_instruction_following_strict(inp, {text: response})
                        details = dict(correct=score.follow_all_instructions,
                                       instruction_pass=score.follow_instruction_list,
                                       instruction_ids=row["instruction_id_list"])
                    record = dict(id=row["unique_id"] if math else row["key"], prompt=text,
                                  response=response, variant=args.variant, benchmark=args.benchmark,
                                  response_tokens=n, budget=args.tokens, finish_reason=answer.finish_reason, **details)
                    if args.samples == 4:
                        record.update(draw=draw, seed=seeds[i], sampling=policy)
                    if args.thinking:
                        record["thinking"] = True
                    print(json.dumps(record, ensure_ascii=False), file=out, flush=True)
    os.replace(partial, args.out)
    if merged is not None:
        merged.cleanup()


if __name__ == "__main__":
    main()
