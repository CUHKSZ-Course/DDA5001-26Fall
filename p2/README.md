# Part II: Off-Policy and On-Policy Knowledge Distillation

Train and compare off-policy supervised fine-tuning (SFT) and on-policy distillation (OPD) on a 160-question MATH panel (`data/math-panel.jsonl`), using **Qwen3-4B-Instruct-2507** as the teacher and **Qwen3-1.7B** as the student.

## 1. Environment setup

### Option A: Classroom workstation (setup-free, recommended)

On your assigned Ubuntu workstation, open `~/work/project2` in VS Code and open a terminal (**Terminal → New Terminal**). All Python environments, model snapshots, and benchmark assets are pre-installed—no setup is required.

### Option B: Your own Linux GPU machine (optional)

Requires a Linux GPU with at least 24 GiB VRAM (tested on RTX 5090 D v2 and RTX 3090) and Python 3.12. From the `p2/` folder:

1. Create three lightweight virtual environments (`env/sft`, `env/opd`, `env/eval`) with the pinned packages:
   ```bash
   python3 -m venv env/sft
   env/sft/bin/pip install torch==2.9.1 trl==0.25.1 transformers==4.57.3 datasets==4.0.0 peft==0.18.1 accelerate==1.12.0 safetensors==0.8.0 huggingface-hub==0.36.2 tokenizers==0.22.2

   python3 -m venv env/opd
   env/opd/bin/pip install torch==2.9.1 trl==1.8.0 transformers==4.57.3 datasets==4.7.0 peft==0.18.0 accelerate==1.16.0 safetensors==0.8.0 huggingface-hub==0.36.2 tokenizers==0.22.2

   python3 -m venv env/eval
   env/eval/bin/pip install vllm==0.12.0 transformers==4.57.3 peft==0.18.1 math-verify==0.9.0 nltk==3.9.1 langdetect==1.0.9 absl-py==2.1.0 immutabledict==4.2.1 accelerate==1.16.0 safetensors==0.8.0
   ```
2. Stage the public benchmark files, Google IFEval grader, and NLTK data into `assets/`:
   ```bash
   env/eval/bin/python prepare.py
   ```
3. Download the pinned student and teacher snapshots into `models/`:
   ```bash
   env/sft/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-1.7B', revision='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e', local_dir='models/student'); snapshot_download('Qwen/Qwen3-4B-Instruct-2507', revision='cdbee75f17c01a7cc42f958dc650907174af0554', local_dir='models/teacher')"
   ```
4. Export the paths before running `bash run.sh`:
   ```bash
   export P2_STUDENT="$(pwd)/models/student"
   export P2_TEACHER="$(pwd)/models/teacher"
   export P2_SFT_PYTHON="$(pwd)/env/sft/bin/python"
   export P2_OPD_PYTHON="$(pwd)/env/opd/bin/python"
   export P2_EVAL_PYTHON="$(pwd)/env/eval/bin/python"
   ```

## 2. Train and evaluate both adapters

```bash
bash run.sh
```

The script trains one SFT adapter (using the pre-generated teacher solutions in `data/math-panel.jsonl`) and one OPD adapter (using student-generated solutions), then evaluates both on all 500 MATH-500 questions and 541 IFEval prompts with four draws per prompt. Progress prints in the terminal.

**Runtime:** SFT finishes in a few minutes, OPD takes under an hour, and full evaluation takes roughly another hour. On a classroom **RTX 5090 D v2** workstation, the full pipeline finishes in roughly 1.5 to 2 hours—well within a single **4.5-hour** booking session (especially with a coding agent helping on the analysis).

Outputs are saved under `booking-run/`:

| Default path | Contents |
|---|---|
| `booking-run/data/math-panel.jsonl` | Training questions, teacher solutions, and reference answers |
| `booking-run/opd/rollouts.jsonl` | Student responses with question ID, rollout order, and stop status |
| `booking-run/sft/summary.json`, `booking-run/opd/summary.json` | Training summaries |
| `booking-run/sft/adapter/`, `booking-run/opd/adapter/` | Trained LoRA adapters |
| `booking-run/scores/` | Eight per-draw score files (all four checkpoints) and `summary.csv` |

## 3. Complete the TODOs and analyze

Complete the two guided TODOs in `analysis_exercises.py` (see `AGENTS.md` if using a coding agent). The analysis scripts use only Python 3's standard library, so you can run them **directly in the project folder**—either on the workstation right after `run.sh` finishes, or later on your own computer from an exported archive.

```bash
python3 check_analysis.py
python3 analyze_results.py booking-run/scores > analysis.json
python3 inspect_trajectory.py training
python3 inspect_trajectory.py math QUESTION_ID --draw 0
python3 inspect_trajectory.py ifeval PROMPT_ID --draw 0
```

The training viewer pairs a saved OPD student rollout with the teacher solution for that question (pass a question ID after `training` to pick another). For MATH-500 or IFEval, replace `QUESTION_ID` / `PROMPT_ID` with an ID from the score files and choose a draw from `0` to `3` to inspect original, SFT, and OPD responses side by side (including per-instruction checks for IFEval).

## 4. Package for submission (or offline backup)

```bash
python3 export_analysis.py
```

This creates `part2-analysis.zip` containing your completed `analysis_exercises.py`, supplied analysis scripts, training summaries, saved OPD rollouts, and score files (without model weights or LoRA adapters). Add your report PDF (e.g., `report.pdf`), `analysis.json`, and any custom analysis scripts you wrote at the top level of that archive before uploading that single ZIP to Blackboard (you may also copy `part2-analysis.zip` to your own computer first to finish the analysis offline).

---

Training questions come from the MATH train split; teacher solutions were generated by Qwen3-4B-Instruct-2507 and screened for final-answer correctness, EOS, length, and overlap with MATH-500. Reference scores are precomputed for the original Qwen3-1.7B student and frozen teacher. [MATH](https://github.com/hendrycks/math) · [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) · [IFEval](https://huggingface.co/datasets/google/IFEval)
