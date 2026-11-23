#!/usr/bin/env bash
# Train SFT and OPD and evaluate both adapters on one GPU.
set -euo pipefail
umask 077
panel=${1:-data/math-panel.jsonl}
references=${2:-data/reference}
run=${3:-booking-run}
seed=42
if [[ -L .project2 ]]; then
    shared=$(realpath -e .project2)
    export P2_STUDENT=${P2_STUDENT:-"$shared/models/student"}
    export P2_TEACHER=${P2_TEACHER:-"$shared/models/teacher"}
    export P2_SFT_PYTHON=${P2_SFT_PYTHON:-"$shared/env/sft/bin/python"}
    export P2_OPD_PYTHON=${P2_OPD_PYTHON:-"$shared/env/opd/bin/python"}
    export P2_EVAL_PYTHON=${P2_EVAL_PYTHON:-"$shared/env/eval/bin/python"}
    export HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1
fi
: "${P2_STUDENT:?stage the student model}"
: "${P2_TEACHER:?stage the teacher model}"
: "${P2_SFT_PYTHON:?set the stock SFT interpreter}"
: "${P2_OPD_PYTHON:?set the stock OPD interpreter}"
: "${P2_EVAL_PYTHON:?set the evaluator interpreter}"

if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]; then
    echo "GPU is busy. Wait for other GPU jobs to finish before running run.sh." >&2
    exit 1
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ -e $run ]]; then
    echo "Output directory $run already exists. Move or remove it before starting a new run." >&2
    exit 1
fi
mkdir "$run"
mkdir "$run/scores" "$run/data"
if [[ -L .project2 ]]; then
    mkdir "$run/cache" "$run/tmp"
    export TMPDIR="$(realpath "$run/tmp")" XDG_CACHE_HOME="$(realpath "$run/cache")"
    export HF_HOME="$XDG_CACHE_HOME/hf" CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
    export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    # vLLM's Unix socket name has a separate, shorter path limit.
    rpc_dir=$(mktemp -d /tmp/dda-p2.XXXXXXXX)
    export VLLM_RPC_BASE_PATH="$rpc_dir"
    trap 'find "$rpc_dir" -maxdepth 1 -type s -delete; rmdir -- "$rpc_dir" 2>/dev/null || true' EXIT
fi
mark() { printf '%s\t%s\n' "$1" "$(date +%s)" >>"$run/timing.tsv"; }

mark start
cp "$panel" "$run/data/math-panel.jsonl"
panel="$run/data/math-panel.jsonl"
questions=$("$P2_EVAL_PYTHON" -c '
import json
import sys
with open(sys.argv[1], encoding="utf-8") as source:
    rows = [json.loads(line) for line in source if line.strip()]
if not rows or len({row["id"] for row in rows}) != len(rows):
    raise ValueError("Expected a nonempty panel with unique question IDs")
print(len(rows))
' "$panel")
sft_steps=$(((questions + 7) / 8))
if ((sft_steps < 64)); then sft_steps=64; fi
for arm in original teacher; do
    for task in math ifeval; do
        cp "$references/$arm-$task.jsonl" "$run/scores/"
    done
done

"$P2_SFT_PYTHON" schema_sft.py "$panel" "$run/sft" --lr 1e-5 --steps "$sft_steps" --seed "$seed"
mark sft_trained
"$P2_OPD_PYTHON" schema_opd.py "$panel" "$P2_STUDENT" "$P2_TEACHER" "$run/opd" --steps "$questions" --seed "$seed" --save-rollouts
mark opd_trained
"$P2_EVAL_PYTHON" - "$panel" "$run" <<'PY'
import json
from pathlib import Path
import sys

panel, run = map(Path, sys.argv[1:])
ids = {json.loads(line)["id"] for line in panel.read_bytes().splitlines() if line.strip()}
sft, opd = [
    json.loads((run / variant / "summary.json").read_bytes()) for variant in ("sft", "opd")
]
if sft["panel_sha256"] != opd["panel_sha256"] or sft["questions"] != len(ids):
    raise ValueError("The two trainers used different training data")
rollouts = opd["rollouts"]
if len(rollouts) != len(ids) or {r["id"] for r in rollouts} != ids:
    raise ValueError("Expected one observed on-policy rollout per shared training question")
saved = [json.loads(line) for line in (run / "opd/rollouts.jsonl").read_bytes().splitlines()]
if len(saved) != len(rollouts) or any(
    any(row.get(field) != value for field, value in meta.items())
    for row, meta in zip(saved, rollouts)
):
    raise ValueError("Saved OPD trajectories do not match the training log")
PY

for arm in sft opd; do
    for task in math ifeval; do
        "$P2_EVAL_PYTHON" evaluate.py "$arm" "$task" --samples 4 --seed "$seed" \
            --adapter "$run/$arm/adapter" --out "$run/scores/$arm-$task.jsonl"
        mark "${arm}_${task}_scored"
    done
done
"$P2_EVAL_PYTHON" booking_report.py "$run/scores" --assets assets
mark report_ready
start=$(awk '$1=="start" {print $2}' "$run/timing.tsv")
end=$(awk '$1=="report_ready" {print $2}' "$run/timing.tsv")
elapsed=$((end - start))
printf '%s\n' "$elapsed" >"$run/elapsed-seconds.txt"
printf 'Report ready after %s seconds\n' "$elapsed"
