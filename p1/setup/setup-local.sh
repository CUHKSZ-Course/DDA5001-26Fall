#!/usr/bin/env bash
# Personal Linux/macOS/WSL installation. Never run this as root on a classroom machine.
set -Eeuo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mode=${1:-cpu}
[[ $mode == cpu || $mode == cuda ]] || { echo 'Usage: bash setup-local.sh cpu|cuda'; exit 1; }
command -v uv >/dev/null || { echo 'Install uv: https://docs.astral.sh/uv/getting-started/installation/'; exit 1; }
[[ ! -e .venv ]] || { echo '.venv exists; refusing to replace your environment.'; exit 1; }
uv venv --python 3.12.16 .venv
if [[ $mode == cuda ]]; then
    .venv/bin/python "$HERE/inspect_system.py" --data-root "$PWD" --output-dir "$PWD/setup-evidence" \
        --stage personal-setup --flavor "${TORCH_FLAVOR:-auto}" --torch-version "${TORCH_VERSION:-2.9.1}"
    plan_lines=$(.venv/bin/python "$HERE/inspect_system.py" --read-plan "$PWD/setup-evidence/latest.json")
    # Avoid mapfile so this helper also parses in the older bash shipped on macOS.
    torch_version=$(printf '%s\n' "$plan_lines" | sed -n '1p')
    torch_index=$(printf '%s\n' "$plan_lines" | sed -n '3p')
    uv pip install --python .venv/bin/python "torch==$torch_version" --index-url "$torch_index"
elif [[ $(uname -s) == Darwin ]]; then
    uv pip install --python .venv/bin/python torch==2.9.1
else
    uv pip install --python .venv/bin/python torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
fi
uv pip install --python .venv/bin/python --constraint "$HERE/requirements-validated-common.txt" \
    -r "$HERE/requirements.txt" "torch==${torch_version:-2.9.1}"
uv pip check --python .venv/bin/python
uv pip freeze --python .venv/bin/python > requirements-local-resolved.txt
if [[ $mode == cuda ]]; then
    .venv/bin/python "$HERE/verify_env.py"
else
    .venv/bin/python "$HERE/verify_env.py" --cpu
fi
echo 'Activate with: source .venv/bin/activate'
