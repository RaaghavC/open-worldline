#!/usr/bin/env bash
# Prepared only. Run inside the approved Linux container, never on the Mac.
# Usage: bash prepare.sh /workspace/open-worldline /workspace/setup-evidence-v1
set -euo pipefail
if [ "$#" -ne 2 ]; then
  printf '%s\n' 'Usage: bash prepare.sh ABSOLUTE_REPO NEW_ABSOLUTE_SETUP_DIR' >&2
  exit 2
fi
task_repo="$1"
task_setup="$2"
case "$task_repo" in /*) ;; *) exit 2 ;; esac
case "$task_setup" in /*) ;; *) exit 2 ;; esac
task_check="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/runtime-check.py"
task_requirements="$task_repo/experiments/wan22_native/cuda_reference/requirements.txt"
test -f "$task_check"
test -f "$task_requirements"
test ! -e "$task_setup"
mkdir "$task_setup"
exec > >(tee "$task_setup/setup.log") 2>&1
trap 'status=$?; printf "exit_code=%s\n" "$status" > "$task_setup/terminal.txt"; exit "$status"' EXIT
/opt/conda/bin/python - "$task_requirements" <<'PY'
import hashlib, pathlib, sys
p = pathlib.Path(sys.argv[1])
assert hashlib.sha256(p.read_bytes()).hexdigest() == '6f2400ac8c11d050fb96d318abcfee6ee6674d3bb0c887c9fbcdf371a1053378'
PY
/opt/conda/bin/python "$task_check" > "$task_setup/before.json"
nvidia-smi > "$task_setup/nvidia-smi.txt"
cp "$task_check" "$task_setup/runtime-check.py"
cp "$task_requirements" "$task_setup/requirements.txt"
/opt/conda/bin/python -m venv --system-site-packages "$task_setup/venv"
task_python="$task_setup/venv/bin/python"
"$task_python" -m pip install --only-binary=:all: -r "$task_requirements"
# This exact binary is selected only after the Python/Torch/CUDA/ABI checks.
task_wheel_name='flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl'
task_wheel="$task_setup/$task_wheel_name"
curl --fail --show-error --location --proto '=https' --tlsv1.2 \
  --connect-timeout 20 --max-time 180 --retry 0 \
  --output "$task_wheel" \
  'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl'
"$task_python" - "$task_wheel" "$task_setup/wheel-download.json" <<'PY'
import hashlib, json, pathlib, sys
p = pathlib.Path(sys.argv[1])
assert p.stat().st_size == 187815463, p.stat().st_size
h = hashlib.sha256()
with p.open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        h.update(chunk)
pathlib.Path(sys.argv[2]).write_text(json.dumps({
    'file': p.name, 'bytes': p.stat().st_size, 'sha256': h.hexdigest(),
    'official_asset_id': 224783840, 'upstream_published_digest': None,
    'meaning': 'Local download identity; upstream GitHub API supplies no digest.'
}, indent=2) + '\n')
PY
"$task_python" -m pip install --no-deps "$task_wheel"
"$task_python" "$task_check" --flash > "$task_setup/after.json"
"$task_python" -m pip freeze --all > "$task_setup/pip-freeze.txt"
# Records inherited package conflicts without silently changing the base image.
"$task_python" -m pip check > "$task_setup/pip-check.txt"
printf '%s\n' 'Setup checks passed. No attention kernel, model, cloud termination or paid-run admission was executed by this script.'
