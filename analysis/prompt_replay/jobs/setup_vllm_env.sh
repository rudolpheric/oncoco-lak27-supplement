#!/bin/bash
#SBATCH --job-name=vllm_setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=<partition>
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --qos=<qos>
#SBATCH --output=<scratch>/oncoco_replay/logs/setup_%j.out
#SBATCH --error=<scratch>/oncoco_replay/logs/setup_%j.err

# One-off: build a vLLM venv for openai/gpt-oss-120b and pre-download the weights.
# Runs as a job (never on the login node). Re-runnable: skips what already exists.

set -euo pipefail

echo "=================================================================="
echo "vLLM env setup for gpt-oss-120b at $(date)"
echo "Job ID: ${SLURM_JOB_ID:-unknown}   Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -n 1 || echo 'no gpu detected')"
echo "=================================================================="

module purge 2>/dev/null || true
module load cuda/cuda-12.6 2>/dev/null || true

SUBMIT_DIR="<scratch>/oncoco_replay"
SCRATCH_ROOT="${SCRATCH_ROOT:-<scratch-large>}"
VENV_DIR="${VENV_DIR:-<scratch>/.venvs/vllm_gptoss}"
BASE_PY="${BASE_PY:-$SCRATCH_ROOT/.conda_envs/llm_ft_comparison/bin/python}"
VLLM_SPEC="${VLLM_SPEC:-vllm}"
MODEL_ID="${MODEL_ID:-openai/gpt-oss-120b}"
# Weights (~65 GB). The shared file systems lack the quota headroom, so the
# default is a node-local directory; override HF_HOME if a shared location becomes free.
export HF_HOME="${HF_HOME:-/tmp/$USER/hf}"
export HF_HUB_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false
# caches on the node-local tmpfs: the shared file system has ~40 GB quota headroom, the venv alone needs ~10 GB
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/$USER/pip}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/$USER/uv}"
DOWNLOAD="${DOWNLOAD:-0}"   # 1 = also fetch the weights into HF_HOME (node-local, ephemeral) as a timing test
if [ -z "${HF_TOKEN:-}" ] && [ -f "$SCRATCH_ROOT/.hf_token" ]; then
    export HF_TOKEN="$(tr -d '\r\n' < "$SCRATCH_ROOT/.hf_token")"
fi

mkdir -p "$HF_HUB_CACHE" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$(dirname "$VENV_DIR")" "$SUBMIT_DIR/logs"
cd "$SUBMIT_DIR"

echo "Venv:      $VENV_DIR"
echo "Base py:   $BASE_PY ($($BASE_PY --version 2>&1))"
echo "HF cache:  $HF_HUB_CACHE ($(df -h "$HF_HOME" | tail -1))"

if [ ! -x "$VENV_DIR/bin/python" ]; then
    "$BASE_PY" -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/python" -m pip install --upgrade pip uv >/dev/null
"$VENV_DIR/bin/uv" pip install --python "$VENV_DIR/bin/python" "$VLLM_SPEC" openai huggingface_hub
echo "Installed: $("$VENV_DIR/bin/python" -c 'import vllm, torch; print("vllm", vllm.__version__, "torch", torch.__version__)')"

if [ "$DOWNLOAD" = "1" ]; then
"$VENV_DIR/bin/python" - <<PY
import os
from huggingface_hub import snapshot_download
p = snapshot_download("$MODEL_ID", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.py", "*.jinja"])
print("weights at", p)
PY
du -sh "$HF_HUB_CACHE"/models--* 2>/dev/null || true
fi
df -h "$SCRATCH_ROOT" | tail -1; du -sh "$VENV_DIR"
echo "Done at $(date)"
