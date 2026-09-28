#!/bin/bash
#SBATCH --job-name=oncoco_replay
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=<partition>
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G                 # /tmp is a tmpfs: the 65 GB of weights count against the job
#SBATCH --gres=gpu:1
#SBATCH --qos=<qos>
#SBATCH --output=<scratch>/oncoco_replay/logs/replay_%j.out
#SBATCH --error=<scratch>/oncoco_replay/logs/replay_%j.err

# Serve openai/gpt-oss-120b with vLLM on one H200 and regenerate the client turns of the
# prompt-replay experiment (analysis/prompt_replay/generate_replay.py). Output is appended
# per request, so a preempted/requeued job resumes where it stopped.

set -euo pipefail

echo "=================================================================="
echo "OnCoCo prompt replay (gpt-oss-120b via vLLM) at $(date)"
echo "Job ID: ${SLURM_JOB_ID:-unknown}   QoS: ${SLURM_JOB_QOS:-?}   Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -n 1 || echo 'no gpu detected')"
echo "=================================================================="

# No CUDA module: the torch wheel bundles its CUDA runtime, and the module points CUDA_HOME at a
# path that does not exist on the compute nodes (FlashInfer then tries to JIT with a missing nvcc).
export VLLM_USE_FLASHINFER_SAMPLER=0

SUBMIT_DIR="<scratch>/oncoco_replay"
SCRATCH_ROOT="${SCRATCH_ROOT:-<scratch-large>}"
VENV_DIR="${VENV_DIR:-<scratch>/.venvs/vllm_gptoss}"
MODEL_ID="${MODEL_ID:-openai/gpt-oss-120b}"
export HF_HOME="${HF_HOME:-/tmp/$USER/hf}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$SCRATCH_ROOT/.hf_token" ]; then
    export HF_TOKEN="$(tr -d '\r\n' < "$SCRATCH_ROOT/.hf_token")"
fi

mkdir -p "$HF_HUB_CACHE" "$SUBMIT_DIR/logs" "$SUBMIT_DIR/outputs"
cd "$SUBMIT_DIR"
export PATH="$VENV_DIR/bin:$PATH"
hash -r

# ---------------------------------------------------------------------------
# Run config (override with sbatch --export=ALL,ARMS=A+C,LIMIT=5 ...; sbatch splits --export on
# commas, so separate arms with + or ;)
ARMS="${ARMS:-A+C}"
ARMS="${ARMS//[;+]/,}"
LIMIT="${LIMIT:-0}"
CONCURRENCY="${CONCURRENCY:-16}"
ROLE="${ROLE:-user}"
REASONING_EFFORT="${REASONING_EFFORT:-}"
REQUESTS="${REQUESTS:-$SUBMIT_DIR/replay_requests.jsonl}"
OUTPUT="${OUTPUT:-$SUBMIT_DIR/outputs/replay_outputs.jsonl}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_UTIL="${GPU_UTIL:-0.92}"   # 66 GiB weights on an 80 GB A100 leave ~7 GB for KV cache

echo "Arms: $ARMS   Limit/arm: $LIMIT   Concurrency: $CONCURRENCY   Role: $ROLE   Effort: ${REASONING_EFFORT:-default}"
echo "Requests: $REQUESTS"
echo "Output:   $OUTPUT"
echo "vLLM: $(python -c 'import vllm; print(vllm.__version__)')"

# Weights: download first if the (node-local) cache is empty.
python - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download("$MODEL_ID", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.py", "*.jinja"])
print("weights at", p)
PY

# ---------------------------------------------------------------------------
# Serve
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" --port "$PORT" --host 127.0.0.1 \
    --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_UTIL" \
    --served-model-name "$MODEL_ID" \
    > "$SUBMIT_DIR/logs/vllm_server_${SLURM_JOB_ID:-manual}.log" 2>&1 &
SERVER_PID=$!
trap 'echo "stopping vLLM ($SERVER_PID)"; kill $SERVER_PID 2>/dev/null || true' EXIT

for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "vLLM ready after ${i}x10s"; break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "vLLM server died; see logs/vllm_server_${SLURM_JOB_ID:-manual}.log"; exit 1
    fi
    sleep 10
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "vLLM not ready after 30 min"; exit 1; }
curl -s "http://127.0.0.1:$PORT/v1/models" | head -c 400; echo

# ---------------------------------------------------------------------------
# Generate
EXTRA=()
[ "$LIMIT" != "0" ] && EXTRA+=(--limit "$LIMIT")
[ -n "$REASONING_EFFORT" ] && EXTRA+=(--reasoning-effort "$REASONING_EFFORT")
python -B generate_replay.py \
    --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL_ID" \
    --requests "$REQUESTS" --output "$OUTPUT" \
    --arms "$ARMS" --concurrency "$CONCURRENCY" --role "$ROLE" "${EXTRA[@]}"

echo ""
echo "Done at $(date). Output: $OUTPUT ($(wc -l < "$OUTPUT") lines)"
