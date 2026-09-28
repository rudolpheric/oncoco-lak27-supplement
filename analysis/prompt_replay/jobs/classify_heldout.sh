#!/bin/bash
#SBATCH --job-name=oncoco_classify
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=<partition>
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --qos=<qos>
#SBATCH --output=<scratch>/oncoco_replay/logs/classify_%j.out
#SBATCH --error=<scratch>/oncoco_replay/logs/classify_%j.err

# OnCoCo classification of the held-out replay arms on a GPU node: SaT-6l + LoRA segmentation and
# the XLM-RoBERTa-large OnCoCo classifier, in a venv pinned to the versions of the local pipeline
# (wtpsplit 2.1.7, adapters 1.2.0, transformers 4.51.3, torch 2.10.0). BASE mirrors the repository
# layout, so classification_all.py runs with its default model paths. Arm O is the check: its output
# must reproduce the local classification of the same text.

set -euo pipefail
BASE="${BASE:-<scratch>/oncoco_replay/classify}"
VENV="${VENV:-<scratch>/.venvs/oncoco_classify}"
PYBASE="${PYBASE:-<scratch-large>/.conda_envs/llm_ft_comparison/bin/python3.11}"
ARMS="${ARMS:-O+A+D+G}"
ARMS="${ARMS//[;+]/ }"
export HF_HOME="${HF_HOME:-/tmp/$USER/hf}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/$USER/pip}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR"

echo "OnCoCo held-out classification at $(date) on $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)"

if [ ! -x "$VENV/bin/python" ]; then
    "$PYBASE" -m venv "$VENV"
    "$VENV/bin/python" -m pip install --upgrade pip
    "$VENV/bin/python" -m pip install torch==2.10.0 transformers==4.51.3 wtpsplit==2.1.7 adapters==1.2.0 pandas numpy
fi
"$VENV/bin/python" -c "import torch, transformers, wtpsplit, adapters; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| transformers', transformers.__version__, '| wtpsplit', wtpsplit.__version__, '| adapters', adapters.__version__)"

cd "$BASE"
H=data/processed/prompt_replay_heldout
for a in $ARMS; do
    echo "--- arm $a $(date +%T)"
    "$VENV/bin/python" analysis/semantic/classification/classification_all.py --chat_csv "$H/replay_chat_$a.csv" \
        --no-include_mail --segmentation sat6l --use_role_prefix --output "$H/oncoco_classification_$a.json"
done
echo "Done at $(date)"
