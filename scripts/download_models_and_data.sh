#!/bin/bash
# Download models and datasets for offline use
# Run this on a machine WITH internet access, then copy to the offline machine
#
# Usage: ./scripts/download_models_and_data.sh [--proxy PROXY_URL]
#
# Models are saved to: pretrained_models/
# Data is saved to:    dataset/

set -e

PROXY=""
if [ "$1" = "--proxy" ] && [ -n "$2" ]; then
    export https_proxy="$2"
    export http_proxy="$2"
    echo "Using proxy: $2"
fi

# ============================================================
# Models to download
# ============================================================
MODELS=(
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
)

MODEL_DIR="pretrained_models"
mkdir -p "$MODEL_DIR"

for MODEL in "${MODELS[@]}"; do
    LOCAL_NAME=$(echo "$MODEL" | tr '/' '_')
    TARGET="${MODEL_DIR}/${LOCAL_NAME}"
    
    if [ -d "$TARGET" ] && [ -f "$TARGET/config.json" ]; then
        echo "✅ Already downloaded: $MODEL → $TARGET"
        continue
    fi
    
    echo "⬇️  Downloading model: $MODEL → $TARGET"
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${MODEL}',
    local_dir='${TARGET}',
    local_dir_use_symlinks=False,
)
print('Done: ${MODEL}')
"
    echo "✅ Downloaded: $MODEL → $TARGET"
done

# ============================================================
# Datasets to download
# ============================================================
DATASETS=(
    "open-r1/OpenR1-Math-220k"
)

DATA_DIR="dataset/llm_rl"
mkdir -p "$DATA_DIR"

for DATASET in "${DATASETS[@]}"; do
    LOCAL_NAME=$(echo "$DATASET" | cut -d'/' -f2)
    TARGET="${DATA_DIR}/${LOCAL_NAME}"
    
    if [ -d "$TARGET" ]; then
        echo "✅ Already downloaded: $DATASET → $TARGET"
        continue
    fi
    
    echo "⬇️  Downloading dataset: $DATASET → $TARGET"
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${DATASET}',
    repo_type='dataset',
    local_dir='${TARGET}',
    local_dir_use_symlinks=False,
)
print('Done: ${DATASET}')
"
    echo "✅ Downloaded: $DATASET → $TARGET"
done

echo ""
echo "============================================================"
echo "All downloads complete. Directory structure:"
echo "============================================================"
echo ""
echo "Models:"
for MODEL in "${MODELS[@]}"; do
    LOCAL_NAME=$(echo "$MODEL" | tr '/' '_')
    echo "  ${MODEL_DIR}/${LOCAL_NAME}"
done
echo ""
echo "Datasets:"
for DATASET in "${DATASETS[@]}"; do
    LOCAL_NAME=$(echo "$DATASET" | cut -d'/' -f2)
    echo "  ${DATA_DIR}/${LOCAL_NAME}"
done
echo ""
echo "Copy these directories to the offline machine, then run training scripts."
