#!/bin/bash
#SBATCH --job-name=unbranding_vss
#SBATCH --time=01:00:00
#SBATCH --qos=big
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=dgxh100
#SBATCH --output=logs/unbranding_%A_%a.out
#SBATCH --error=logs/unbranding_%A_%a.err
#SBATCH --array=0-4

set -euo pipefail

# ---- Config ----
WORLD_SIZE=5                 # total number of ranks across the array
PORT_BASE=8000                          # ports: 8000..8015
HOST=127.0.0.1                          # bind servers locally; unique per node
MODEL=google/gemma-3-4b-it #"nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1"
MODEL_MAP=gemma3
ENV_PATH="PATH"
CONDA_SH="PATH"
WORKDIR="PATH"
HF_HOME="PATH"

# ---- Derived per-task vars ----
RANK="${SLURM_ARRAY_TASK_ID}"
PORT=$((PORT_BASE + RANK))
API_BASE="http://${HOST}:${PORT}"

echo "[task ${SLURM_ARRAY_TASK_ID}] Starting: WORLD_SIZE=${WORLD_SIZE} RANK=${RANK} PORT=${PORT}"
mkdir -p "${WORKDIR}/logs"
cd "${WORKDIR}"

# ---- Start single vLLM server on the assigned GPU ----
# Slurm sets CUDA_VISIBLE_DEVICES to the single allocated GPU; no need to override.
echo "[task ${SLURM_ARRAY_TASK_ID}] Launching vLLM server on ${API_BASE}"
echo "[task ${SLURM_ARRAY_TASK_ID}] Launching vLLM server on ${API_BASE}"
bash -lc "
  source ${CONDA_SH} ${ENV_PATH}
  export HF_HOME='${HF_HOME}'
  export HF_HUB_OFFLINE=1
  exec vllm serve ${MODEL} \
    --host 0.0.0.0 \
    --port ${PORT} \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --limit-mm-per-prompt '{\"image\": 2}'
" > "logs/vllm_${SLURM_ARRAY_TASK_ID}_${MODEL_MAP}.log" 2>&1 &
SERVER_PID=$!


cleanup() {
  echo "[task ${SLURM_ARRAY_TASK_ID}] Cleaning up..."
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# ---- Health check ----
echo "[task ${SLURM_ARRAY_TASK_ID}] Waiting for server ${API_BASE} ..."
for j in {1..150}; do  # 150*2s = 300s max
  if curl -sS --max-time 2 --fail "${API_BASE}/v1/models" >/dev/null; then
    echo "[task ${SLURM_ARRAY_TASK_ID}] Server is up."
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[task ${SLURM_ARRAY_TASK_ID}] Server exited early. See logs/vllm_${SLURM_ARRAY_TASK_ID}.log"
    exit 1
  fi
  sleep 2
done

# Final check
if ! curl -sS --max-time 2 --fail "${API_BASE}/v1/models" >/dev/null; then
  echo "[task ${SLURM_ARRAY_TASK_ID}] Server failed to start within timeout."
  exit 1
fi

# ---- Launch one client for this rank (CPU-only) ----
echo "[task ${SLURM_ARRAY_TASK_ID}] Launching client for rank ${RANK} -> ${API_BASE}"
bash -lc "
  source ${CONDA_SH} ${ENV_PATH}
  export HF_HOME='${HF_HOME}'
  export WORLD_SIZE='${WORLD_SIZE}'
  export RANK='${RANK}'
  # ensure client does not grab the GPU
  CUDA_VISIBLE_DEVICES='' python client.py \
    --exp EXP_NAME \
    --model-type ${MODEL_MAP} \
    --gt_imgs_dir GT_DIR \
    --gen_imgs_dir GEN_DIR \
    --prompts-file configs/vlm_vss.json \
    --server-url ${API_BASE}
"

echo "[task ${SLURM_ARRAY_TASK_ID}] Client finished."

# ---- Stop server ----
echo "[task ${SLURM_ARRAY_TASK_ID}] Stopping server..."
kill "${SERVER_PID}" 2>/dev/null || true
wait "${SERVER_PID}" 2>/dev/null || true
echo "[task ${SLURM_ARRAY_TASK_ID}] Done."
