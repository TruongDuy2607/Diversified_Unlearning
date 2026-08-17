#!/bin/bash
#SBATCH --job-name=unbranding_generate
#SBATCH --time=10:00:00
#SBATCH --qos=big
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --partition=dgxh100
#SBATCH --output=logs/unbranding_%A_%a.out
#SBATCH --error=logs/unbranding_%A_%a.err
#SBATCH --array=0-5

# Models: sd14 sdxl sd35-large flux1-schnell flux1-dev qwen-image

MODELS=(flux1-schnell)

# Map array index -> (model_idx, rank)
WORLD_SIZE=6
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
MODEL_IDX=$(( TASK_ID / $WORLD_SIZE ))
RANK=$(( TASK_ID % $WORLD_SIZE ))
MODEL=${MODELS[$MODEL_IDX]}

cd PATH
export HF_HOME="PATH HERE"
export WORLD_SIZE
export RANK

echo "JOB=${SLURM_JOB_ID} TASK=${SLURM_ARRAY_TASK_ID} NODE=$HOSTNAME GPU=$CUDA_VISIBLE_DEVICES MODEL=$MODEL WORLD_SIZE=$WORLD_SIZE RANK=$RANK"
nvidia-smi -L || true
OUT_DIR="output"
mkdir -p "$OUT_DIR"

uv run generate_images.py \
  --model "$MODEL" \
  --seed 42 \
  --output_dir "$OUT_DIR" \
  --prompts_file "prompts.csv"
