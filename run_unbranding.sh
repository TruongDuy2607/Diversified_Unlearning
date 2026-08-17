#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

generate_evaluation_images() {
    local model_name="$1"
    local gpu="$2"

    CUDA_VISIBLE_DEVICES="${gpu}" python3 eval-scripts/generate-images.py \
        --models_path "models" \
        --model_name "${model_name}" \
        --prompts_path "diverse_prompts/characters/erasure_prompts/erased_apple_laptop_unbranding.csv" \
        --save_path "images-gen/unbranding" \
        --num_samples 1 \
        --from_case 0 \
        --to_case -1

    CUDA_VISIBLE_DEVICES="${gpu}" python3 eval-scripts/generate-images.py \
        --models_path "models" \
        --model_name "${model_name}" \
        --prompts_path "diverse_prompts/characters/retaining_prompts/retained_apple_laptop_unbranding.csv" \
        --save_path "images-gen/unbranding" \
        --num_samples 1 \
        --from_case 0 \
        --to_case -1
}

# CUDA_VISIBLE_DEVICES=0 python3 train-scripts/ESD/train-esd.py \
#     --train_method "xattn" \
#     --ckpt_path "models/erase/sd-v1-4-full-ema.ckpt" \
#     --prompt "apple laptop" \
#     --target_prompt "gaming laptop" \
#     --config_path "configs/stable-diffusion/v1-inference.yaml" \
#     --info "unbranding-apple-laptop"
# generate_evaluation_images "esd" 0

# Generate this reference once before running UNBRANDING VSS.
# generate_evaluation_images "SD-v1-4" 0

# CUDA_VISIBLE_DEVICES=1 python3 train-scripts/ESD/train-esd-diverse.py \
#     --prompt_csv "diverse_prompts/characters/training_prompts/unbranding_apple_laptop_level-1.csv" \
#     --train_method "xattn" \
#     --ckpt_path "models/erase/sd-v1-4-full-ema.ckpt" \
#     --prompt "apple laptop" \
#     --config_path "configs/stable-diffusion/v1-inference.yaml" \
#     --info "unbranding-apple-laptop" \
#     --level "level-1"

generate_evaluation_images "esd-level-1" 1
