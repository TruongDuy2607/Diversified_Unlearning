#!/bin/bash

# Usage: bash gen_apple_laptop.sh "<prompt>"

if [[ -z "$1" ]]; then
    echo "Usage: $0 \"<prompt>\""
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_CSV=$(mktemp /tmp/prompt_XXXXXX.csv)

echo "case_number,prompt,evaluation_seed" > "${TMP_CSV}"
echo "0,\"$1\",0" >> "${TMP_CSV}"

python "${SCRIPT_DIR}/eval-scripts/generate-images.py" \
    --model_name "SD-v1-4" \
    --prompts_path "${TMP_CSV}" \
    --save_path "${SCRIPT_DIR}/images-gen" \
    --models_path "${SCRIPT_DIR}/models/erase" \
    --device "cuda:0" \
    --guidance_scale 7.5 \
    --image_size 512 \
    --ddim_steps 100 \
    --num_samples 1 \
    --from_case 0

rm -f "${TMP_CSV}"
