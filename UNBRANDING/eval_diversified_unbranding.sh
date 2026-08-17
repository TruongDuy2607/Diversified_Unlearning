#!/usr/bin/env bash
set -euo pipefail

BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${BENCHMARK_DIR}/.." && pwd)"
IMAGES_ROOT="${IMAGES_ROOT:-${PROJECT_DIR}/images-gen/unbranding}"
DATASET_CSV="${DATASET_CSV:-${PROJECT_DIR}/data/UNBRANDING/unbranding_v1.csv}"
RESULTS_DIR="${RESULTS_DIR:-${BENCHMARK_DIR}/results/diversified-unlearn}"
MODEL_TYPE="${MODEL_TYPE:-llava}"
VLM_MODEL="${VLM_MODEL:-llava-hf/llava-1.5-7b-hf}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-8000}"
SERVER_GPU="${SERVER_GPU:-0}"
MODELS="${MODELS:-esd esd-level-1 SD-v1-4}"
SPLITS="${SPLITS:-erase retain}"
REFERENCE_MODEL="${REFERENCE_MODEL:-SD-v1-4}"
RUN_BPS="${RUN_BPS:-1}"
RUN_VSS="${RUN_VSS:-1}"

command -v uv >/dev/null || {
    echo "uv is required: https://docs.astral.sh/uv/" >&2
    exit 1
}

mkdir -p "${RESULTS_DIR}" "${BENCHMARK_DIR}/logs"

CUDA_VISIBLE_DEVICES="${SERVER_GPU}" uv run --project "${BENCHMARK_DIR}" \
    vllm serve "${VLM_MODEL}" \
    --host "${SERVER_HOST}" \
    --port "${SERVER_PORT}" \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --limit-mm-per-prompt '{"image": 2}' \
    >"${BENCHMARK_DIR}/logs/vllm_${MODEL_TYPE}.log" 2>&1 &
SERVER_PID=$!

cleanup() {
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

SERVER_URL="http://${SERVER_HOST}:${SERVER_PORT}"
echo "Waiting for vLLM at ${SERVER_URL} ..."
for _ in $(seq 1 180); do
    if curl --silent --fail "${SERVER_URL}/v1/models" >/dev/null; then
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "vLLM exited early; see ${BENCHMARK_DIR}/logs/vllm_${MODEL_TYPE}.log" >&2
        exit 1
    fi
    sleep 2
done
curl --silent --fail "${SERVER_URL}/v1/models" >/dev/null || {
    echo "Timed out waiting for vLLM" >&2
    exit 1
}

for model_name in ${MODELS}; do
    for split in ${SPLITS}; do
        if [[ "${split}" == "erase" ]]; then
            split_dir="erased_apple_laptop_unbranding"
        else
            split_dir="retained_apple_laptop_unbranding"
        fi

        images_dir="${IMAGES_ROOT}/${model_name}/${split_dir}"
        if [[ ! -d "${images_dir}" ]]; then
            echo "Skipping missing image directory: ${images_dir}" >&2
            continue
        fi

        common_args=(
            --images-dir "${images_dir}"
            --dataset-csv "${DATASET_CSV}"
            --split "${split}"
            --erased-brand apple
            --model-type "${MODEL_TYPE}"
            --server-url "${SERVER_URL}"
        )

        if [[ "${RUN_BPS}" == "1" ]]; then
            uv run --project "${BENCHMARK_DIR}" python "${BENCHMARK_DIR}/evaluate.py" \
                --mode bps \
                "${common_args[@]}" \
                --output "${RESULTS_DIR}/${model_name}_${split}_bps.jsonl"
        fi

        reference_dir="${IMAGES_ROOT}/${REFERENCE_MODEL}/${split_dir}"
        if [[ "${RUN_VSS}" == "1" && -d "${reference_dir}" ]]; then
            uv run --project "${BENCHMARK_DIR}" python "${BENCHMARK_DIR}/evaluate.py" \
                --mode vss \
                "${common_args[@]}" \
                --reference-dir "${reference_dir}" \
                --output "${RESULTS_DIR}/${model_name}_${split}_vss.jsonl"
        elif [[ "${RUN_VSS}" == "1" ]]; then
            echo "Skipping VSS; missing reference directory: ${reference_dir}" >&2
        fi
    done
done