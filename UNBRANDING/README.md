# From Unlearning to UNBRANDING: A Benchmark for Trademark‑Safe Text‑to‑Image Generation


[![arXiv](https://img.shields.io/badge/arXiv-2512.13953-b31b1b.svg)](https://arxiv.org/abs/2512.13953)
[![Website](https://img.shields.io/badge/Website-Project%20Page-22c55e.svg)](https://gmum.github.io/UNBRANDING/)
[![Stars gmum/VeGaS](https://img.shields.io/github/stars/gmum/UNBRANDING?style=social)](https://gmum.github.io/UNBRANDING)

![Teaser](images/teaser_ub_v4.png)

> Benchmark and task for trademark‑safe text‑to‑image generation. We introduce unbranding — fine‑grained removal of both explicit logos and implicit trade‑dress while preserving object and scene semantics — together with a dataset and a VLM‑QA evaluation protocol.


## Authors
Dawid Malarz, Artur Kasymov, Filip Manjak, Maciej Zięba, Przemysław Spurek

## Abstract
Modern text‑to‑image diffusion models can faithfully reproduce trademarks. Prior unlearning works target general concepts (e.g., styles, celebrities) and miss brand‑specific identifiers. Brand recognition is multi‑dimensional, spanning explicit logos and distinctive structural features (e.g., a car’s front grille). We define unbranding as the fine‑grained removal of both trademarks and subtle trade‑dress while preserving semantic coherence. We introduce a benchmark dataset and a VLM‑based QA metric that probes for both explicit and implicit brand signals, going beyond classic logo detectors. Our analysis shows newer models tend to synthesize brands more readily, underscoring the urgency of unbranding. Results validated by our metric indicate that unbranding is a distinct, practically relevant problem requiring specialized techniques.

## Highlights
- Task: fine‑grained removal of brand identifiers while preserving object fidelity.
- Benchmark: dataset spanning explicit logos and trade‑dress cues.
- Metric: VLM‑based QA probing both explicit and implicit brand signals.
- Motivation: newer models (e.g., SDXL, FLUX) reproduce brands more readily than older ones.

## Results Snapshot
Trade‑off between fidelity and removal. Baseline preserves structure but often fails to remove brands; ESD removes brands but alters semantics. Effective unbranding must achieve both.

![Trade‑off](images/fig4.png)

## Installation

### Prerequisites
- Python 3.13+
- CUDA-compatible GPU (for image generation and VLM inference)
- [uv](https://docs.astral.sh/uv/) package manager

### Setup with uv

1. Clone the repository:
```bash
git clone https://github.com/gmum/UNBRANDING.git
cd UNBRANDING
```

2. Install dependencies using uv:
```bash
uv sync
```

This will create a virtual environment in `.venv` and install all required packages including:
- PyTorch 2.8.0 with CUDA support
- Diffusers 0.35.2 (Stable Diffusion, FLUX models)
- Transformers 4.57.1 (HuggingFace models)
- VLLM 0.11.0 (optimized VLM inference)
- Streamlit 1.51.0 (annotation UI)
- Ray 2.51.1 (distributed processing)

3. Activate the environment:
```bash
source .venv/bin/activate
```

Or run commands directly with uv:
```bash
uv run python <script.py>
```

## Quick Start

### 1. Generate Images

```bash
# Using Stable Diffusion XL
uv run python generate_images.py --model sdxl --prompts configs/vlm_vss.json --output-dir output/sdxl --seed 42

# Using FLUX.1 Schnell (fast variant)
uv run python generate_images.py --model flux-schnell --prompts configs/vlm_bps.json --output-dir output/flux

# Using Stable Diffusion 3.5 Large
uv run python generate_images.py --model sd35 --prompts configs/vlm_vss.json --output-dir output/sd35
```

**Supported Models:**
- `sd14` - Stable Diffusion v1.4
- `sdxl` - Stable Diffusion XL
- `sd35` - Stable Diffusion 3.5 Large
- `flux-schnell` - FLUX.1 Schnell
- `flux-dev` - FLUX.1 Dev
- `qwen-image` - Qwen Image
- `custom` - Any HuggingFace model (use with `--model-id`)

### 2. Evaluate with VLM-QA

Start VLLM server (in separate terminal):
```bash
uv run vllm serve llava-hf/llava-1.5-7b-hf --port 8000
```

Run VLM evaluation:
```bash
uv run python client.py \
  --gt_imgs_dir data/ground_truth \
  --gen_imgs_dir output/sdxl \
  --model-type llava \
  --output results/evaluation.json
```

### Evaluate Diversified-Unlearn outputs

The adapter in `evaluate.py` supports images stored as:

```text
images-gen/unbranding/<model>/erased_apple_laptop_unbranding/<case>_0.png
images-gen/unbranding/<model>/retained_apple_laptop_unbranding/<case>_0.png
```

It maps each numeric case back to `brand`, `prompt_set`, and the original
UNBRANDING metadata in `data/UNBRANDING/unbranding_v1.csv`. BPS therefore asks
about Apple for the erase split and the actual source brand for the retain split.

From the parent Diversified-Unlearn repository, create the benchmark environment:

```bash
uv sync --project UNBRANDING
```

For VSS, first generate the SD 1.4 reference images with the same converted
prompts, seed, and sampling settings. `run_unbranding.sh` includes the commented
command `generate_evaluation_images "SD-v1-4" 0`, which writes the expected
reference layout under `images-gen/unbranding/SD-v1-4/`.

Run BPS and VSS for ESD and ESD-diverse:

```bash
bash UNBRANDING/eval_diversified_unbranding.sh
```

The script starts a LLaVA vLLM server and evaluates every currently available
image for ESD, ESD-diverse, and the SD 1.4 reference when present. VSS is skipped
with a warning when its SD 1.4 reference directory is missing. Per-image JSONL
and aggregate summaries are written to:

```text
UNBRANDING/results/diversified-unlearn/
```

Common overrides:

```bash
# BPS only while retaining-image generation is still in progress
RUN_VSS=0 bash UNBRANDING/eval_diversified_unbranding.sh

# Evaluate only completed erasure images
SPLITS=erase bash UNBRANDING/eval_diversified_unbranding.sh

# Select models, server GPU, or a different supported VLM
MODELS="esd esd-level-1" SERVER_GPU=0 MODEL_TYPE=gemma3 \
  VLM_MODEL=google/gemma-3-4b-it \
  bash UNBRANDING/eval_diversified_unbranding.sh
```

Summaries include the per-image mean, macro-average across brands, breakdowns by
brand and `prompt_set`, BPS recognition rate, and CLIP rejection rate for VSS.
For BPS, lower is better on the erase split. For retain BPS, compare against the
SD 1.4 reference per brand. For VSS, higher is better on both splits.

**Supported VLM Types:**
- `llava` - LLaVA 1.5 7B
- `nemotron` - NVIDIA Nemotron Nano VL 8B
- `gemma3` - Google Gemma-3 4B IT


## Citation
Please cite our work if you find it useful:

```
@misc{malarz2025unlearningunbrandingbenchmarktrademarksafe,
  title={From Unlearning to UNBRANDING: A Benchmark for Trademark-Safe Text-to-Image Generation},
  author={Dawid Malarz and Artur Kasymov and Filip Manjak and Maciej Zięba and Przemysław Spurek},
  year={2025},
  eprint={2512.13953},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2512.13953},
}
```

## Acknowledgments
The project page is adapted from the Academic Project Page Template (inspired by Nerfies). Images in this repository are for research/illustration.