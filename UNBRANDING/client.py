import argparse
import base64
import json
import os
import logging
from pathlib import Path
from typing import List, Dict, Any
import time
import requests
from PIL import Image
from io import BytesIO
from vlm_outputs import BrandBinaryRecognitionOutput, UnbrandingEvaluationOutput, VisualSimilarityEvaluationOutput, VisualSimilarityEvaluationOutputV2
from pydantic import ValidationError
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms


# ---------- LOGGING ----------
def setup_logging(rank=0, model=""):
    from datetime import datetime
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logger = logging.getLogger(f"client_rank_{rank}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"client_rank_{rank}_{timestamp}_model_{model}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


# ---------- CONFIG ----------
HEADERS = {
    "Authorization": f"Bearer {os.environ.get('API_KEY', '')}",
    "Content-Type": "application/json"
}

MODEL_MAP = {
    "llava": "llava-hf/llava-1.5-7b-hf",
    "nemotron": "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1",
    "gemma3": "google/gemma-3-4b-it",
    "deepseek_tiny": "deepseek-ai/deepseek-vl2-tiny",
    "qwen25vl_7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "smolvlm": "HuggingFaceTB/SmolVLM-Instruct",
}



# ---------- HELPERS ----------
def encode_base64_content_from_file(pil_image, quality: int = 95) -> str:
    """Load image and encode it to base64 JPEG."""
    raw_bytes = BytesIO()
    pil_image.save(raw_bytes, format="JPEG", quality=quality)
    raw_bytes.seek(0)
    return base64.b64encode(raw_bytes.read()).decode()


def get_distributed_params() -> tuple[int, int]:
    """Return process rank and world size for distributed runs."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return rank, world_size


def load_prompts(prompts_file: str) -> Dict[str, Any]:
    """Load JSON prompt template file."""
    with open(prompts_file, 'r') as f:
        return json.load(f)


def create_messages(task: str, system_description: str,
                    format_instructions: List[str], question: str,
                    image_base64: list, mode: str = "qa") -> List[Dict]:
    """Compose system and user message payload for the API."""
    instructions = "\n".join(format_instructions)
    system_content = f"{system_description}\n\n{task}\n\n{instructions}"

    user_content = [{"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64[0]}"}}]

    # For comparison mode, attach both reference and generated images
    if mode == "comparison" and len(image_base64) > 1:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_base64[1]}"}
        })

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]


def query_vllm(messages: List[Dict], model_config: Dict, url: str,
               seed: int, mode: str) -> str:

    if mode == "comparison":
        json_schema = VisualSimilarityEvaluationOutputV2.model_json_schema()
        schema_name = "visual_similarity_eval_outputV2"
    else:
        json_schema = BrandBinaryRecognitionOutput.model_json_schema()
        schema_name = "brand_binary_recognition_output"

    payload = {
        "model": model_config["model_name"],
        "messages": messages,
        "max_tokens": model_config.get("max_tokens", 128),
        "temperature": model_config.get("temperature", 0.15),
        "seed": seed,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": json_schema},
        },
    }

    def send():
        return requests.post(url, headers=HEADERS, json=payload)

    try:
        response = send()
    except requests.exceptions.RequestException:
        time.sleep(5)
        response = send()

    if response.status_code != 200:
        time.sleep(5)
        response = send()
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code} - {response.text}")

    return response.json()["choices"][0]["message"]["content"]


def parse_unlearning_response(response_text: str) -> Dict[str, Any]:
    """Parse JSON response for unbranding evaluation."""
    try:
        data = json.loads(response_text.strip())
        return UnbrandingEvaluationOutput(**data).model_dump()
    except Exception as e:
        return {"error": str(e), "raw_response": response_text}


def parse_recognition_response(response_text: str) -> Dict[str, Any]:
    """Parse JSON response for brand recognition."""
    data = json.loads(response_text.strip())
    return BrandBinaryRecognitionOutput(**data).model_dump()


def parse_visual_similarity_response(response_text: str) -> Dict[str, Any]:
    """Parse JSON response for visual similarity evaluation."""
    try:
        data = json.loads(response_text.strip())
        return VisualSimilarityEvaluationOutputV2(**data).model_dump()
    except Exception as e:
        return {"error": str(e), "raw_response": response_text}


# ---------- MAIN ----------
def parse_args():
    parser = argparse.ArgumentParser(description="Multi-brand / visual-similarity client for VLLM server")
    parser.add_argument("--exp", type=str, required=True, help="Experiment name")
    parser.add_argument("--gt_imgs_dir", type=str, required=True, help="Directory with GT images")
    parser.add_argument("--gen_imgs_dir", type=str, required=False, help="Directory with generated images")
    parser.add_argument("--model-type", "-m", type=str, default="llava", choices=list(MODEL_MAP.keys()))
    parser.add_argument("--prompts-file", type=str, default="prompts.json", help="JSON file with prompts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--server-url", type=str, default="http://localhost:8000", help="VLLM server URL")
    return parser.parse_args()


def main():
    args = parse_args()
    rank, world_size = get_distributed_params()
    logger = setup_logging(rank, args.model_type)

    url = f"{args.server_url}/v1/chat/completions"
    logger.info(f"Using server URL: {url}")

    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)

    # Load prompt template
    try:
        prompts_data = load_prompts(args.prompts_file)
    except Exception as e:
        logger.error(f"Error loading prompts file {args.prompts_file}: {e}")
        return

    model_type = args.model_type
    model_config = prompts_data["model_configs"].get(model_type)
    if not model_config:
        logger.error(f"Model '{model_type}' not found in prompts file.")
        return

    template_type = prompts_data.get("template_type", "qa").lower()
    logger.info(f"Template type: {template_type}")

    gt_imgs = sorted(os.listdir(args.gt_imgs_dir))
    gen_imgs = sorted(os.listdir(args.gen_imgs_dir)) if template_type == "comparison" else []
    logger.info(f"Loaded {len(gt_imgs)} ground-truth images")

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"{args.exp}_{model_type}_rank_{rank}.jsonl"

    for idx, gt_filename in enumerate(gt_imgs):
        if idx % world_size != rank:
            continue

        gt_path = os.path.join(args.gt_imgs_dir, gt_filename)
        gt_pil_image = Image.open(gt_path).convert("RGB")
        gt_b64 = encode_base64_content_from_file(gt_pil_image)
        image_b64_list = [gt_b64]

        if template_type == "comparison" and len(gen_imgs) > idx:
            gen_path = os.path.join(args.gen_imgs_dir, gt_filename)
            gen_pil_image = Image.open(gen_path).convert("RGB")
            gen_b64 = encode_base64_content_from_file(gen_pil_image)
            image_b64_list = [gt_b64, gen_b64]
            inputs = processor(images=[gt_pil_image, gen_pil_image], return_tensors="pt")

        task = prompts_data["task"]
        system_description = prompts_data.get("system_description", "")
        format_instructions = prompts_data["format_instructions"]

        # Handle visual similarity comparison
        if template_type == "comparison":
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
            image_features = F.normalize(image_features, p=2, dim=-1)
            cosine_similarity = (image_features[0] * image_features[1]).sum().item()
            cosine_distance = 1 - cosine_similarity

            if cosine_distance > 0.2:
                record = {
                        "image_pair": gt_filename,
                        "result": {
                            "similarity_score": 0,
                            "explanation": "CLIP threshold" 
                            },
                        "model_type": "clip",
                        "rank": rank,
                        "template_type": template_type,
                    }
            else:
            
                question = "Evaluate the visual similarity between these two images."
                try:
                    messages = create_messages(task, system_description,
                                            format_instructions, question,
                                            image_b64_list, mode="comparison")
                    response = query_vllm(messages, model_config, url, args.seed, "comparison")
                    parsed = parse_visual_similarity_response(response)

                    record = {
                        "image_pair": gt_filename,
                        "result": parsed,
                        "model_type": model_type,
                        "rank": rank,
                        "template_type": template_type,
                    }
                    logger.info(f"Processed {gt_filename} with score {parsed.get('final_similarity_score', 'N/A')}")
                except Exception as inner_e:
                    logger.error(f"Error on visual similarity eval for {gt_filename}: {inner_e}")
                    record = {
                        "image_pair": gt_filename,
                        "error": str(inner_e),
                        "model_type": model_type,
                        "rank": rank,
                        "template_type": template_type,
                    }

        else:
            # Default QA path for brand presence detection
            brand = gt_filename.split("_")[0].title()
            results_per_brand = {}
            questions_map = {f"Q{idx+1}": q for idx, q in enumerate(prompts_data.get("questions", []))}
            brand_results = {}

            for q_idx, q_template in enumerate(prompts_data.get("questions", [])):
                question = q_template.replace("{BRAND}", brand)
                try:
                    messages = create_messages(task, system_description,
                                                format_instructions, question,
                                                image_b64_list, mode=template_type)
                    response = query_vllm(messages, model_config, url, args.seed, template_type)
                    parsed = parse_recognition_response(response)
                    brand_results[f"Q{q_idx+1}"] = {"response": parsed}
                except Exception as inner_e:
                    logger.error(f"Error on {brand} Q{q_idx+1} for {gt_filename}: {inner_e}")
                    brand_results[f"Q{q_idx+1}"] = {"error": str(inner_e)}

            results_per_brand[brand] = brand_results
            record = {
                "image": gt_filename,
                "brands": results_per_brand,
                "questions": questions_map,
                "model_type": model_type,
                "rank": rank,
                "template_type": template_type,
            }

        # Write result to output file
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Processing complete.")


if __name__ == "__main__":
    main()
