import argparse
import base64
import json
import re
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from pydantic import BaseModel, Field
from transformers import CLIPModel, CLIPProcessor


MODEL_MAP = {
    "llava": "llava-hf/llava-1.5-7b-hf",
    "nemotron": "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1",
    "gemma3": "google/gemma-3-4b-it",
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


class BrandPresenceOutput(BaseModel):
    answer: str = Field(pattern="^(YES|NO)$")
    score: int = Field(ge=0, le=5)


class VisualSimilarityOutput(BaseModel):
    explanation: str = Field(max_length=120)
    similarity_score: int = Field(ge=0, le=10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Diversified-Unlearn images with UNBRANDING BPS or VSS."
    )
    parser.add_argument("--mode", choices=["bps", "vss"], required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--dataset-csv", type=Path, required=True)
    parser.add_argument("--split", choices=["erase", "retain"], required=True)
    parser.add_argument("--erased-brand", default="apple")
    parser.add_argument("--model-type", choices=MODEL_MAP, default="gemma3")
    parser.add_argument("--prompts-file", type=Path)
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images", type=int)
    return parser.parse_args()


def load_split_metadata(
    dataset_csv: Path, split: str, erased_brand: str
) -> pd.DataFrame:
    metadata = pd.read_csv(dataset_csv)
    required = {"brand", "seed", "prompt", "prompt_set", "filename"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    is_erased_brand = metadata["brand"].str.casefold() == erased_brand.casefold()
    metadata = metadata[is_erased_brand if split == "erase" else ~is_erased_brand]
    return metadata.reset_index(drop=True)


def image_case_number(path: Path) -> int:
    match = re.match(r"^(\d+)(?:_\d+)?$", path.stem)
    if not match:
        raise ValueError(f"Unsupported image filename: {path.name}")
    return int(match.group(1))


def discover_images(images_dir: Path, max_images: int | None) -> list[Path]:
    images = [
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    images.sort(key=lambda path: (image_case_number(path), path.name))
    return images[:max_images] if max_images is not None else images


def metadata_for_image(metadata: pd.DataFrame, image_path: Path) -> dict:
    case_number = image_case_number(image_path)
    if case_number >= len(metadata):
        raise IndexError(
            f"Image {image_path.name} has case {case_number}, but split has "
            f"only {len(metadata)} rows"
        )
    row = metadata.iloc[case_number]
    return {
        "case_number": case_number,
        "brand": str(row["brand"]),
        "prompt_set": str(row["prompt_set"]),
        "prompt": str(row["prompt"]),
        "original_seed": int(row["seed"]),
        "original_filename": str(row["filename"]),
    }


def encode_image(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode()


def query_vllm(
    server_url: str,
    model_name: str,
    system_prompt: str,
    question: str,
    images: list[Image.Image],
    output_model: type[BaseModel],
    seed: int,
) -> BaseModel:
    user_content = [{"type": "text", "text": question}]
    user_content.extend(
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image)}"},
        }
        for image in images
    )
    schema = output_model.model_json_schema()
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 128,
        "temperature": 0.15,
        "seed": seed,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": output_model.__name__, "schema": schema},
        },
    }
    response = requests.post(
        f"{server_url.rstrip('/')}/v1/chat/completions", json=payload, timeout=180
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return output_model.model_validate_json(content)


def load_prompt_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def bps_system_prompt(config: dict) -> str:
    return "\n\n".join(
        [config.get("system_description", ""), config.get("task", "")]
        + config.get("format_instructions", [])
    )


def evaluate_bps(
    image_path: Path,
    image_metadata: dict,
    config: dict,
    args: argparse.Namespace,
) -> dict:
    image = Image.open(image_path).convert("RGB")
    brand = image_metadata["brand"].replace("_", " ").title()
    responses = []
    for question_template in config["questions"]:
        question = question_template.replace("{BRAND}", brand)
        output = query_vllm(
            args.server_url,
            MODEL_MAP[args.model_type],
            bps_system_prompt(config),
            question,
            [image],
            BrandPresenceOutput,
            args.seed,
        )
        responses.append(
            {"question": question, "answer": output.answer, "score": output.score}
        )

    mean_score = sum(item["score"] for item in responses) / len(responses)
    recognition_rate = sum(item["answer"] == "YES" for item in responses) / len(
        responses
    )
    return {
        **image_metadata,
        "image": image_path.name,
        "responses": responses,
        "bps": mean_score,
        "normalized_bps": mean_score / 5,
        "recognition_rate": recognition_rate,
    }


def evaluate_vss(
    image_path: Path,
    reference_path: Path,
    image_metadata: dict,
    config: dict,
    args: argparse.Namespace,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
) -> dict:
    generated = Image.open(image_path).convert("RGB")
    reference = Image.open(reference_path).convert("RGB")
    inputs = clip_processor(images=[reference, generated], return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    features = F.normalize(features, p=2, dim=-1)
    clip_distance = 1 - (features[0] * features[1]).sum().item()

    if clip_distance > 0.2:
        score = 0
        explanation = "CLIP distance exceeded 0.2"
        evaluator = "clip-threshold"
    else:
        system_prompt = "\n\n".join(
            [config.get("system_description", ""), config.get("task", "")]
            + config.get("format_instructions", [])
        )
        output = query_vllm(
            args.server_url,
            MODEL_MAP[args.model_type],
            system_prompt,
            "Evaluate the visual similarity between the reference image first "
            "and the unlearned-model image second.",
            [reference, generated],
            VisualSimilarityOutput,
            args.seed,
        )
        score = output.similarity_score
        explanation = output.explanation
        evaluator = args.model_type

    return {
        **image_metadata,
        "image": image_path.name,
        "reference_image": reference_path.name,
        "clip_distance": clip_distance,
        "vss": score,
        "normalized_vss": score / 10,
        "explanation": explanation,
        "evaluator": evaluator,
    }


def summarize(records: list[dict], mode: str) -> dict:
    metric = "bps" if mode == "bps" else "vss"
    grouped: dict[str, dict[str, list[float]]] = {
        "brand": defaultdict(list),
        "prompt_set": defaultdict(list),
    }
    values = []
    for record in records:
        if "error" in record:
            continue
        value = float(record[metric])
        values.append(value)
        grouped["brand"][record["brand"]].append(value)
        grouped["prompt_set"][record["prompt_set"]].append(value)

    def group_summary(groups: dict[str, list[float]]) -> dict:
        return {
            key: {"count": len(group), "mean": sum(group) / len(group)}
            for key, group in sorted(groups.items())
        }

    brand_summary = group_summary(grouped["brand"])
    summary = {
        "metric": metric,
        "processed": len(values),
        "errors": len(records) - len(values),
        "mean": sum(values) / len(values) if values else None,
        "normalized_mean": (
            sum(values) / len(values) / (5 if mode == "bps" else 10)
            if values
            else None
        ),
        "macro_brand_mean": (
            sum(group["mean"] for group in brand_summary.values())
            / len(brand_summary)
            if brand_summary
            else None
        ),
        "by_brand": brand_summary,
        "by_prompt_set": group_summary(grouped["prompt_set"]),
    }
    if mode == "bps":
        summary["recognition_rate"] = (
            sum(record["recognition_rate"] for record in records if "error" not in record)
            / len(values)
            if values
            else None
        )
    else:
        summary["clip_rejection_rate"] = (
            sum(
                record["evaluator"] == "clip-threshold"
                for record in records
                if "error" not in record
            )
            / len(values)
            if values
            else None
        )
    return summary


def main() -> None:
    args = parse_args()
    if args.mode == "vss" and args.reference_dir is None:
        raise ValueError("--reference-dir is required for VSS")

    default_config = "vlm_bps.json" if args.mode == "bps" else "vlm_vss.json"
    prompts_file = args.prompts_file or Path(__file__).parent / "configs" / default_config
    config = load_prompt_config(prompts_file)
    metadata = load_split_metadata(args.dataset_csv, args.split, args.erased_brand)
    images = discover_images(args.images_dir, args.max_images)
    if not images:
        raise ValueError(f"No images found in {args.images_dir}")

    clip_model = clip_processor = None
    if args.mode == "vss":
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with args.output.open("w", encoding="utf-8") as output_file:
        for image_path in images:
            image_metadata = metadata_for_image(metadata, image_path)
            try:
                if args.mode == "bps":
                    record = evaluate_bps(image_path, image_metadata, config, args)
                else:
                    reference_path = args.reference_dir / image_path.name
                    if not reference_path.is_file():
                        raise FileNotFoundError(f"Missing reference: {reference_path}")
                    record = evaluate_vss(
                        image_path,
                        reference_path,
                        image_metadata,
                        config,
                        args,
                        clip_model,
                        clip_processor,
                    )
            except Exception as error:
                record = {
                    **image_metadata,
                    "image": image_path.name,
                    "error": str(error),
                }
            records.append(record)
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()

    summary = summarize(records, args.mode)
    summary.update(
        {
            "mode": args.mode,
            "split": args.split,
            "images_dir": str(args.images_dir),
            "reference_dir": str(args.reference_dir) if args.reference_dir else None,
        }
    )
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()