import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from image_generation.generate import generate_one
from image_generation.pipelines import load_pipeline
from image_generation.utils import read_data, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate images using FLUX, SDXL (incl. custom), or SD 1.4 via Diffusers."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "sdxl",
            "sd14",
            "sd35-large",
            "flux1-schnell",
            "flux1-dev",
            "qwen-image",
        ],
        required=True,
        help="Model key.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Optional Hugging Face model id (required for sdxl-custom).",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        required=True,
        default="configs/adidas.json",
        help="Path to a text file with prompts.",
    )
    parser.add_argument(
        "--prompt_set",
        type=str,
        nargs="+",
        default=["directed"],
        help="Set of prompts to use.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save images."
    )

    parser.add_argument("--seed", type=int, default=42, help="Global seed (optional).")
    parser.add_argument(
        "--log_dir", type=str, default="logs", help="Directory to save logs."
    )
    return parser.parse_args()


def save_config(path, config):
    with open(path, "w") as f:
        json.dump(config, f, indent=4)


def main() -> None:
    RANK = int(os.environ.get("RANK", "0"))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))

    args = parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(args.log_dir, f"{args.model}_{timestamp}_rank_{RANK}.log")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", filename=log_filename, filemode="w")
    logger = logging.getLogger(__name__)
    logger.info("Starting image generation with args: %s", args)

    pipe = load_pipeline(args.model, args.model_id)

    df_prompts = pd.read_csv(args.prompts_file)

    exp_params = {
        "model": args.model,
        "prompts_file": args.prompts_file,
        "seed": args.seed,
    }

    exp_name = "_".join(
        f"{key}-{exp_params[key]}" for key in ["model", "seed", "prompts_file"]
    )

    exp_dir = os.path.join(args.output_dir, exp_name)
    exp_images_dir = os.path.join(exp_dir, "images")
    os.makedirs(exp_images_dir, exist_ok=True)
    save_config(os.path.join(exp_dir, "config.json"), exp_params)
    negative_prompt = " " if args.model == "qwen-image" else None
    global_idx = 0
    
    for global_idx, row in df_prompts.iterrows():
        out_path = os.path.join(exp_images_dir, row["filename"])
        if global_idx % WORLD_SIZE != RANK:
            logger.info("Skipping rank")
            continue

        if Path(out_path).exists():
            logger.info("Skipping existing: %s", out_path)
            continue
        try:
            generation_start_time = datetime.now()
            logger.info(
                f"Generating image {global_idx} for prompt: {row['prompt']} at {generation_start_time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            set_seed(row["seed"])
            image = generate_one(
                pipe,
                prompt=row["prompt"],
                negative_prompt=negative_prompt,
                seed=row["seed"],
            )

            save_time = datetime.now()
            image.save(out_path)
            logger.info(
                "Saved: %s at %s", out_path, save_time.strftime("%Y-%m-%d %H:%M:%S")
            )
        except Exception as exc:
            logging.exception("Failed for prompt '%s': %s", row["prompt"], exc)


if __name__ == "__main__":
    main()
