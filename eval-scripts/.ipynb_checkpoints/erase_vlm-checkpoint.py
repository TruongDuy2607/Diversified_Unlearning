import os
import json
import time
import re
import csv
import argparse
import base64
from pathlib import Path
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from openai import OpenAI
import pandas as pd


def ensure_dir(directory):
    """Make sure the directory exists."""
    Path(directory).mkdir(parents=True, exist_ok=True)

def load_image(image_path):
    """Load an image from path."""
    image = Image.open(image_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    return image

def encode_image_into_base64(image):
    """Convert PIL image to base64."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def read_prompt_template(file_path):
    """Read prompt template from file."""
    try:
        with open(file_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: Could not find prompt file: {file_path}")
        return None

def call_qwen_vl(reference_image_base64, image_base64, prompt, max_retries=3):
    """Call Qwen VL model using OpenAI client format."""
    client = OpenAI(
        api_key='EMPTY',
        base_url='http://127.0.0.1:2607/v1',
    )
    
    # Get available model name
    try:
        model = client.models.list().data[0].id
    except:
        model = "Qwen/Qwen2.5-VL-72B-Instruct"  # Default fallback
    
    for attempt in range(max_retries):
        try:
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "Here is reference image: "},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{reference_image_base64}", "detail": "high"}},
                        {"type": "text", "text": "Here is generated image: "},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}", "detail": "high"}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=300
            )
            
            return completion.choices[0].message.content
        
        except Exception as e:
            print(f"Error calling Qwen VL API: {e}. Retrying... ({attempt+1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(1)
    
    return None

def label_to_ref_filename(label_str):
    name = label_str.strip().lower().replace(' ', '_')
    return f"{name}.jpg"

def extract_score(content):
    pattern = r"(score|Score):\s*(\d+)"
    score_matches = re.findall(pattern, content)
    if not score_matches:
        return None
    try:
        score = int(score_matches[0][1])
    except Exception:
        return None
    if score not in [0, 1, 2, 3, 4]:
        score = min(max(score, 0), 4)
    return score

def evaluate_from_csv(csv_file, target_dir, ref_images_dir, output_dir):
    """
    Đọc CSV đầu vào, duyệt từng hàng, lấy reference theo label_str và target theo case_number_0.png.
    Xuất 2 file CSV: (1) bản CSV gốc + cột score, output; (2) summary theo label_str và overall.
    """
    ensure_dir(output_dir)
    # Load prompt template (optional)
    user_prompt = read_prompt_template('prompts/user_prompt_character_erase.txt')
    if not user_prompt:
        user_prompt = " "

    # Read input CSV
    df = pd.read_csv(csv_file)
    if 'case_number' not in df.columns or 'label_str' not in df.columns:
        raise ValueError('CSV must contain columns case_number and label_str')

    scores = []
    outputs = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Evaluating from CSV'):
        case_number = row['case_number']
        label_str = str(row['label_str'])
        target_file = os.path.join(target_dir, f"{case_number}_0.png")

        # Resolve reference image path
        ref_filename = label_to_ref_filename(label_str)
        reference_image_path = os.path.join(ref_images_dir, ref_filename)
        if not os.path.exists(reference_image_path):
            # Try .jpeg fallback
            ref_filename_jpeg = ref_filename[:-4] + '.jpeg'
            reference_image_path = os.path.join(ref_images_dir, ref_filename_jpeg)
        if not os.path.exists(reference_image_path):
            print(f"WARNING: Reference image not found for label {label_str}: {ref_filename}")
            scores.append(None)
            outputs.append(None)
            continue

        if not os.path.exists(target_file):
            print(f"WARNING: Target image not found: {target_file}")
            scores.append(None)
            outputs.append(None)
            continue

        try:
            # Load and encode images
            reference_image = load_image(reference_image_path)
            reference_image_base64 = encode_image_into_base64(reference_image)
            target_image = load_image(target_file)
            target_image_base64 = encode_image_into_base64(target_image)

            # Call model with retries
            max_retries = 3
            content = None
            for attempt in range(max_retries):
                try:
                    content = call_qwen_vl(reference_image_base64, target_image_base64, user_prompt)
                    if content:
                        break
                except Exception as e:
                    print(f"Error evaluating {Path(target_file).name}: {e}. Retrying... ({attempt+1}/{max_retries})")
                    time.sleep(1)
            if not content:
                scores.append(None)
                outputs.append(None)
                continue
            score = extract_score(content)
            scores.append(score)
            outputs.append(content)
        except Exception as e:
            print(f"Error processing case {case_number}: {e}")
            scores.append(None)
            outputs.append(None)

    # Attach results to DataFrame
    df['score'] = scores
    df['output'] = outputs

    # Save per-row results CSV
    base = Path(csv_file).stem
    results_csv = os.path.join(output_dir, f"{base}_scored.csv")
    df.to_csv(results_csv, index=False)

    # Build summary per concept and overall (normalized by 4)
    summary_rows = []
    grouped = df.groupby('label_str', dropna=False)
    for label, group in grouped:
        valid_scores = [s for s in group['score'].tolist() if isinstance(s, (int, float))]
        num_samples = len(valid_scores)
        if num_samples == 0:
            avg_norm = None
        else:
            avg_norm = sum(valid_scores) / 4.0 / num_samples
        summary_rows.append({'concept': label, 'num_samples': num_samples, 'score': avg_norm})

    # Overall
    all_valid = [s for s in df['score'].tolist() if isinstance(s, (int, float))]
    if len(all_valid) == 0:
        overall = {'concept': 'OVERALL', 'num_samples': 0, 'score': None}
    else:
        overall = {'concept': 'OVERALL', 'num_samples': len(all_valid), 'score': sum(all_valid) / 4.0 / len(all_valid)}
    summary_rows.append(overall)

    summary_df = pd.DataFrame(summary_rows, columns=['concept', 'num_samples', 'score'])
    summary_csv = os.path.join(output_dir, f"{base}_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    print(f"Saved per-row results to: {results_csv}")
    print(f"Saved summary to: {summary_csv}")
    return results_csv, summary_csv

def main():
    parser = argparse.ArgumentParser(description="Evaluate images using CSV mapping of case_number and label_str")
    parser.add_argument("--output_dir", type=str, default="", help="Directory to save results")
    parser.add_argument("--csv_file", type=str, required=True, help="Input CSV with columns including case_number and label_str")
    parser.add_argument("--target_dir", type=str, required=True, help="Directory containing target images named {case_number}_0.png")
    parser.add_argument("--ref_images_dir", type=str, required=True, help="Directory containing reference images like iron_man.jpg, mario.jpg")
    args = parser.parse_args()

    OUTPUT_DIR = args.output_dir
    ensure_dir(OUTPUT_DIR)

    print("\n=== Starting CSV-driven evaluation ===")
    evaluate_from_csv(args.csv_file, args.target_dir, args.ref_images_dir, OUTPUT_DIR)
    print("\n=== Evaluation completed ===")

if __name__ == "__main__":
    main() 