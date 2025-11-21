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

# Setup directories
# OUTPUT_DIR = "k-results/uce"
# SOURCE_IMAGES_DIR = "k-images/SD-v1-4/retain_henry_cavill"  # Directory containing source images
# TARGET_IMAGES_DIR = "k-images/uce/retain_henry_cavill"  # Directory containing target images to compare

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
        base_url='http://127.0.0.1:1309/v1',
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

def compare_images_between_folders(reference_image_path, target_dir, output_dir):
    """
    So sánh một ảnh reference cố định với tất cả ảnh trong folder target_dir.
    """
    # Create output directory
    output_dir_ = os.path.join(output_dir, "results")
    ensure_dir(output_dir_)
    
    # Check if reference image exists
    if not os.path.exists(reference_image_path):
        print(f"ERROR: Reference image not found: {reference_image_path}")
        return
    # Check if target directory exists
    if not os.path.exists(target_dir):
        print(f"ERROR: Target directory not found: {target_dir}")
        return
    
    # Load user prompt
    user_prompt = read_prompt_template('prompts/user_prompt_celeb_erase.txt')
    if not user_prompt:
        # Fallback prompt if file not found
        user_prompt = " "
    # Load reference image ONCE
    try:
        reference_image = load_image(reference_image_path)
        reference_image_base64 = encode_image_into_base64(reference_image)
    except Exception as e:
        print(f"ERROR: Failed to load reference image: {e}")
        return
    
    # Get list of files in target directory
    target_files = [f for f in os.listdir(target_dir) 
                   if os.path.isfile(os.path.join(target_dir, f)) and 
                   f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Lists to track all results
    all_results = []
    
    # Process each file in target directory
    for target_filename in tqdm(target_files, desc="Comparing images"):
        target_file = os.path.join(target_dir, target_filename)
        try:
            # Load target image
            target_image = load_image(target_file)
            target_image_base64 = encode_image_into_base64(target_image)
            # Call Qwen VL API
            max_retries = 3
            retry_count = 0
            content = None
            while retry_count < max_retries:
                try:
                    # So sánh reference_image với target_image
                    content = call_qwen_vl(reference_image_base64, target_image_base64, user_prompt)
                    if content:
                        break
                    retry_count += 1
                except Exception as e:
                    print(f"Error evaluating {target_filename}: {e}. Retrying...")
                    retry_count += 1
                    time.sleep(1)
            # If successful, process the result
            if content:
                # Extract score (0-4)
                pattern = r"(score|Score):\s*(\d+)"
                score_matches = re.findall(pattern, content)
                if score_matches:
                    score = int(score_matches[0][1])
                    # Validate score is 0-4
                    if score not in [0, 1, 2, 3, 4]:
                        print(f"Invalid score for {target_filename}: {score}. Should be 0-4.")
                        score = min(max(score, 0), 4)  # Clamp to 0-4
                    # Store result
                    result = {
                        "filename": target_filename,
                        "score": score,
                        "content": content
                    }
                    all_results.append(result)
                    # Save individual result
                    result_filename = f"{os.path.splitext(target_filename)[0]}_eval.json"
                    with open(os.path.join(output_dir, result_filename), 'w') as f:
                        json.dump(result, f, indent=2)
                else:
                    print(f"Failed to extract score from Qwen VL response for {target_filename}.")
            else:
                print(f"Failed to evaluate {target_filename} after {max_retries} attempts")
        except Exception as e:
            print(f"Error processing file {target_filename}: {e}")
    # Save all results
    with open(os.path.join(output_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    # Calculate overall stats
    if all_results:
        scores = [item['score'] for item in all_results]
        avg_score = sum(scores) / len(scores)
        erasure_success_rate = sum(1 for score in scores if score <= 2) / len(scores)  # Count scores 0-2 as successful erasure
        overall_summary = {
            "total_images_evaluated": len(scores),
            "average_score": avg_score,
            "score_distribution": {str(score): scores.count(score) for score in range(5)},
            "erasure_success_rate": erasure_success_rate
        }
        with open(os.path.join(output_dir, 'overall_summary.json'), 'w') as f:
            json.dump(overall_summary, f, indent=2)
        # Save results in CSV format for easier analysis
        with open(os.path.join(output_dir, 'evaluation_results.csv'), "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["filename", "score", "content"])
            for result in all_results:
                writer.writerow([result["filename"], result["score"], result["content"]])
        # Save summary in CSV format
        with open(os.path.join(output_dir, 'summary.csv'), "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["total_images", "average_score", "erasure_success_rate"])
            writer.writerow([
                len(scores),
                f"{avg_score:.2f}",
                f"{erasure_success_rate:.2%}"
            ])
        print("\n=== Overall Image Comparison Results ===")
        print(f"Total images evaluated: {len(scores)}")
        print(f"Average score: {avg_score:.2f} (lower is better for erasure)")
        print(f"Erasure success rate (scores 0-2): {erasure_success_rate:.2%}")
        print(f"Score distribution: {overall_summary['score_distribution']}")
    return overall_summary if all_results else None

def main():
    parser = argparse.ArgumentParser(description="Compare a fixed reference image to all images in a folder")
    parser.add_argument("--output_dir", type=str, default="k-results/uce", help="Directory to save results")
    parser.add_argument("--reference_image", type=str, default="henry_cavill.jpeg", help="Path to the fixed reference image")
    parser.add_argument("--target_dir", type=str, default="", help="Directory containing target images to compare")
    args = parser.parse_args()
    # Create output directories
    OUTPUT_DIR = args.output_dir
    ensure_dir(OUTPUT_DIR)
    ensure_dir(os.path.join(OUTPUT_DIR, "results"))
    # Run comparison between reference image and all images in target_dir
    print("\n=== Starting image comparison between reference and target folder ===")
    results = compare_images_between_folders(args.reference_image, args.target_dir, OUTPUT_DIR)
    if results:
        print("\n=== Image Comparison completed! ===")
        print("\nFind the results in:")
        print(f"- Overall Summary: {os.path.join(OUTPUT_DIR, 'results', 'overall_summary.json')}")
        print(f"- Summary CSV: {os.path.join(OUTPUT_DIR, 'results', 'summary.csv')}")
        print(f"- All Results CSV: {os.path.join(OUTPUT_DIR, 'results', 'evaluation_results.csv')}")
    else:
        print("\n=== No results were generated ===")

if __name__ == "__main__":
    main() 