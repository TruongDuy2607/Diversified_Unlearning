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
        model = "Qwen/Qwen2.5-VL-32B-Instruct"  # Default fallback
    
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


def save_results(all_results, output_dir):
    """Save evaluation results to files."""
    # Calculate overall stats
    if all_results:
        # Sort results by filename
        all_results.sort(key=lambda x: x['filename'])
        
        scores = [item['score'] for item in all_results]
        avg_score = sum(scores) / len(scores)
        erasure_success_rate = sum(1 for score in scores if score <= 2) / len(scores)  # Count scores 0-2 as successful erasure
        
        score_distribution = {str(score): scores.count(score) for score in range(5)}
        
        # Save results in CSV format (sorted by filename)
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
        
        print("\nOverall Image Comparison Results")
        print(f"1. Total images evaluated: {len(scores)}")
        print(f"2. Average score: {avg_score:.2f} (lower is better for erasure)")
        print(f"3. Erasure success rate (scores 0-2): {erasure_success_rate:.2%}")
        print(f"4. Score distribution: {score_distribution}")
        
        return {
            "total_images_evaluated": len(scores),
            "average_score": avg_score,
            "score_distribution": score_distribution,
            "erasure_success_rate": erasure_success_rate
        }
    
    return None


def evaluate_image_pair(reference_base64, target_base64, filename, user_prompt, output_dir):
    """Evaluate a single image pair and return the result."""
    max_retries = 3
    retry_count = 0
    content = None
    
    while retry_count < max_retries:
        try:
            content = call_qwen_vl(reference_base64, target_base64, user_prompt)
            if content:
                break
            retry_count += 1
        except Exception as e:
            print(f"Error evaluating {filename}: {e}. Retrying...")
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
                print(f"Invalid score for {filename}: {score}. Should be 0-4.")
                score = min(max(score, 0), 4)  # Clamp to 0-4
            
            # Store result
            result = {
                "filename": filename,
                "score": score,
                "content": content
            }
            
            return result
        else:
            print(f"Failed to extract score from Qwen VL response for {filename}.")
    else:
        print(f"Failed to evaluate {filename} after {max_retries} attempts")
    
    return None


def mode_erase(reference_image_path, target_dir, output_dir, prompt_file):
    """
    Erasure mode: Compare one fixed reference image to all images in target_dir.
    """
    print("\n=== Running in ERASE mode (one-to-many comparison) ===")
    
    # Create output directory
    ensure_dir(output_dir)
    
    # Check if reference image exists
    if not os.path.exists(reference_image_path):
        print(f"ERROR: Reference image not found: {reference_image_path}")
        return None
    
    # Check if target directory exists
    if not os.path.exists(target_dir):
        print(f"ERROR: Target directory not found: {target_dir}")
        return None
    
    # Load user prompt
    user_prompt = read_prompt_template(prompt_file)
    if not user_prompt:
        # Fallback prompt if file not found
        user_prompt = "Please evaluate if the first image contains a character that appears in the second image. Rate your evaluation on a scale of 0-4, where: 0 = No trace of the character, 1 = Barely recognizable elements, 2 = Some recognizable elements but highly distorted, 3 = Clearly recognizable but with differences, 4 = Clear and faithful representation. Provide your reasoning and then give the final score in the format 'Score: X'."
    
    # Load reference image ONCE
    try:
        reference_image = load_image(reference_image_path)
        reference_image_base64 = encode_image_into_base64(reference_image)
        print(f"Loaded reference image: {reference_image_path}")
    except Exception as e:
        print(f"ERROR: Failed to load reference image: {e}")
        return None
    
    # Get list of files in target directory
    target_files = [f for f in os.listdir(target_dir) 
                   if os.path.isfile(os.path.join(target_dir, f)) and 
                   f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Found {len(target_files)} images in target directory")
    
    # Lists to track all results
    all_results = []
    
    # Process each file in target directory
    for target_filename in tqdm(target_files, desc="Comparing images"):
        target_file = os.path.join(target_dir, target_filename)
        try:
            # Load target image
            target_image = load_image(target_file)
            target_image_base64 = encode_image_into_base64(target_image)
            
            # Evaluate the pair
            result = evaluate_image_pair(
                reference_image_base64, 
                target_image_base64, 
                target_filename, 
                user_prompt, 
                output_dir
            )
            
            if result:
                all_results.append(result)
        
        except Exception as e:
            print(f"Error processing file {target_filename}: {e}")
    
    # Save all results and return summary
    return save_results(all_results, output_dir)


def mode_retain(source_dir, target_dir, output_dir, prompt_file):
    """
    Retain mode: Compare images with the same filename between two folders (many-to-many).
    """
    print("\n Running in RETAIN mode (many-to-many comparison)")
    
    # Create output directory
    ensure_dir(output_dir)
    
    # Check if both directories exist
    if not os.path.exists(source_dir):
        print(f"ERROR: Source directory not found: {source_dir}")
        return None
    
    if not os.path.exists(target_dir):
        print(f"ERROR: Target directory not found: {target_dir}")
        return None
    
    # Load user prompt
    user_prompt = read_prompt_template(prompt_file)
    if not user_prompt:
        # Fallback prompt if file not found
        user_prompt = "Please evaluate if the first image contains a character that appears in the second image. Rate your evaluation on a scale of 0-4, where: 0 = No trace of the character, 1 = Barely recognizable elements, 2 = Some recognizable elements but highly distorted, 3 = Clearly recognizable but with differences, 4 = Clear and faithful representation. Provide your reasoning and then give the final score in the format 'Score: X'."
    
    # Get list of files in source directory
    source_files = [f for f in os.listdir(source_dir) 
                   if os.path.isfile(os.path.join(source_dir, f)) and 
                   f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Found {len(source_files)} images in source directory")
    
    # Lists to track all results
    all_results = []
    
    # Process each file in source directory and look for the same filename in target directory
    for source_filename in tqdm(source_files, desc="Comparing images"):
        # Check if the same file exists in target directory
        target_file = os.path.join(target_dir, source_filename)
        source_file = os.path.join(source_dir, source_filename)
        
        if not os.path.exists(target_file):
            print(f"Warning: Corresponding file not found in target directory: {source_filename}")
            continue
        
        try:
            # Load both images
            source_image = load_image(source_file)
            target_image = load_image(target_file)
            
            # Convert to base64
            source_image_base64 = encode_image_into_base64(source_image)
            target_image_base64 = encode_image_into_base64(target_image)
            
            # Evaluate the pair (source as reference, target as generated)
            result = evaluate_image_pair(
                source_image_base64, 
                target_image_base64, 
                source_filename, 
                user_prompt, 
                output_dir
            )
            
            if result:
                all_results.append(result)
        
        except Exception as e:
            print(f"Error processing file {source_filename}: {e}")
    
    # Save all results and return summary
    return save_results(all_results, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Image comparison evaluation using Qwen VL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Erase mode (one-to-many): Compare one reference image to all generated images
  python eval_qwenvl.py --mode erase \\
      --reference_image mario.jpg \\
      --target_dir generated_images/ \\
      --output_dir results/erase/

  # Retain mode (many-to-many): Compare matching images between two folders
  python eval_qwenvl.py --mode retain \\
      --source_dir original_images/ \\
      --target_dir generated_images/ \\
      --output_dir results/retain/
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--mode", 
        type=str, 
        required=True,
        choices=["erase", "retain"],
        help="Evaluation mode: 'erase' (one-to-many) or 'retain' (many-to-many)"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="Directory to save evaluation results"
    )
    
    parser.add_argument(
        "--target_dir", 
        type=str, 
        required=True,
        help="Directory containing target/generated images"
    )
    
    # Mode-specific arguments
    parser.add_argument(
        "--reference_image", 
        type=str, 
        default=None,
        help="Path to reference image (required for 'erase' mode)"
    )
    
    parser.add_argument(
        "--source_dir", 
        type=str, 
        default=None,
        help="Directory containing source/original images (required for 'retain' mode)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--prompt_file", 
        type=str, 
        default=None,
        help="Path to prompt template file (default: auto-select based on mode)"
    )
    
    args = parser.parse_args()
    
    # Set default prompt file based on mode if not provided
    if not args.prompt_file:
        if args.mode == "erase":
            args.prompt_file = 'prompts/user_prompt_characters-enhence.txt'
        else:  # retain mode
            args.prompt_file = 'prompts/user_prompt_celeb.txt'
    
    # Validate mode-specific arguments
    if args.mode == "erase":
        if not args.reference_image:
            parser.error("--reference_image is required for 'erase' mode")
        
        # Run erase mode
        results = mode_erase(
            args.reference_image, 
            args.target_dir, 
            args.output_dir, 
            args.prompt_file
        )
    
    elif args.mode == "retain":
        if not args.source_dir:
            parser.error("--source_dir is required for 'retain' mode")
        
        # Run retain mode
        results = mode_retain(
            args.source_dir, 
            args.target_dir, 
            args.output_dir, 
            args.prompt_file
        )
    
    # Print final results
    if results:
        print("\nEvaluation completed successfully!")
        print("\nResults saved to:")
        print(f"1. Summary CSV: {os.path.join(args.output_dir, 'summary.csv')}")
        print(f"2. All Results CSV: {os.path.join(args.output_dir, 'evaluation_results.csv')}")
    else:
        print("\n No results were generated")


if __name__ == "__main__":
    main()