#!/usr/bin/env python3

import argparse
import os
import sys
import time
import torch
from tabulate import tabulate

# Add project root to Python path to ensure 'src' package is accessible
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import helper functions
from helpers import load_true, load_predicted, get_top1_accuracy, get_top3_accuracy, load_test_input, write_pred
# Import model wrapper directly
from Transformer_Based.transformer_wrapper import TransformerModelWrapper
# Import language constants
from language_consts import CHARACTER_SET_GROUPS, LANGUAGES

def get_language_name(lang_code):
    """Return the full language name for a given language code."""
    if lang_code == "Combined":
        return "Combined"
    return LANGUAGES.get(lang_code, {}).get("name", lang_code)

def get_language_charset(lang_code):
    """Return the character set group for a given language code."""
    if lang_code == "Combined":
        return "mixed"
        
    # Get script directly from LANGUAGES
    return LANGUAGES.get(lang_code, {}).get("script", "unknown")

def calculate_averages(results):
    """Calculate average accuracies by character set."""
    charset_averages = {}
    
    for charset, charset_data in results.items():
        if charset == "Combined":
            continue
            
        top1_scores = []
        top3_scores = []
        
        for lang_result in charset_data["languages"]:
            # Skip error results
            if lang_result[1] != "Error" and lang_result[2] != "Error":
                try:
                    top1_val = float(lang_result[1].replace('%', ''))
                    top3_val = float(lang_result[2].replace('%', ''))
                    top1_scores.append(top1_val)
                    top3_scores.append(top3_val)
                except ValueError:
                    continue
        
        if top1_scores and top3_scores:
            charset_averages[charset] = {
                "top1": sum(top1_scores) / len(top1_scores),
                "top3": sum(top3_scores) / len(top3_scores),
                "count": len(top1_scores)
            }
    
    return charset_averages

def main():
    parser = argparse.ArgumentParser(description="Evaluate model accuracy by language")
    parser.add_argument("--work_dir", type=str, default="./work", 
                        help="Directory containing the model and vocabulary")
    parser.add_argument("--test_base_dir", type=str, default="./test", 
                        help="Base directory containing test data")
    args = parser.parse_args()
    
    # Dynamically build test directories from subdirectories
    test_dirs = {"Combined": args.test_base_dir}
    
    # Add all subdirectories as separate languages
    for item in os.listdir(args.test_base_dir):
        full_path = os.path.join(args.test_base_dir, item)
        if os.path.isdir(full_path):
            test_dirs[item] = full_path
    
    results = {}  # Group results by character set
    
    # Set up device and model once to reuse
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModelWrapper(device,args.work_dir)
    model.load()
    
    # Run evaluation for each language directory
    for language, test_dir in test_dirs.items():
        
        if not os.path.exists(test_dir):
            print(f"Warning: Test directory {test_dir} does not exist. Skipping {language} evaluation.")
            continue
            
        # Get full language name
        language_name = get_language_name(language)
        print(f"\n===== Evaluating {language_name} ({language}) =====")
        
        # Get character set info
        charset = get_language_charset(language)
        
        # Get charset description from CHARACTER_SET_GROUPS or from the script name
        if charset in CHARACTER_SET_GROUPS:
            charset_desc = CHARACTER_SET_GROUPS.get(charset, {}).get("description", "Unknown character set")
        else:
            charset_desc = f"{charset.capitalize()} script"
            
        if language == "Combined":
            charset, charset_desc = "mixed", "Mixed character sets"
        
        try:
            # Step 1: Generate predictions directly
            print(f"Generating predictions for {language}...")
            input_file = os.path.join(test_dir, "input.txt")
            pred_file = os.path.join(test_dir, "pred.txt")
            
            # Load test input
            test_input = load_test_input(input_file)

            start = time.perf_counter()
            
            # Generate predictions in batches to avoid memory issues
            batch_size = 5000
            preds = []
            for i in range(0, len(test_input), batch_size):
                batch = test_input[i:i + batch_size]
                preds.extend(model.predict(batch))

            end = time.perf_counter()
            elapsed_time = end - start
            
            # Calculate number of examples and time per prediction
            num_examples = len(test_input)
            ms_per_prediction = (elapsed_time / num_examples) * 1000 if num_examples > 0 else 0
            
            # Write predictions to file
            write_pred(preds, pred_file)

            # Step 2: Direct evaluation using helpers
            print(f"Evaluating accuracy for {language}...")
            
            # Load true labels and predictions
            y_true = load_true(test_dir)
            pred = load_predicted(test_dir)
            
            # Calculate top-1 and top-3 accuracy
            top1_acc = get_top1_accuracy(y_true, pred)
            top3_acc = get_top3_accuracy(y_true, pred)
            top1_pct = top1_acc * 100
            top3_pct = top3_acc * 100
            
            # Initialize charset group if not exists
            if charset not in results:
                results[charset] = {
                    "description": charset_desc,
                    "languages": []
                }
            # Store results grouped by charset with full language name
            results[charset]["languages"].append([
                f"{language_name} ({language})", 
                f"{top1_pct:.2f}%", 
                f"{top3_pct:.2f}%", 
                f"{elapsed_time:.2f} sec",
                f"{num_examples}",
                f"{ms_per_prediction:.2f} ms"
            ])
            
            # Print results
            print(f"Character Set: {charset} ({charset_desc})")
            print(f"Top-1 Accuracy: {top1_acc:.2%}")
            print(f"Top-3 Accuracy: {top3_acc:.2%}")
            print(f"Number of examples: {num_examples}")
            print(f"Time per prediction: {ms_per_prediction:.2f} ms")
            
        except Exception as e:
            print(f"Error during evaluation for {language_name} ({language}): {e}")
            if charset not in results:
                results[charset] = {
                    "description": charset_desc,
                    "languages": []
                }
            results[charset]["languages"].append([f"{language_name} ({language})", "Error", "Error", "Error", "Error", "Error"])

    # Print summary table grouped by character set
    if results:
        print("\n===== SUMMARY BY CHARACTER SET =====")
        
        # Calculate averages
        charset_averages = calculate_averages(results)
        
        # Sort character sets for consistent output
        for charset in sorted(results.keys()):
            charset_data = results[charset]
            print(f"\n{charset} ({charset_data['description']})")
            print("=" * (len(charset) + len(charset_data['description']) + 3))
            
            if charset_data["languages"]:
                # Sort languages by top-3 accuracy (low to high)
                sorted_languages = sorted(charset_data["languages"], 
                                         key=lambda x: float(x[2].replace('%', '')) if x[2] != "Error" else -1)
                
                print(tabulate(sorted_languages, headers=[
                    "Language", 
                    "Top-1 Accuracy", 
                    "Top-3 Accuracy", 
                    "Total Time", 
                    "Examples", 
                    "ms/prediction"
                ], tablefmt="grid"))
                
                # Show charset average if available
                if charset in charset_averages:
                    avg_data = charset_averages[charset]
                    print(f"\nCharacter Set Average (n={avg_data['count']}): Top-1: {avg_data['top1']:.2f}%, Top-3: {avg_data['top3']:.2f}%")
            else:
                print("No languages evaluated in this character set.")
        
        # Print character set averages summary
        if charset_averages:
            print("\n===== CHARACTER SET AVERAGES SUMMARY =====")
            charset_summary = []
            for charset, avg_data in sorted(charset_averages.items()):
                description = results[charset]["description"]
                charset_summary.append([
                    charset,
                    description,
                    f"{avg_data['top1']:.2f}%",
                    f"{avg_data['top3']:.2f}%",
                    avg_data['count']
                ])
            
            print(tabulate(charset_summary, headers=[
                "Character Set", 
                "Description",
                "Avg Top-1", 
                "Avg Top-3", 
                "Languages Tested"
            ], tablefmt="grid"))
    else:
        print("\nNo results were generated.")

if __name__ == "__main__":
    main()
