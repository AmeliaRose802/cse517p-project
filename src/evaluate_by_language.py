#!/usr/bin/env python3

import argparse
import os
import torch
from tabulate import tabulate
# Import helper functions
from helpers import load_true, load_predicted, get_top1_accuracy, get_top3_accuracy, load_test_input, write_pred
# Import model wrapper directly
from Transformer_Based.transformer_wrapper import TransformerModelWrapper
import time

def main():
    parser = argparse.ArgumentParser(description="Evaluate model accuracy by language")
    parser.add_argument("--work_dir", type=str, default="./model_experiments/exp_4_more_layers", 
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
    
    results = []
    
    # Set up device and model once to reuse
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModelWrapper(device, args.work_dir)
    model.load()
    
    # Run evaluation for each language directory
    for language, test_dir in test_dirs.items():

        
        
        if not os.path.exists(test_dir):
            print(f"Warning: Test directory {test_dir} does not exist. Skipping {language} evaluation.")
            continue
            
        print(f"\n===== Evaluating {language} =====")
        
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
            
            # Store results
            results.append([
                language, 
                f"{top1_pct:.2f}%", 
                f"{top3_pct:.2f}%", 
                f"{elapsed_time:.2f} sec",
                f"{num_examples}",
                f"{ms_per_prediction:.2f} ms"
            ])
            
            # Print results
            print(f"Top-1 Accuracy: {top1_acc:.2%}")
            print(f"Top-3 Accuracy: {top3_acc:.2%}")
            print(f"Number of examples: {num_examples}")
            print(f"Time per prediction: {ms_per_prediction:.2f} ms")
            
        except Exception as e:
            print(f"Error during evaluation for {language}: {e}")
            results.append([language, "Error", "Error", "Error", "Error", "Error"])

    # Print summary table
    if results:
        print("\n===== SUMMARY =====")
        print(tabulate(results, headers=[
            "Language", 
            "Top-1 Accuracy", 
            "Top-3 Accuracy", 
            "Total Time", 
            "Examples", 
            "ms/prediction"
        ], tablefmt="grid"))
    else:
        print("\nNo results were generated.")

if __name__ == "__main__":
    main()