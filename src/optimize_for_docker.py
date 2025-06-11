#!/usr/bin/env python
"""
Pre-optimization script for Docker evaluation.
Run this before evaluation to create optimized models and improve performance.
"""
import os
import time
import torch
from argparse import ArgumentParser
from Transformer_Based.transformer_wrapper import TransformerModelWrapper

def export_optimized_model(work_dir):
    """Export optimized model formats for faster Docker evaluation"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Step 1: Load the model
    print("Loading model...")
    start_time = time.time()
    model_wrapper = TransformerModelWrapper(device, work_dir)
    model_wrapper.load()
    print(f"Model loaded in {time.time() - start_time:.2f}s")
    
    # Step 2: Export TorchScript model
    print("Exporting to TorchScript format...")
    torchscript_path = os.path.join(work_dir, "character_transformer_script.pt")
    
    start_time = time.time()
    model_wrapper.model.eval()
    traced_script_module = torch.jit.script(model_wrapper.model)
    traced_script_module.save(torchscript_path)
    print(f"TorchScript export completed in {time.time() - start_time:.2f}s")
    print(f"TorchScript model saved to {torchscript_path}")
    
    # Step 3: Preprocess test data (if provided)
    parser = ArgumentParser()
    parser.add_argument('--work_dir', help='Model directory', default='work')
    parser.add_argument('--test_data', help='Preprocess this test file', default=None)
    args = parser.parse_args()
    
    if args.test_data and os.path.exists(args.test_data):
        print(f"\nPreprocessing test data file: {args.test_data}")
        from helpers import load_test_input
        
        # This will create a binary cache file
        start_time = time.time()
        data = load_test_input(args.test_data, use_cache=True, use_mmap=True)
        print(f"Test data preprocessing completed in {time.time() - start_time:.2f}s")
        print(f"Future runs will load much faster using the binary cache")
    
    print("\nOptimization completed successfully!")
    print("For best performance in Docker, run:")
    print(f"python predict.py --work_dir {work_dir} --test_data test/input.txt --torchscript --use_data_cache")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--work_dir', help='Model directory', default='work')
    args = parser.parse_args()
    
    export_optimized_model(args.work_dir)
