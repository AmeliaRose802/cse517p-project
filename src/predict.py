#!/usr/bin/env python
import os
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from Transformer_Based.transformer_wrapper import TransformerModelWrapper
import torch
from helpers import load_test_input, write_pred

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='test/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    parser.add_argument('--time', help='Measure execution time', action='store_true')
    parser.add_argument('--torchscript', help='Use TorchScript for faster loading', action='store_true')
    parser.add_argument('--use_data_cache', help='Use binary cached data if available', action='store_true')
    parser.add_argument('--mmap', help='Use memory mapping for faster loading', action='store_true')
    args = parser.parse_args()

    # In Docker with sufficient GPU memory, we focus on optimizing loading times
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    total_start_time = time.time() if args.time else 0
    
    # Step 1: Load model - optimized for speed
    model_load_start = time.time() if args.time else 0
    model = TransformerModelWrapper(
        device=device, 
        work_directory=args.work_dir,
        use_torchscript=args.torchscript,
        use_mmap=args.mmap
    )
    model.load()
    
    if args.time:
        model_load_time = time.time() - model_load_start
        print(f"Model loaded in {model_load_time:.2f}s")
    
    # Step 2: Load test data - optimized for speed
    data_load_start = time.time() if args.time else 0
    test_input = load_test_input(
        args.test_data,
        use_cache=args.use_data_cache,
        use_mmap=args.mmap
    )
    
    if args.time:
        data_load_time = time.time() - data_load_start
        print(f"Test data loaded in {data_load_time:.2f}s")
    
    # Step 3: Run prediction - single batch for Docker with sufficient memory
    predict_start = time.time() if args.time else 0
    preds = model.predict(test_input)
    
    if args.time:
        predict_time = time.time() - predict_start
        examples_per_second = len(test_input) / predict_time
        print(f"Predictions completed in {predict_time:.2f}s ({examples_per_second:.1f} examples/s)")
    
    # Step 4: Write predictions
    write_start = time.time() if args.time else 0
    write_pred(preds, args.test_output)
    
    if args.time:
        write_time = time.time() - write_start
        print(f"Results written in {write_time:.2f}s")
        
        # Report total time
        total_time = time.time() - total_start_time
        print(f"\nTotal execution time: {total_time:.2f}s")
        print(f"  - Model loading: {model_load_time:.2f}s ({model_load_time/total_time*100:.1f}%)")
        print(f"  - Data loading:  {data_load_time:.2f}s ({data_load_time/total_time*100:.1f}%)")
        print(f"  - Prediction:    {predict_time:.2f}s ({predict_time/total_time*100:.1f}%)")
        print(f"  - Writing:       {write_time:.2f}s ({write_time/total_time*100:.1f}%)")