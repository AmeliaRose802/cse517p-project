#!/usr/bin/env python
import os
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from Transformer_Based.transformer_wrapper import TransformerModelWrapper
import torch
import numpy as np

def streaming_load_and_predict(
    model, input_file, output_file, batch_size=1000, 
    buffer_size=10000, show_progress=True
):
    """
    Process a large input file in streaming mode without loading everything into memory.
    
    Args:
        model: The TransformerModelWrapper instance
        input_file: Path to input file
        output_file: Path to output predictions file
        batch_size: Batch size for prediction
        buffer_size: Number of examples to buffer in memory
        show_progress: Whether to show progress information
    """
    total_lines = 0
    processed_lines = 0
    start_time = time.time()
    
    # Count total lines for progress reporting
    if show_progress:
        print("Counting lines in input file...")
        with open(input_file, 'r', encoding='utf-8') as f:
            for _ in f:
                total_lines += 1
        print(f"Total lines: {total_lines}")
    
    # Process in streaming mode
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        buffer = []
        batch_times = []
        
        for line in f_in:
            buffer.append(line.strip())
            
            # Process when buffer reaches target size
            if len(buffer) >= buffer_size:
                # Process buffer in batches
                for i in range(0, len(buffer), batch_size):
                    batch_start = time.time()
                    batch = buffer[i:i + batch_size]
                    
                    # Make predictions
                    preds = model.predict(batch)
                    
                    # Write predictions
                    for pred in preds:
                        f_out.write(pred + '\n')
                    
                    processed_lines += len(batch)
                    batch_end = time.time()
                    batch_time = batch_end - batch_start
                    batch_times.append(batch_time)
                    
                    if show_progress:
                        elapsed = time.time() - start_time
                        percent = processed_lines / total_lines * 100 if total_lines > 0 else 0
                        examples_per_sec = processed_lines / elapsed if elapsed > 0 else 0
                        eta = (total_lines - processed_lines) / examples_per_sec if examples_per_sec > 0 else 0
                        
                        print(f"Processed {processed_lines}/{total_lines} ({percent:.1f}%) | "
                              f"Speed: {examples_per_sec:.1f} examples/s | "
                              f"ETA: {eta:.1f}s")
                
                # Clear buffer after processing
                buffer = []
        
        # Process any remaining items in buffer
        if buffer:
            for i in range(0, len(buffer), batch_size):
                batch = buffer[i:i + batch_size]
                preds = model.predict(batch)
                for pred in preds:
                    f_out.write(pred + '\n')
                processed_lines += len(batch)
    
    # Show summary
    if show_progress and batch_times:
        total_time = time.time() - start_time
        print(f"\nProcessing complete!")
        print(f"Total processing time: {total_time:.2f}s")
        print(f"Average batch time: {np.mean(batch_times):.4f}s")
        print(f"Overall speed: {processed_lines/total_time:.1f} examples/s")

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='test/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    parser.add_argument('--batch_size', help='Batch size for prediction', type=int, default=1000)
    parser.add_argument('--buffer_size', help='Buffer size for streaming', type=int, default=10000)
    parser.add_argument('--time', help='Measure processing time', action='store_true')
    parser.add_argument('--torchscript', help='Use TorchScript for faster loading', action='store_true')
    parser.add_argument('--quantize', help='Use quantized model for faster loading and inference', action='store_true')
    parser.add_argument('--mmap', help='Use memory mapping for faster loading', action='store_true')
    parser.add_argument('--reuse_model', help='Reuse model from previous runs', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.time:
        start_time = time.time()
    
    # Load model
    print("Loading model...")
    model = TransformerModelWrapper(
        device=device, 
        work_directory=args.work_dir,
        use_torchscript=args.torchscript,
        use_quantization=args.quantize,
        use_mmap=args.mmap,
        reuse_model=args.reuse_model
    )
    model.load()
    
    if args.time:
        model_load_time = time.time() - start_time
        print(f"Model loaded in {model_load_time:.2f}s")
    
    # Process in streaming mode
    streaming_load_and_predict(
        model=model,
        input_file=args.test_data,
        output_file=args.test_output,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        show_progress=True
    )
    
    if args.time:
        total_time = time.time() - start_time
        print(f"Total execution time: {total_time:.2f}s")
