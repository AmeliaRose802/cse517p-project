#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from Transformer_Based.transformer_wrapper import TransformerModelWrapper
import torch
from helpers import load_test_input, write_pred
import time

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='test/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    parser.add_argument('--data_fraction', help='Fraction the training data to train on', default=1)
    parser.add_argument('--time', help='Measure training time', action='store_true')
    parser.add_argument('--continue_training', help='Continue to train an exisiting model', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pre-initialize CUDA to avoid the penalty during model creation
    if torch.cuda.is_available():
        # Force CUDA initialization with a small operation
        dummy = torch.ones(1).cuda()
        # Perform a simple operation and synchronize
        dummy = dummy * 2
        torch.cuda.synchronize()
        del dummy
        torch.cuda.empty_cache()

    if args.time:
        start_time = time.time()
        

    model = TransformerModelWrapper(device, args.work_dir)
    
    model.load()
    
    test_data_file = args.test_data
    output_file = args.test_output

    test_input = load_test_input(test_data_file)

    #TODO: Tune batch size on their machine for best perf
    # batch_size = 50000

    preds = model.predict(test_input)

    # preds = []
    # for i in range(0, len(test_input), batch_size):
    #     batch = test_input[i:i + batch_size]
    #     preds.extend(model.predict(batch))

    write_pred(preds, output_file)

    if args.time:
        end_time = time.time()
        print(f"Running predict took {end_time-start_time}s")
    
    """
    Profiler analysis shows the following bottlenecks:
    
    1. CUDA initialization overhead: 
       - cudaDeviceGetStreamPriorityRange takes 47.13% of Self CPU time (630.810ms)
       - This indicates significant time is spent just setting up the CUDA environment
    
    2. Tensor operations and memory management:
       - aten::to, aten::_to_copy, and aten::empty_strided operations consume ~50% of CPU total time
       - These operations are related to tensor creation and device transfers (CPU to GPU)
    
    3. Memory allocations:
       - aten::empty_strided uses 15.10 MB of CPU memory and 19.29 MB of CUDA memory
       - This suggests significant memory allocation overhead during model initialization
    
    4. Most work is CPU-bound:
       - Self CPU time (1.338s) vs Self CUDA time (1.891ms)
       - This indicates model instantiation is primarily CPU-bound rather than GPU-bound
    
    Optimization opportunities:
    - Consider pre-initializing CUDA before model creation
    - Batch tensor allocations where possible
    - Consider using CPU tensors for initialization, then transferring to GPU as a batch
    
    Benefits of pre-initializing CUDA before model creation:
    
    1. First-time initialization penalty: The first CUDA operation in a process is 
       particularly expensive (as shown by cudaDeviceGetStreamPriorityRange taking 630ms).
       This includes driver initialization, context creation, and memory system setup.
    
    2. By performing simple CUDA operations before model creation, these one-time costs 
       are paid upfront, resulting in more consistent and faster model initialization.
    
    3. Implementation approach:
       - Run a small dummy tensor operation on the GPU before model creation
       - This forces CUDA initialization to happen earlier
       - The subsequent model creation will have reduced overhead
       
    Example implementation:
    ```
    # Pre-initialize CUDA to avoid the penalty during model creation
    if torch.cuda.is_available():
        # Force CUDA initialization with a small operation
        dummy = torch.ones(1).cuda()
        # Perform a simple operation and synchronize
        dummy = dummy * 2
        torch.cuda.synchronize()
        del dummy
        torch.cuda.empty_cache()
    ```
    
    This would move the ~630ms CUDA initialization overhead outside the model creation,
    potentially reducing model instantiation time by up to 47%.
    """