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

    if args.time:
        start_time = time.time()
        
    # Timing model loading
    if args.time:
        load_start = time.time()
        
    model = TransformerModelWrapper(device, args.work_dir, enable_timing=args.time)
    model.load()
    
    if args.time:
        load_end = time.time()
        print(f"Loading model took {load_end - load_start:.2f}s")
    
    test_data_file = args.test_data
    output_file = args.test_output

    # Timing data loading
    if args.time:
        data_load_start = time.time()
        
    test_input = load_test_input(test_data_file)
    
    if args.time:
        data_load_end = time.time()
        print(f"Loading test data took {data_load_end - data_load_start:.2f}s")

    #TODO: Tune batch size on their machine for best perf
    # batch_size = 50000

    # Timing prediction
    if args.time:
        predict_start = time.time()
        
    preds = model.predict(test_input)
    
    if args.time:
        predict_end = time.time()
        print(f"Model prediction took {predict_end - predict_start:.2f}s")

    # preds = []
    # for i in range(0, len(test_input), batch_size):
    #     batch = test_input[i:i + batch_size]
    #     preds.extend(model.predict(batch))

    # Timing writing predictions
    if args.time:
        write_start = time.time()
        
    write_pred(preds, output_file)
    
    if args.time:
        write_end = time.time()
        print(f"Writing predictions took {write_end - write_start:.2f}s")

    if args.time:
        end_time = time.time()
        print(f"Running predict took {end_time-start_time}s")