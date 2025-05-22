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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.time:
        start_time = time.time()

    model = TransformerModelWrapper(device, args.work_dir)

    
    model.load()
    
    test_data_file = args.test_data
    output_file = args.test_output
    print(test_data_file)

    test_input = load_test_input(test_data_file)

    preds = model.predict(test_input)

    write_pred(preds, output_file)

    if args.time:
        end_time = time.time()
        print(f"Running predict took {end_time-start_time}s")