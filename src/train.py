#!/usr/bin/env python
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from Transformer_Based.character_dataset import CharDatasetWrapper
from Transformer_Based.transformer_wrapper import TransformerModelWrapper
import torch
import random
import time
from typing import List, Union

if __name__ == '__main__':
    
    random.seed(42)
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--data_dir', help='where to load train data', default='work')
    parser.add_argument('--data_fraction', help='Fraction of the training data to train on', default=1)
    parser.add_argument('--charset', help='Character set(s) to use for training (comma-separated, e.g., "latin,devanagari")', default="latin")
    parser.add_argument('--time', help='Measure training time', action='store_true')
    parser.add_argument('--continue_training', help='Continue to train an exisiting model', action='store_true')
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Process charset parameter - convert comma-separated string to list if needed
    charset_input: str = args.charset.strip()
    charsets: Union[str, List[str]] = charset_input
    if "," in charset_input:
        charsets = [cs.strip() for cs in charset_input.split(",")]
    
    model = TransformerModelWrapper(device, args.work_dir, use_existing_vocab=False, character_set="all")

    dataset = CharDatasetWrapper(model.vocab, args.data_dir, model.context_length, float(args.data_fraction), charsets)

    if args.continue_training:
        model.load()

    if args.time:
        start_time = time.time()

    model.train(dataset)

    if args.time:
        elapsed_time = time.time() - start_time
        print(f"Total training time: {elapsed_time:.2f} seconds")