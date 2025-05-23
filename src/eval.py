from helpers import load_true, get_accuracy, load_predicted
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torch

if __name__ == '__main__':

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--test_dir', help='path to data to be evald', default='test/')
    args = parser.parse_args()

    y_true = load_true(args.test_dir)

    pred = load_predicted(args.test_dir)

    accuracy = get_accuracy(y_true, pred)

    print(f"Accuracy is: {accuracy:.2%}")