import os
import pandas as pd

def load_test_input(input_file):
    with open(input_file, encoding='utf-8') as f:
        loaded = []
        for line in f:
            line = line[:-1].lower()
            loaded.append(line)
        return loaded

def write_pred(preds, fname):
    with open(fname, 'wt') as f:
        for p in preds:
            f.write('{}\n'.format(p))

def get_top1_accuracy(gold, pred):
    """
    Returns the fraction of cases where the first character in pred matches the gold label.
    pred: list of strings, each string contains 3 possible characters (predictions)
    gold: list of strings, each string is the gold label
    """
    correct = 0
    for i, (g, p) in enumerate(zip(gold, pred)):
        if len(p) > 2 and p[0] == g:
            correct += 1
    return correct / len(gold)

def get_top3_accuracy(gold, pred):
    """
    Returns the fraction of cases where any character in pred matches the gold label.
    pred: list of strings, each string contains 3 possible characters (predictions)
    gold: list of strings, each string is the gold label
    """
    correct = 0
    for i, (g, p) in enumerate(zip(gold, pred)):
        if g in p:
            correct += 1
    return correct / len(gold)

def load_true(test_dir):
    with open(f"{test_dir}/answer.txt", encoding='utf-8') as f:
        loaded = []
        for line in f:
            line = line[:-1].lower()
            loaded.append(line)
        return loaded

def load_predicted(test_dir):
    with open(f"{test_dir}/pred.txt", encoding='utf-8') as f:
        loaded = []
        for line in f:
            line = line[:-1].lower()
            loaded.append(line)
        return loaded

class DatasetFileLoader():
    def __init__(self):
        self.test_data = None
        self.dev_data = None
        self.train_data = None
    
    def load(self, data_directory, train_fraction: float=1, dev_fraction: float=1, test_fraction: float = 1):

        files = [f for f in os.listdir(data_directory) if os.path.isfile(os.path.join(data_directory, f))]

        self.test_data = pd.DataFrame()
        self.dev_data = pd.DataFrame()
        self.train_data = pd.DataFrame()

        for file_name in files:
            data = pd.read_csv(f"{data_directory}/{file_name}")["dialogue"]

            if "dev" in file_name:
               self.dev_data = pd.concat([self.dev_data, data], ignore_index=True)
            if f"test" in file_name:
                self.test_data = pd.concat([self.test_data, data], ignore_index=True)
            if "train" in file_name:
                self.train_data = pd.concat([self.train_data, data], ignore_index=True)
        
        self.test_data  = self.test_data.sample(frac=test_fraction).reset_index(drop=True)
        self.dev_data = self.dev_data.sample(frac=dev_fraction).reset_index(drop=True)
        self.train_data = self.train_data.sample(frac=train_fraction).reset_index(drop=True)
