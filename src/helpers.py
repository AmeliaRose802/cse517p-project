import os
import time
import mmap
import numpy as np
import pandas as pd
import pickle

# Cache for loaded test data
_test_data_cache = {}

def load_test_input(test_data_file, use_cache=True, use_mmap=True, num_workers=1):
    """Optimized test data loading for Docker environments"""
    start_time = time.time()
    
    # Check for preprocessed binary version first (fastest)
    binary_cache_path = f"{test_data_file}.bin"
    if use_cache and os.path.exists(binary_cache_path):
        print(f"Loading pre-processed binary data from {binary_cache_path}")
        with open(binary_cache_path, 'rb') as f:
            data = pickle.load(f)
            load_time = time.time() - start_time
            print(f"Loaded {len(data)} examples from binary cache in {load_time:.2f}s")
            return data
    
    # Use optimized file reading based on file size
    file_size = os.path.getsize(test_data_file)
    print(f"Loading test data from {test_data_file} ({file_size/1024/1024:.2f} MB)")
    
    if use_mmap and file_size > 1024 * 1024:  # >1MB use mmap
        try:
            # Memory-mapped reading - much faster for large files
            with open(test_data_file, 'r', encoding='utf-8') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    # Bulk read is faster than line-by-line
                    content = mm.read().decode('utf-8')
                    lines = content.splitlines()
        except Exception as e:
            print(f"Memory mapping failed ({e}), falling back to buffered reading")
            # Fallback to large buffer reading
            with open(test_data_file, 'r', encoding='utf-8', buffering=16*1024*1024) as f:
                lines = f.read().splitlines()
    else:
        # For smaller files, use buffered reading
        with open(test_data_file, 'r', encoding='utf-8', buffering=8*1024*1024) as f:
            lines = f.read().splitlines()
    
    # Optionally save binary version for future use
    if use_cache:
        try:
            with open(binary_cache_path, 'wb') as f:
                pickle.dump(lines, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved binary cache to {binary_cache_path} for faster future loading")
        except Exception as e:
            print(f"Failed to save binary cache: {e}")
    
    load_time = time.time() - start_time
    print(f"Loaded {len(lines)} examples in {load_time:.2f}s ({len(lines)/load_time:.1f} examples/s)")
    return lines

def write_pred(preds, output_file):
    """Optimized prediction writing"""
    start_time = time.time()
    with open(output_file, 'w', encoding='utf-8', buffering=16*1024*1024) as f:
        # Writing as a single joined string is faster than line-by-line
        f.write('\n'.join(preds))
    
    write_time = time.time() - start_time
    print(f"Wrote {len(preds)} predictions to {output_file} in {write_time:.2f}s")

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
            print(f"Reading file: {file_name}") 
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
