import torch
from torch.utils.data import Dataset
import pandas as pd

from data_parsing.helpers import DatasetFileLoader


def build_vocab(text=None):
    """Create a vocabulary of printable lowercase English and Spanish characters."""
    import string
    
    # Start with ASCII printable characters but filter out uppercase and control chars
    printable_chars = [char for char in string.printable 
                      if not char.isupper() 
                      and char not in ['\t', '\n', '\r', '\v', '\f']]
    
    # Add only lowercase Spanish-specific characters
    spanish_chars = ['á', 'é', 'í', 'ó', 'ú', 'ü', 'ñ', '¿', '¡']
    
    # Combine and sort all characters
    all_chars = sorted(set(printable_chars + spanish_chars))
    
    # Create the mappings
    char_to_index = {char: idx for idx, char in enumerate(all_chars)}
    index_to_char = {idx: char for char, idx in char_to_index.items()}
    
    return char_to_index, index_to_char

class CharDataset(Dataset):
    def __init__(self, data: str,context_len=32):
        self.context_len = context_len
        self.data_indices = data 

    def __len__(self):
        return len(self.data_indices) - self.context_len

    def __getitem__(self, idx):
        input_seq = self.data_indices[idx:idx+self.context_len]
        target_char = self.data_indices[idx+self.context_len]
        return input_seq, target_char
    
class CharDatasetWrapper():
    # Load training data
    def __init__(self, device, data_directory, context_length: int = 32, dataset_fraction: float = 1):
        data_loader = DatasetFileLoader()
        data_loader.load(data_directory, dataset_fraction)

        # TODO: Fix this so getting [0] and slicing first line isn't needed
        train_data = data_loader.train_data[0][1:]

        dev_data = data_loader.dev_data[0][1:]

        train_text = "\n".join(train_data)
        dev_text = "\n".join(dev_data)

        # Build vocabulary 
        self._vocab, _ = build_vocab(train_text)

        train_data = torch.tensor([self._vocab[char] for char in train_text if char in self._vocab])
        dev_data = torch.tensor([self._vocab[char] for char in dev_text if char in self._vocab])

        self._train_dataset = CharDataset(train_data,  context_length)
        self._dev_dataset = CharDataset(dev_data, context_length)

    def vocab(self):
        return self._vocab
    
    def train_dataset(self):
        return self._train_dataset
    
    def dev_dataset(self):
        return self._dev_dataset
    
    def vocab_size(self):
        return len(self._vocab)
