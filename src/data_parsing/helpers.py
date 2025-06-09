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

def get_accuracy(pred, gold):
    correct = 0
    for i, (p, g) in enumerate(zip(pred, gold)):
        right = g in p
        correct += right
    return correct/len(gold)

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

# Map from character set names to directory paths
character_sets = {
    "latin": ["latin"],  # Updated - Latin no longer has subdirectories
    "english": ["english"],  # Special marker for English files
    "cyrillic": ["cyrillic"],
    "cjk": ["cjk"],
    "devanagari": ["devanagari"],
    "arabic": ["arabic"],
    "hebrew": ["hebrew"],
    "greek": ["greek"],
    "bengali": ["bengali"],
    "thai": ["thai"],
    "all": None  # None means include all
}

# Reverse mapping from language family paths to character set names
family_to_charset = {
    "latin": "latin",  # Updated for flat structure
    "cyrillic": "cyrillic",
    "cjk": "cjk",
    "devanagari": "devanagari",
    "arabic": "arabic",
    "hebrew": "hebrew",
    "greek": "greek",
    "bengali": "bengali",
    "thai": "thai"
}

class DatasetFileLoader():
    def __init__(self):
        self.test_data = None
        self.dev_data = None
        self.train_data = None
    
    def load(self, data_directory, fraction : float=1, character_set=None):
        """
        Load data from the specified directory, filtering by character set if specified.
        
        Args:
            data_directory: The directory containing the data files
            fraction: Fraction of data to sample (default: 1, meaning all data)
            character_set: The character set to filter by (default: None, meaning all character sets)
                          Can be a single character set name, a list of character set names, or "all"
        """
        self.test_data = pd.DataFrame()
        self.dev_data = pd.DataFrame()
        self.train_data = pd.DataFrame()
        
        # If character_set is "all" or None, load all data
        if character_set == "all" or character_set is None:
            return self._load_all_data(data_directory, fraction)
        
        # Convert single character set to list
        if isinstance(character_set, str):
            character_set = [character_set]
        
        # Get all family directories to include
        family_dirs = []
        load_english = False
        
        for cs in character_set:
            if cs == "english":
                load_english = True
                continue  # Skip adding to family_dirs since English is handled specially
            if cs in character_sets and character_sets[cs] is not None:
                family_dirs.extend(character_sets[cs])
        
        # Load data from each family directory
        for family_dir in family_dirs:
            family_path = os.path.join(data_directory, family_dir)
            if os.path.isdir(family_path):
                self._load_from_directory(family_path, fraction)
        
        # Handle English files specially if requested
        if load_english:
            self._load_english_files(data_directory, fraction)
        
        # Sample and reset index
        self.test_data = self.test_data.sample(frac=fraction).reset_index(drop=True)
        self.dev_data = self.dev_data.sample(frac=fraction).reset_index(drop=True)
        self.train_data = self.train_data.sample(frac=fraction).reset_index(drop=True)
    
    def _load_all_data(self, data_directory, fraction):
        """Load all data from the root directory and all subdirectories"""
        # First load files from the root directory
        if os.path.exists(data_directory):
            self._load_from_directory(data_directory, fraction)
        
        # Then load from each subdirectory
        for item in os.listdir(data_directory):
            item_path = os.path.join(data_directory, item)
            if os.path.isdir(item_path):
                # If it's a nested directory structure (like latin)
                for subitem in os.listdir(item_path):
                    subitem_path = os.path.join(item_path, subitem)
                    if os.path.isdir(subitem_path):
                        self._load_from_directory(subitem_path, fraction)
                # Also load from the directory itself
                self._load_from_directory(item_path, fraction)
        
        # Sample and reset index
        self.test_data = self.test_data.sample(frac=fraction).reset_index(drop=True)
        self.dev_data = self.dev_data.sample(frac=fraction).reset_index(drop=True)
        self.train_data = self.train_data.sample(frac=fraction).reset_index(drop=True)
    
    def _load_from_directory(self, directory, fraction):
        """Load all CSV files from a directory"""
        if not os.path.exists(directory):
            return
        
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith('.csv')]
        
        for file_name in files:
            print(f"Reading file: {os.path.join(directory, file_name)}")
            try:
                data = pd.read_csv(os.path.join(directory, file_name))["dialogue"]
                
                if "dev" in file_name:
                    self.dev_data = pd.concat([self.dev_data, data], ignore_index=True)
                if "test" in file_name:
                    self.test_data = pd.concat([self.test_data, data], ignore_index=True)
                if "train" in file_name:
                    self.train_data = pd.concat([self.train_data, data], ignore_index=True)
            except Exception as e:
                print(f"Error reading {file_name}: {str(e)}")
                
    def _load_english_files(self, data_directory, fraction):
        """Load specifically NASA and Star Trek files which are in English"""
        latin_dir = os.path.join(data_directory, "latin")  # Updated path
        if not os.path.exists(latin_dir):
            print(f"Warning: English files directory {latin_dir} not found")
            return
            
        english_files = [
            "dev_nasa.csv", "dev_trek.csv",
            "test_nasa.csv", "test_trek.csv",
            "train_nasa.csv", "train_trek.csv"
        ]
        
        for file_name in english_files:
            file_path = os.path.join(latin_dir, file_name)  # Updated path
            if not os.path.exists(file_path):
                print(f"Warning: English file {file_path} not found")
                continue
                
            print(f"Reading English file: {file_path}")
            try:
                data = pd.read_csv(file_path)["dialogue"]
                
                if "dev" in file_name:
                    self.dev_data = pd.concat([self.dev_data, data], ignore_index=True)
                if "test" in file_name:
                    self.test_data = pd.concat([self.test_data, data], ignore_index=True)
                if "train" in file_name:
                    self.train_data = pd.concat([self.train_data, data], ignore_index=True)
            except Exception as e:
                print(f"Error reading English file {file_name}: {str(e)}")
