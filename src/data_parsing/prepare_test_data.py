import os
import sys
import pandas as pd
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Add the parent directory (src) to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Now we can import helpers and other modules
from helpers import DatasetFileLoader
from src.language_consts import LANGUAGES

# Function to get the language family path for a given language code
def get_language_family_path(lang_code):
    """Return the script subdirectory for a given language code"""
    if lang_code in LANGUAGES:
        return LANGUAGES[lang_code].get("script", "other")
    return "other"

def to_sample_and_expected_result(data, min_input_size = 3, max_input_size=300):
    """
    This function is used to build a test set in the example format so the model can be evaluated
    Splits the provided input into sample, y_true pairs.
    Each line is split at a random index between MIN_INPUT_SIZE and MAX_INPUT_SIZE
    y_true represents the next chracter after the split. The remainder of the string is discarded

    Output:

    x: list of training samples
    y_true: list of expected outputs
    """
    x = []
    y_true = []

    for i, line in enumerate(data):
        if line != "" and type(line) == str:
            
            words = line.split()

            splittable_words = []
            
            # We only split within words since we will not need to predict whitespace
            #Since we know we will get minimum one character, of the word and will not predict whitespace, we will only split words of at least 3 characters
            for i,word in enumerate(words):
                if len(word) >= 3:
                    splittable_words.append((word, i))

            if len(splittable_words) > 0:

                #Pick a random word to split
                split_word_index = random.randint(0, len(splittable_words)-1)
                
                #Get the index and the word to split
                to_split, og_index = splittable_words[split_word_index]

                split_index = random.randint(1, len(to_split)-1) #Get a spot to split. It cannot be first or last character
                
                #split the word
                splitted = to_split[0:split_index]

                # Record the expected y value
                y = to_split[split_index]

                # Split our words to only get ones that come before the split word
                res = words[0:og_index]

                #Add back in the word we split
                res.append(splitted)

                x.append(" ".join(res))
                y_true.append(y)
        

    return x, y_true

def write_test_data(test_data_dir, x_test, y_test):
    with open(f'{test_data_dir}/input.txt', 'w') as f:
        for item in x_test:
            f.write(f"{item}\n")

    with open(f'{test_data_dir}/answer.txt', 'w') as f:
        for item in y_test:
            f.write(f"{item}\n")

def discover_test_files(data_directory):
    """
    Discover test files automatically based on naming convention.
    Returns a dictionary mapping language codes to their test files with full paths.
    """
    test_files = {}
    
    # Get all CSV files in the data directory and its subdirectories
    if not os.path.exists(data_directory):
        print(f"Warning: Data directory {data_directory} does not exist.")
        return test_files
    
    # First check for English test files in the root data directory
    for filename in os.listdir(data_directory):
        if filename in ['test_nasa.csv', 'test_trek.csv']:
            # Handle English test files
            if 'en' not in test_files:
                test_files['en'] = []
            test_files['en'].append(os.path.join(data_directory, filename))
    
    # Now search in script subdirectories
    for script_dir in os.listdir(data_directory):
        script_path = os.path.join(data_directory, script_dir)
        
        # Skip if not a directory
        if not os.path.isdir(script_path):
            continue
            
        # If script_dir is itself a directory with subdirectories (like latin)
        if script_dir in ["latin"]:
            for subfolder in os.listdir(script_path):
                subfamily_path = os.path.join(script_path, subfolder)
                if os.path.isdir(subfamily_path):
                    for filename in os.listdir(subfamily_path):
                        if filename.startswith('test_culturax_') and filename.endswith('.csv'):
                            # Extract language code from test_culturax_XX.csv
                            lang_code = filename.replace('test_culturax_', '').replace('.csv', '')
                            if lang_code not in test_files:
                                test_files[lang_code] = []
                            test_files[lang_code].append(os.path.join(subfamily_path, filename))
        
        # For other script directories
        else:
            for filename in os.listdir(script_path):
                if filename.startswith('test_culturax_') and filename.endswith('.csv'):
                    # Extract language code from test_culturax_XX.csv
                    lang_code = filename.replace('test_culturax_', '').replace('.csv', '')
                    if lang_code not in test_files:
                        test_files[lang_code] = []
                    test_files[lang_code].append(os.path.join(script_path, filename))
    
    return test_files

if __name__ == '__main__':
    print("YOU ARE HERE")
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_directory', help='Directory containing clean data', default='data/parsed_data')
    parser.add_argument('--test_data', help='path to write test data', default='test/')
    args = parser.parse_args()
    
    print("HERE")

    # Load combined data
    loader = DatasetFileLoader()
    loader.load(args.data_directory)
    loader.test_data.head()

    loader.test_data = loader.test_data.drop(index=0)

    # Automatically discover test files by language
    test_data_by_language = discover_test_files(args.data_directory)
    
    print(f"Discovered test files for languages: {list(test_data_by_language.keys())}")

    # Process test data for each language
    for lang_code, csv_files in test_data_by_language.items():
        print(f"Processing {lang_code} test data...")
        
        # Create language-specific directory using language code
        language_dir = os.path.join(args.test_data, lang_code)
        os.makedirs(language_dir, exist_ok=True)
        
        # Combine all test samples for this language
        all_x_test = []
        all_y_test = []
        
        # Load and process each CSV file for the language
        for file_path in csv_files:
            # Skip if file doesn't exist
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} does not exist. Skipping...")
                continue
                
            # Load the CSV file
            data_df = pd.read_csv(file_path)
            
            # Generate test samples
            x_test, y_test = to_sample_and_expected_result(data_df.iloc[:, 0])
            
            all_x_test.extend(x_test)
            all_y_test.extend(y_test)
            print(f"  Added {len(x_test)} test samples from {os.path.basename(file_path)}")
        
        # Write the combined test data for this language
        if all_x_test:
            write_test_data(language_dir, all_x_test, all_y_test)
            print(f"  Total: {len(all_x_test)} test samples for {lang_code}")
        else:
            print(f"  No test samples generated for {lang_code}")
        
    # Also process the combined test data as before
    # Since loader.load was called with the data directory, it should have already loaded
    # all test files from their respective language family subdirectories
    combined_test_data = loader.test_data[0] if not loader.test_data.empty else []
    
    if not loader.test_data.empty:
        x_test, y_test = to_sample_and_expected_result(combined_test_data)
        write_test_data(args.test_data, x_test, y_test)
        print(f"Wrote {len(x_test)} test samples for combined data")
    else:
        print("No combined test data available")