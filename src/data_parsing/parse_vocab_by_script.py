# Count frequency of all charactres in cjk language family
# Display number of unque chars for each language 
# Build a vocab list that contains 95% of the probility mass of characters for each language

import os
import glob
import collections
import argparse
import json
from typing import Dict, List, Tuple


def parse_arguments():
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../..'))
    
    parser = argparse.ArgumentParser(description='Process CJK language text to build vocabulary lists')
    parser.add_argument('--data_dir', type=str, default=os.path.join(project_root, 'data/parsed_data'),
                        help='Directory containing language data files')
    parser.add_argument('--output_dir', type=str, default=os.path.join(project_root, 'data/vocab_files'),
                        help='Directory to save output vocabulary files')
    parser.add_argument('--threshold', type=float, default=0.95,
                        help='Probability mass threshold for vocabulary')
    return parser.parse_args()


def get_files_by_language(data_dir: str) -> Dict[str, Dict[str, List[str]]]:
    """Get text files organized by script group and language."""
    script_language_files = {}
    
    # Scan all subdirectories in the data directory
    for script_dir in os.listdir(data_dir):
        script_path = os.path.join(data_dir, script_dir)
        
        # Skip if not a directory
        if not os.path.isdir(script_path):
            continue
            
        # Initialize script group dictionary
        if script_dir not in script_language_files:
            script_language_files[script_dir] = {}
            
        # Find all CSV files in this script directory
        csv_files = glob.glob(os.path.join(script_path, '*.csv'))
        
        # Group files by language code (assuming format: *_LANGCODE.csv)
        for file_path in csv_files:
            filename = os.path.basename(file_path)
            # Extract language code before the extension
            parts = filename.split('_')
            if len(parts) > 1:
                lang_code = parts[-1].split('.')[0]  # Get language code before .csv
                
                if lang_code not in script_language_files[script_dir]:
                    script_language_files[script_dir][lang_code] = []
                    
                script_language_files[script_dir][lang_code].append(file_path)
    
    return script_language_files


def count_characters(files: List[str]) -> collections.Counter:
    """Count character frequencies in a list of files."""
    counter = collections.Counter()
    import string
    punctuation_set = set(string.punctuation)
    
    for file_path in files:
        try:
            if file_path.endswith('.csv'):
                # For CSV files, read line by line and extract text content
                with open(file_path, 'r', encoding='utf-8') as f:
                    # Skip header if present
                    header = f.readline()
                    for line in f:
                        # Assuming the text content is in the second column (index 1)
                        # Adjust this if the CSV structure is different
                        try:
                            parts = line.strip().split(',')
                            if len(parts) > 1:
                                text = parts[1].lower()  # Convert to lowercase
                                # Filter out punctuation but keep all script characters
                                filtered_text = ''.join(ch for ch in text if ch not in punctuation_set)
                                counter.update(filtered_text)
                        except Exception as e:
                            print(f"Error processing line in {file_path}: {e}")
            else:
                # For regular text files
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read().lower()  # Convert to lowercase
                    # Filter out punctuation but keep all script characters
                    filtered_text = ''.join(ch for ch in text if ch not in punctuation_set)
                    counter.update(filtered_text)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    return counter


def build_vocab(counter: collections.Counter, threshold: float) -> Tuple[List[str], float]:
    """Build a vocabulary that covers the given probability mass threshold."""
    total_count = sum(counter.values())
    
    # Sort characters by frequency
    sorted_chars = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    
    vocab = []
    cumulative_prob = 0.0
    
    for char, count in sorted_chars:
        # Skip whitespace, non-printable, and uppercase characters
        if char.isspace() or not char.isprintable() or char.isupper():
            continue
            
        prob = count / total_count
        cumulative_prob += prob
        vocab.append(char)
        
        if cumulative_prob >= threshold:
            break
    
    coverage = cumulative_prob
    return vocab, coverage


def process_language(language: str, files: List[str], threshold: float) -> Dict:
    """Process a language and return vocabulary statistics."""
    print(f"Processing {language} language...")
    
    # Count character frequencies
    counter = count_characters(files)
    
    # Remove non-printable characters and whitespace
    for char in list(counter.keys()):
        if char.isspace() or not char.isprintable():
            del counter[char]
    
    # Get unique character count
    unique_chars = len(counter)
    
    # Build vocabulary
    vocab, coverage = build_vocab(counter, threshold)
    
    # Calculate statistics
    vocab_size = len(vocab)
    vocab_percentage = (vocab_size / unique_chars) * 100 if unique_chars > 0 else 0
    
    return {
        'language': language,
        'unique_characters': unique_chars,
        'vocabulary_size': vocab_size,
        'vocabulary_percentage': vocab_percentage,
        'coverage': coverage,
        'vocabulary': vocab
    }


def process_script_group(script_group: str, language_files: Dict[str, List[str]], threshold: float) -> Dict:
    """Process all languages in a script group and return combined vocabulary statistics."""
    print(f"Processing {script_group} script group...")
    
    all_language_results = {}
    combined_counter = collections.Counter()
    
    # Process each language in the script group
    for language, files in language_files.items():
        if not files:
            print(f"No files found for {language} in {script_group}")
            continue
            
        # Count characters for this language
        language_counter = count_characters(files)
        
        # Clean the counter (remove non-printable and whitespace)
        for char in list(language_counter.keys()):
            if char.isspace() or not char.isprintable():
                del language_counter[char]
        
        # Store language-specific statistics
        unique_chars = len(language_counter)
        all_language_results[language] = {
            'language_code': language,
            'unique_characters': unique_chars,
            'file_count': len(files)
        }
        
        # Add to combined counter for the script
        combined_counter.update(language_counter)
    
    # Build combined vocabulary for the script group
    vocab, coverage = build_vocab(combined_counter, threshold)
    
    # Calculate overall statistics
    total_unique_chars = len(combined_counter)
    vocab_size = len(vocab)
    vocab_percentage = (vocab_size / total_unique_chars) * 100 if total_unique_chars > 0 else 0
    
    return {
        'script_group': script_group,
        'languages': all_language_results,
        'unique_characters_total': total_unique_chars,
        'vocabulary_size': vocab_size,
        'vocabulary_percentage': vocab_percentage,
        'coverage': coverage,
        'vocabulary': vocab
    }


def main():
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get files organized by script group and language
    script_language_files = get_files_by_language(args.data_dir)
    
    # Process each script group
    results = {}
    for script_group, language_files in script_language_files.items():
        if not language_files:
            print(f"No language files found for {script_group}")
            continue
            
        script_result = process_script_group(script_group, language_files, args.threshold)
        results[script_group] = script_result
        
        # Save vocabulary to file
        output_file = os.path.join(args.output_dir, f'{script_group}_vocab.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(script_result, f, ensure_ascii=False, indent=2)
    
    # Display summary
    print("\nSummary:")
    print("=" * 80)
    print(f"{'Script Group':<15} {'Languages':<25} {'Total Chars':<15} {'Vocab Size':<15} {'Coverage':<10}")
    print("-" * 80)
    
    for script_group, result in results.items():
        languages = ", ".join(result['languages'].keys())
        print(f"{script_group:<15} {languages:<25} {result['unique_characters_total']:<15} {result['vocabulary_size']:<15} {result['coverage']:.4f}")
    
    print("=" * 80)
    print(f"Vocabulary files saved to {args.output_dir}")


if __name__ == "__main__":
    main()