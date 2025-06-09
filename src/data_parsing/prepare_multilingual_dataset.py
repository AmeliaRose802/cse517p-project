from datasets import load_dataset
import os
import sys
import csv

# Add project root to Python path to ensure 'src' package is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Now we can import from src
from src.language_consts import LANGUAGES, get_language_famlies

# Define the languages to process - use the keys from LANG_FAMLIES
langs = list(LANGUAGES.keys())

# Define number of lines per language based on script family
DEFAULT_LINES = 10000

output_dir = "data/parsed_data"

# Function to check if all split files exist for a language
def files_exist_for_language(lang):
    """Check if train, dev, and test files already exist for a language"""
    script = LANGUAGES.get(lang).get("script", "other")
    script_dir = os.path.join(output_dir, script)
    for split in ["train", "dev", "test"]:
        filename = os.path.join(script_dir, f"{split}_culturax_{lang}.csv")
        if not os.path.exists(filename):
            return False
    return True

def main():
    # Check to ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create subdirectories for each script group
    scripts = set(lang_data["script"] for lang_data in LANGUAGES.values())
    for script in scripts:
        script_dir = os.path.join(output_dir, script)
        os.makedirs(script_dir, exist_ok=True)

    # Filter languages to only include those that need to be processed
    langs_to_process = [lang for lang in langs if not files_exist_for_language(lang)]

    if not langs_to_process:
        print("All languages have already been processed. No downloads needed.")
        sys.exit(0)

    print(f"Languages to process: {langs_to_process}")
    print(f"Skipping already processed languages: {[lang for lang in langs if lang not in langs_to_process]}")

    # Prepare separate writers for train/dev/test splits for each language
    lang_split_writers = {}
    lang_split_files = {}

    # Initialize writers for each language and split combination
    for lang in langs_to_process:
        lang_split_writers[lang] = {}
        lang_split_files[lang] = {}
        
        script = LANGUAGES.get(lang).get("script", "other")
        script_dir = os.path.join(output_dir, script)
        
        for split in ["train", "dev", "test"]:
            filename = os.path.join(script_dir, f"{split}_culturax_{lang}.csv")
            f = open(filename, "w", encoding="utf-8", newline="")
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["dialogue"])
            lang_split_files[lang][split] = f
            lang_split_writers[lang][split] = writer

    # Write each language's data to separate files
    # Not downloading all the data at once as it is time consuming. Lets stream the data instead based on the language and sample the required number of lines.
    try:
        for lang in langs_to_process:
            print(f"Streaming CulturaX: {lang}")
            dataset = load_dataset("uonlp/CulturaX", lang, streaming=True)
            stream = dataset["train"]
            
            count = 0
            lines_for_lang = LANGUAGES[lang]["line-multiplier"]*DEFAULT_LINES
            for example in stream:
                text = example.get("text", "").strip()
                if text:
                    text = text.replace("\n", " ")
                    if count < int(lines_for_lang * 0.8):
                        lang_split_writers[lang]["train"].writerow([text])
                    elif count < int(lines_for_lang * 0.9):
                        lang_split_writers[lang]["dev"].writerow([text])
                    elif count < lines_for_lang:
                        lang_split_writers[lang]["test"].writerow([text])
                    else:
                        break
                    count += 1

            print(f"Collected {count} lines for {lang} (requested: {lines_for_lang})")
            if count < lines_for_lang:
                print(f"Warning: Only {count} lines available for {lang}, less than requested {lines_for_lang}")

        print("Language-specific train/dev/test files created and saved:")
        for lang in langs_to_process:
            script = LANGUAGES.get(lang).get("script", "other")
            script_dir = os.path.join(output_dir, script)
            for split in ["train", "dev", "test"]:
                filename = os.path.join(script_dir, f"{split}_culturax_{lang}.csv")
                print(f" - {lang} {split}: {filename}")

    except Exception as e:
        print(f"An error occurred while processing the dataset: {e}")
        sys.exit(1)

    finally:
        for lang in langs_to_process:
            for split in ["train", "dev", "test"]:
                if lang in lang_split_files and split in lang_split_files[lang]:
                    lang_split_files[lang][split].close()

# Only run the main code when the script is executed directly, not when imported
if __name__ == "__main__":
    main()
