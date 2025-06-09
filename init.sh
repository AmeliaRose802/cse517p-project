#!/bin/bash

# Exit on error
set -e

# Create a virtual environment
echo "Creating Python virtual environment..."
python -m venv venv

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
if [ -f requirements.txt ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found. Installing basic requirements..."
    pip install numpy pandas torch matplotlib scikit-learn
fi

# Create necessary directories
mkdir -p data/parsed_data
mkdir -p test

# Download datasets for all languages
echo "Downloading multilingual datasets..."
python ./src/data_parsing/prepare_multilingual_dataset.py

# Copy additional English files
echo "Copying additional English data files..."
mkdir -p data/parsed_data/latin
if [ -d "data/english_persistent" ]; then
    cp -r data/english_persistent/* data/parsed_data/latin/
    echo "Additional English files copied successfully"
else
    echo "Warning: Additional English data directory not found at data/english_persistent"
fi

# Set up test data folders
echo "Setting up test data folders..."
if [ ! -f "test/input.txt" ] || [ ! -f "test/answer.txt" ]; then
    python src/data_parsing/prepare_test_data.py --data_directory data/parsed_data --test_data cse517p-project/test
    echo "Test data setup complete."
else
    echo "Test data already exists, skipping setup."
fi

# Create vocabulary files if they don't exist
if [ ! -f "data/vocab_files/latin_vocab.txt" ]; then
    echo "Creating vocabulary files..."
    python src/data_parsing/parse_vocab_by_script.py --threshold 99
else
    echo "Vocabulary files already exist, skipping creation."
fi

echo "Setup complete. To activate the environment in the future, run: source venv/bin/activate"
