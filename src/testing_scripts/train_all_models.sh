#!/bin/bash
# filepath: /home/user/cse517p-project/train_all_models.sh

# Create base directories
mkdir -p ./work_models

# Define character sets to train
CHARSETS=("all" "latin" "cyrillic" "cjk" "arabic" "devanagari")

# Train individual models for each character set
for charset in "${CHARSETS[@]}"; do
  echo "=== Training model for character set: $charset ==="
  
  # Create dedicated output directory
  OUTPUT_DIR="./work_models/${charset}"
  mkdir -p "$OUTPUT_DIR"
  
  # Run training
  python ./src/train.py \
    --work_dir "$OUTPUT_DIR" \
    --data_dir ./data/parsed_data \
    --data_fraction 1 \
    --time \
    --charset "$charset"
    
  echo "=== Finished training $charset model ==="
  echo ""
done

# Train English + Cyrillic combination model
echo "=== Training combination model: English + Cyrillic ==="
OUTPUT_DIR="./work_models/english_cyrillic"
mkdir -p "$OUTPUT_DIR"

python ./src/train.py \
  --work_dir "$OUTPUT_DIR" \
  --data_dir ./data/parsed_data \
  --data_fraction 1 \
  --time \
  --charset "english,cyrillic"
  
echo "=== Finished training English + Cyrillic model ==="
echo ""

echo "All training complete! Models saved to ./work_models/"