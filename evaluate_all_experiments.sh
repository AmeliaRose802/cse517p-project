#!/bin/bash

# Script to evaluate all experiment models
# Can be run from anywhere - automatically finds the project root

echo "=== Evaluating All Experiment Models ==="
echo "Started at: $(date)"
echo ""

# Determine project root directory (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Base directories
MODEL_EXPERIMENTS_DIR="$PROJECT_ROOT/model_experiments"
TEST_BASE_DIR="$PROJECT_ROOT/test"
EVAL_SCRIPT="$PROJECT_ROOT/src/evaluate_by_language.py"

echo "Project root: $PROJECT_ROOT"
echo ""

# Results file
RESULTS_FILE="$PROJECT_ROOT/evaluation_results_$(date +%Y%m%d_%H%M%S).txt"

# Check if evaluation script exists
if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "Error: Evaluation script not found at $EVAL_SCRIPT"
    exit 1
fi

# Check if test directory exists
if [ ! -d "$TEST_BASE_DIR" ]; then
    echo "Error: Test directory not found at $TEST_BASE_DIR"
    exit 1
fi

# Initialize results file
echo "Experiment Evaluation Results - $(date)" > "$RESULTS_FILE"
echo "============================================" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# Counter for successful evaluations
successful_count=0
failed_count=0

# List of experiment directories (in order)
experiment_dirs=(
    "exp_1_baseline"
    "exp_1_current_model" 
    "exp_1_more_layers"
    "exp_2_larger_embedding"
    "exp_2_tiny_embeddings"
    "exp_3_fewer_layers"
    "exp_4_more_layers"
    "exp_5_fewer_heads"
    "exp_6_more_heads"
    "exp_7_smaller_ffn"
    "exp_8_larger_ffn"
    "exp_9_higher_dropout"
    "exp_10_highest_dropout"
    "exp_12_small_model"
)

# Evaluate each experiment
for exp_dir in "${experiment_dirs[@]}"; do
    full_exp_path="$MODEL_EXPERIMENTS_DIR/$exp_dir"
    
    echo "----------------------------------------"
    echo "Evaluating: $exp_dir"
    echo "----------------------------------------"
    
    # Check if experiment directory exists
    if [ ! -d "$full_exp_path" ]; then
        echo "Warning: Directory $exp_dir not found, skipping..."
        echo "SKIPPED: $exp_dir - Directory not found" >> "$RESULTS_FILE"
        echo "" >> "$RESULTS_FILE"
        continue
    fi
    
    # Run the evaluation (change to project root directory first)
    echo "Running: python3 $EVAL_SCRIPT --work_dir $full_exp_path --test_base_dir $TEST_BASE_DIR"
    
    # Change to project root and run evaluation
    cd "$PROJECT_ROOT"
    if python3 "$EVAL_SCRIPT" --work_dir "$full_exp_path" --test_base_dir "$TEST_BASE_DIR" 2>&1 | tee -a "$RESULTS_FILE"; then
        echo "✅ SUCCESS: $exp_dir completed successfully"
        ((successful_count++))
    else
        echo "❌ FAILED: $exp_dir evaluation failed"
        echo "FAILED: $exp_dir - Evaluation error" >> "$RESULTS_FILE"
        ((failed_count++))
    fi
    
    echo "" >> "$RESULTS_FILE"
    echo ""
done

# Summary
echo "=========================================="
echo "EVALUATION SUMMARY"
echo "=========================================="
echo "Total experiments: ${#experiment_dirs[@]}"
echo "Successful: $successful_count"
echo "Failed: $failed_count"
echo "Results saved to: $RESULTS_FILE"
echo "Completed at: $(date)"

# Add summary to results file
echo "" >> "$RESULTS_FILE"
echo "========================================== " >> "$RESULTS_FILE"
echo "SUMMARY" >> "$RESULTS_FILE"
echo "==========================================" >> "$RESULTS_FILE"
echo "Total experiments: ${#experiment_dirs[@]}" >> "$RESULTS_FILE"
echo "Successful: $successful_count" >> "$RESULTS_FILE"
echo "Failed: $failed_count" >> "$RESULTS_FILE"
echo "Completed at: $(date)" >> "$RESULTS_FILE"
