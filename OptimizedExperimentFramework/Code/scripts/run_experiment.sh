#!/bin/bash

# Run GUIRepair on a single instance or repository
# Usage: bash run_experiment.sh <instance_id> [options]

# Configuration
MODEL="gpt-4o-2024-08-06"
DATASET="princeton-nlp/SWE-bench_Multimodal"
SPLIT="test"
REPO_PATH="../Data/Reproduce_Scenario"
OUTPUT_DIR="results"
API_KEY="${OPENAI_API_KEY}"

# Parse command line arguments
INSTANCE_ID="${1:-}"
TEXT_ONLY="${2:-false}"

if [ -z "$INSTANCE_ID" ]; then
    echo "Usage: bash run_experiment.sh <instance_id> [text_only]"
    echo ""
    echo "Examples:"
    echo "  bash run_experiment.sh bpmn-io__bpmn-js-1080"
    echo "  bash run_experiment.sh bpmn-io__bpmn-js-1080 true"
    echo "  bash run_experiment.sh bpmn-io  # All instances from repo"
    exit 1
fi

if [ -z "$API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable not set"
    echo "Set it with: export OPENAI_API_KEY=your_key"
    exit 1
fi

# Determine if it's a single instance or repo prefix
if [[ "$INSTANCE_ID" == *"__"* ]]; then
    MODE="--instance_id"
    echo "Running single instance: $INSTANCE_ID"
else
    MODE="--repo_prefix"
    echo "Running all instances from: $INSTANCE_ID"
fi

# Build command
CMD="python main.py \
    $MODE $INSTANCE_ID \
    --api_key $API_KEY \
    --model $MODEL \
    --dataset $DATASET \
    --split $SPLIT \
    --repo_path $REPO_PATH \
    --output_dir $OUTPUT_DIR"

# Add text-only flag if requested
if [ "$TEXT_ONLY" = "true" ]; then
    CMD="$CMD --text_only"
    echo "Mode: Text-only (no images)"
else
    echo "Mode: Multimodal (with images)"
fi

# Run
echo ""
echo "Starting GUIRepair..."
echo "=============================="
$CMD



