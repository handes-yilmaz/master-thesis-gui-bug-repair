# GUIRepair: Optimized Experiment Framework

A 3-phase framework for automated GUI bug repair using large language models with multimodal analysis (text + images).

## 1. Repository Checkout

Install all SWE-bench M task instances by running:

```bash
cd Data && python get_all_reproduce_scenario.py
```

## 2. Issue Resolution

Run the repair workflow to generate patches for SWE-bench M instances.

### Running Experiments

Navigate to the Code directory:

```bash
cd Code
```

#### Using the Script (Recommended)

```bash
bash scripts/run_experiment.sh
```

This script requires the `OPENAI_API_KEY` environment variable to be set:

```bash
export OPENAI_API_KEY=your_api_key
```

#### Using Python Directly

##### Single Instance

```bash
python main.py --instance_id bpmn-io__bpmn-js-1080 --api_key YOUR_API_KEY
```

##### Repository Subset

```bash
python main.py --repo_prefix bpmn-io --api_key YOUR_API_KEY
```

##### All Instances

```bash
python main.py --all --api_key YOUR_API_KEY
```

### Experiment Modes

#### Multimodal Mode (Default)
Includes both text and image analysis:

```bash
python main.py --instance_id bpmn-io__bpmn-js-1080 --api_key YOUR_API_KEY
```

#### Text-Only Mode
Disables image analysis:

```bash
python main.py --instance_id bpmn-io__bpmn-js-1080 --text_only --api_key YOUR_API_KEY
```

### Model Options

Use different LLM models:

```bash
# GPT-4o (default)
python main.py --instance_id example --model gpt-4o-2024-08-06 --api_key YOUR_API_KEY

# GPT-4o-mini
python main.py --instance_id example --model gpt-4o-mini-2024-07-18 --api_key YOUR_API_KEY
```

## 3. Result Evaluation

Run evaluation using sb-cli to get repair results.

### Install sb-cli

Follow the [sb-cli installation guide](https://github.com/swe-bench/sb-cli).

### Run Evaluation

Navigate to results directory and run evaluation scripts:

```bash
cd results

# For GPT-4o results
bash val_gpt4o.sh

# For GPT-4o-mini results  
bash val_o4mini.sh
```

Evaluation reports will be generated in the evaluation_results directory.

### Aggregate and Analyze Results

Generate aggregated metrics and tables:

```bash
cd Code/scripts

# Aggregate all metrics
python aggregate_metrics.py ../results

# Generate analysis tables
python generate_tables.py ../results/aggregated_metrics.csv
```

Output files will be in `../results/paper_outputs/`

## Output Files

For each instance, results are saved in:

```
results/{split}/{repo}/{instance_id}/
├── research_metrics.json    # Metrics (domain, complexity, timing, tokens)
├── token_usage.json         # Per-phase token breakdown
├── changes.diff             # Generated patch
└── workflow_results.json    # Overall results
```

## Requirements

```bash
pip install openai anthropic pandas numpy opencv-python pillow datasets pathlib
```
