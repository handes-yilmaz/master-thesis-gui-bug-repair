#!/usr/bin/env python3
"""
Aggregate all metrics from experiment results.
Outputs: comprehensive CSV with all data needed for analysis.
"""

import json
import pandas as pd
from pathlib import Path
import sys

# Pricing (per 1M tokens as of October 2024)
PRICING = {
    'gpt-4o-2024-08-06': {
        'prompt': 2.50,    # USD per 1M input tokens
        'completion': 10.00  # USD per 1M output tokens
    },
    'gpt-4o-mini-2024-07-18': {
        'prompt': 0.150,   # USD per 1M input tokens
        'completion': 0.600  # USD per 1M output tokens
    }
}

def extract_instance_metrics(instance_dir, model_name, modality, resolved_instances):
    """Extract all metrics from a single instance directory."""
    instance_id = instance_dir.name
    
    # Initialize result
    result = {
        'instance_id': instance_id,
        'model': model_name,
        'modality': modality,
        'resolved': instance_id in resolved_instances
    }
    
    # 1. Load research_metrics.json
    research_file = instance_dir / 'research_metrics.json'
    if research_file.exists():
        with open(research_file) as f:
            research = json.load(f)
        
        # Basic info
        result['domain'] = research.get('domain', 'unknown')
        result['visual_complexity'] = research.get('visual_complexity', None)
        result['bug_description_length'] = research.get('bug_description_length', 0)
        result['total_execution_time'] = research.get('total_execution_time', 0)
        
        # Phase metrics
        phases = research.get('phases', {})
        for phase_name in ['file_localization', 'element_localization', 'patch_generation']:
            phase = phases.get(phase_name, {})
            result[f'{phase_name}_success'] = phase.get('success', False)
            result[f'{phase_name}_duration'] = phase.get('duration', 0)
            result[f'{phase_name}_tokens'] = phase.get('tokens_used', 0)
        
        # Visual analysis (MM only)
        visual = research.get('visual_analysis', {})
        if visual:
            result['num_images'] = visual.get('num_images', 0)
            result['avg_ui_elements'] = visual.get('avg_ui_elements', 0)
            result['avg_complexity_score'] = visual.get('avg_complexity_score', 0)
            result['text_complexity_score'] = visual.get('text_complexity_score', 0)
        else:
            result['num_images'] = 0
            result['avg_ui_elements'] = 0
            result['avg_complexity_score'] = 0
            result['text_complexity_score'] = 0
    
    # 2. Load token_usage.json
    token_file = instance_dir / 'token_usage.json'
    if token_file.exists():
        with open(token_file) as f:
            token_data = json.load(f)
        
        # Aggregate tokens per phase
        total_prompt = 0
        total_completion = 0
        
        for phase_name, usage in token_data.items():
            prompt = usage.get('prompt_tokens', 0)
            completion = usage.get('completion_tokens', 0)
            total_prompt += prompt
            total_completion += completion
        
        result['total_prompt_tokens'] = total_prompt
        result['total_completion_tokens'] = total_completion
        result['total_tokens'] = total_prompt + total_completion
        
        # Calculate cost
        if model_name in PRICING:
            model_pricing = PRICING[model_name]
            result['cost_usd'] = (
                (total_prompt / 1_000_000) * model_pricing['prompt'] +
                (total_completion / 1_000_000) * model_pricing['completion']
            )
        else:
            result['cost_usd'] = 0
    else:
        result['total_prompt_tokens'] = 0
        result['total_completion_tokens'] = 0
        result['total_tokens'] = 0
        result['cost_usd'] = 0
    
    # 3. Check if patch exists
    diff_file = instance_dir / 'changes.diff'
    if diff_file.exists():
        patch_content = diff_file.read_text()
        result['has_patch'] = bool(patch_content.strip())
        result['patch_size_bytes'] = len(patch_content)
        result['patch_lines'] = len(patch_content.splitlines())
    else:
        result['has_patch'] = False
        result['patch_size_bytes'] = 0
        result['patch_lines'] = 0
    
    return result

def main():
    """Main aggregation function."""
    print("="*80)
    print("AGGREGATING METRICS FROM EXPERIMENT RESULTS")
    print("="*80)
    print()
    
    # Get results directory from command line or use default
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        results_dir = Path('../results')
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    print(f"Results directory: {results_dir}")
    print()
    
    all_data = []
    instance_count = 0
    
    # Traverse results structure
    for split_dir in results_dir.iterdir():
        if not split_dir.is_dir():
            continue
        
        for repo_dir in split_dir.iterdir():
            if not repo_dir.is_dir():
                continue
            
            for instance_dir in repo_dir.iterdir():
                if not instance_dir.is_dir():
                    continue
                
                try:
                    # Determine model and modality from results structure
                    # This is a simple heuristic - you may need to adjust based on your structure
                    model_name = 'gpt-4o-2024-08-06'  # Default
                    modality = 'multimodal'
                    
                    # Try to infer from parent directory or other markers
                    if 'textonly' in str(instance_dir):
                        modality = 'text-only'
                    if 'mini' in str(results_dir):
                        model_name = 'gpt-4o-mini-2024-07-18'
                    
                    # For now, use a placeholder for resolved instances
                    # You would load this from your evaluation results
                    resolved_instances = set()
                    
                    metrics = extract_instance_metrics(
                        instance_dir, 
                        model_name, 
                        modality, 
                        resolved_instances
                    )
                    all_data.append(metrics)
                    instance_count += 1
                except Exception as e:
                    print(f"  ⚠️  Error processing {instance_dir.name}: {e}")
    
    if not all_data:
        print("❌ No data found!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Save to CSV
    output_file = results_dir / 'aggregated_metrics.csv'
    df.to_csv(output_file, index=False)
    
    print("="*80)
    print("AGGREGATION COMPLETE")
    print("="*80)
    print(f"Total records: {len(df)}")
    print(f"Output: {output_file}")
    print()
    print("Columns:", list(df.columns))
    print()
    print("Summary:")
    print(f"  Resolved: {df['resolved'].sum()} / {len(df)}")
    print(f"  Has patches: {df['has_patch'].sum()} / {len(df)}")
    print()
    print("="*80)
    print("✓ Ready for analysis!")
    print("="*80)

if __name__ == "__main__":
    main()

