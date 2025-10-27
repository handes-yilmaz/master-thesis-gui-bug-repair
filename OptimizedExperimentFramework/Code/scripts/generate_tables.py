#!/usr/bin/env python3
"""
Generate tables and figures for results analysis.
Outputs ready-to-use CSV data for tables and plotting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

def bootstrap_ci(data, n_boot=10000, ci=95):
    """Bootstrap confidence interval for proportion."""
    n = len(data)
    successes = data.sum()
    boot_props = []
    for _ in range(n_boot):
        boot_sample = np.random.choice(data, size=n, replace=True)
        boot_props.append(boot_sample.mean())
    lower = np.percentile(boot_props, (100 - ci) / 2)
    upper = np.percentile(boot_props, 100 - (100 - ci) / 2)
    return lower, upper

def main():
    """Main analysis function."""
    print("="*80)
    print("GENERATING TABLES AND FIGURES")
    print("="*80)
    print()
    
    # Get input file from command line or use default
    if len(sys.argv) > 1:
        input_file = Path(sys.argv[1])
    else:
        input_file = Path('../results/aggregated_metrics.csv')
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        print("Usage: python generate_tables.py [aggregated_metrics.csv]")
        return
    
    print(f"Input: {input_file}")
    
    # Load data
    df = pd.read_csv(input_file)
    
    # Create output directory
    output_dir = input_file.parent / 'paper_outputs'
    output_dir.mkdir(exist_ok=True)
    
    print(f"Output: {output_dir}")
    print()
    
    # ============================================================================
    # OVERALL ACCURACY
    # ============================================================================
    print("="*80)
    print("OVERALL ACCURACY")
    print("="*80)
    print()
    
    # Group by model and modality
    accuracy_data = []
    
    for (model, modality), group_df in df.groupby(['model', 'modality']):
        resolved_count = group_df['resolved'].sum()
        total = len(group_df)
        rate = resolved_count / total * 100 if total > 0 else 0
        
        # Bootstrap CI
        lower, upper = bootstrap_ci(group_df['resolved'].values)
        
        accuracy_data.append({
            'Model': model,
            'Modality': modality,
            'Resolved': resolved_count,
            'Total': total,
            'Rate (%)': f'{rate:.2f}',
            'CI Lower (%)': f'{lower*100:.2f}',
            'CI Upper (%)': f'{upper*100:.2f}',
            '95% CI': f'[{lower*100:.2f}, {upper*100:.2f}]'
        })
    
    accuracy_df = pd.DataFrame(accuracy_data)
    accuracy_df.to_csv(output_dir / 'table_overall_accuracy.csv', index=False)
    print("✓ Overall Accuracy Table")
    print(accuracy_df.to_string(index=False))
    print()
    
    # ============================================================================
    # PER-DOMAIN ANALYSIS
    # ============================================================================
    print("="*80)
    print("PER-DOMAIN ANALYSIS")
    print("="*80)
    print()
    
    if 'domain' in df.columns:
        domain_data = []
        
        for domain in sorted(df['domain'].unique()):
            domain_df = df[df['domain'] == domain]
            
            row = {'Domain': domain}
            
            for (model, modality), group_df in domain_df.groupby(['model', 'modality']):
                resolved = group_df['resolved'].sum()
                total = len(group_df)
                rate = resolved / total * 100 if total > 0 else 0
                
                key = f"{model}_{modality}"
                row[f'{key}_resolved'] = resolved
                row[f'{key}_rate'] = f'{rate:.1f}'
            
            domain_data.append(row)
        
        if domain_data:
            domain_df_result = pd.DataFrame(domain_data)
            domain_df_result.to_csv(output_dir / 'table_domain_accuracy.csv', index=False)
            print("✓ Domain-wise Accuracy")
            print(domain_df_result.to_string(index=False))
            print()
    
    # ============================================================================
    # VISUAL COMPLEXITY ANALYSIS
    # ============================================================================
    print("="*80)
    print("VISUAL COMPLEXITY ANALYSIS")
    print("="*80)
    print()
    
    if 'visual_complexity' in df.columns:
        # Only for multimodal
        mm_df = df[df['modality'] == 'multimodal']
        
        if len(mm_df) > 0:
            complexity_data = []
            
            for complexity in ['simple', 'medium', 'complex']:
                comp_df = mm_df[mm_df['visual_complexity'] == complexity]
                
                if len(comp_df) == 0:
                    continue
                
                for model, group_df in comp_df.groupby('model'):
                    resolved = group_df['resolved'].sum()
                    total = len(group_df)
                    rate = resolved / total * 100 if total > 0 else 0
                    
                    complexity_data.append({
                        'Complexity': complexity,
                        'Model': model,
                        'Resolved': resolved,
                        'Total': total,
                        'Rate (%)': f'{rate:.1f}'
                    })
            
            if complexity_data:
                complexity_df = pd.DataFrame(complexity_data)
                complexity_df.to_csv(output_dir / 'table_complexity_accuracy.csv', index=False)
                print("✓ Visual Complexity Analysis")
                print(complexity_df.to_string(index=False))
                print()
    
    # ============================================================================
    # COST ANALYSIS
    # ============================================================================
    print("="*80)
    print("COST ANALYSIS")
    print("="*80)
    print()
    
    if 'cost_usd' in df.columns and 'total_tokens' in df.columns:
        cost_data = []
        
        for (model, modality), group_df in df.groupby(['model', 'modality']):
            total_cost = group_df['cost_usd'].sum()
            total_tokens = group_df['total_tokens'].sum()
            avg_cost_per_instance = group_df['cost_usd'].mean()
            
            cost_data.append({
                'Model': model,
                'Modality': modality,
                'Total Cost (USD)': f'${total_cost:.2f}',
                'Total Tokens': f'{total_tokens:,}',
                'Avg Cost/Instance': f'${avg_cost_per_instance:.4f}'
            })
        
        cost_df = pd.DataFrame(cost_data)
        cost_df.to_csv(output_dir / 'table_cost_analysis.csv', index=False)
        print("✓ Cost Analysis")
        print(cost_df.to_string(index=False))
        print()
    
    print("="*80)
    print("✓ ALL TABLES GENERATED")
    print("="*80)
    print()
    print(f"Output directory: {output_dir}")
    print(f"Files: {len(list(output_dir.glob('*.csv')))} CSV files created")

if __name__ == "__main__":
    main()

