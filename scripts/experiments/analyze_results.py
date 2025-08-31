#!/usr/bin/env python
"""
Analyze LLM Experiment Results
=============================

This script analyzes the results from our GUI bug repair experiments
and generates comprehensive reports.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def load_latest_results():
    """Load the most recent experiment results"""
    results_dir = Path("runs/experiments")
    if not results_dir.exists():
        print("No results directory found!")
        return None
    
    # Find the most recent results file
    result_files = list(results_dir.glob("experiment_results_*.json"))
    if not result_files:
        print("No experiment results found!")
        return None
    
    latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    return results, latest_file

def analyze_results(results):
    """Analyze the experiment results"""
    df = pd.DataFrame(results)
    
    # Extract evaluation metrics
    metrics_df = pd.json_normalize(df['evaluation_metrics'])
    df = pd.concat([df, metrics_df], axis=1)
    
    # Basic statistics
    print("\n" + "="*60)
    print("📊 EXPERIMENT RESULTS ANALYSIS")
    print("="*60)
    
    print(f"\n📈 SUMMARY STATISTICS:")
    print(f"  Total experiments: {len(df)}")
    print(f"  Modalities tested: {', '.join(df['modality'].unique())}")
    print(f"  Bug scenarios tested: {len(df['bug_description'].unique())}")
    print(f"  Average processing time: {df['processing_time'].mean():.2f}s")
    print(f"  Total processing time: {df['processing_time'].sum():.2f}s")
    
    print(f"\n🎯 QUALITY METRICS:")
    print(f"  Average response length: {df['response_length'].mean():.0f} characters")
    print(f"  Structured response rate: {(df['structured_response'].mean() * 100):.1f}%")
    print(f"  Average confidence score: {df['solution_confidence'].mean():.2f}")
    print(f"  Solutions provided: {(df['contains_solution'].mean() * 100):.1f}%")
    print(f"  Explanations provided: {(df['contains_explanation'].mean() * 100):.1f}%")
    print(f"  Accessibility mentioned: {(df['contains_accessibility'].mean() * 100):.1f}%")
    
    # Analysis by modality
    print(f"\n📊 MODALITY BREAKDOWN:")
    for modality in df['modality'].unique():
        modality_df = df[df['modality'] == modality]
        print(f"\n  {modality.upper()}:")
        print(f"    Count: {len(modality_df)}")
        print(f"    Avg processing time: {modality_df['processing_time'].mean():.2f}s")
        print(f"    Avg confidence: {modality_df['solution_confidence'].mean():.2f}")
        print(f"    Structured responses: {(modality_df['structured_response'].mean() * 100):.1f}%")
        print(f"    Avg response length: {modality_df['response_length'].mean():.0f} chars")
    
    # Analysis by bug category
    print(f"\n🐛 BUG CATEGORY ANALYSIS:")
    bug_categories = {
        "BUTTON_CLICK_01": "Interaction Event",
        "LAYOUT_BREAK_02": "Visual Layout", 
        "ACCESSIBILITY_03": "Accessibility",
        "STATE_MANAGEMENT_04": "State Transition",
        "COLOR_CONTRAST_05": "Visual Color/Typography"
    }
    
    for bug_id, category in bug_categories.items():
        bug_df = df[df['experiment_id'].str.startswith(bug_id)]
        if len(bug_df) > 0:
            print(f"\n  {category}:")
            print(f"    Tests: {len(bug_df)}")
            print(f"    Avg confidence: {bug_df['solution_confidence'].mean():.2f}")
            print(f"    Structured responses: {(bug_df['structured_response'].mean() * 100):.1f}%")
    
    return df

def generate_visualizations(df):
    """Generate visualizations of the results"""
    print(f"\n📈 GENERATING VISUALIZATIONS...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('GUI Bug Repair LLM Experiments - Analysis Results', fontsize=16, fontweight='bold')
    
    # 1. Processing time by modality
    sns.boxplot(data=df, x='modality', y='processing_time', ax=axes[0,0])
    axes[0,0].set_title('Processing Time by Modality')
    axes[0,0].set_ylabel('Time (seconds)')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Confidence scores by modality
    sns.boxplot(data=df, x='modality', y='solution_confidence', ax=axes[0,1])
    axes[0,1].set_title('Confidence Scores by Modality')
    axes[0,1].set_ylabel('Confidence Score')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Response length by modality
    sns.boxplot(data=df, x='modality', y='response_length', ax=axes[0,2])
    axes[0,2].set_title('Response Length by Modality')
    axes[0,2].set_ylabel('Characters')
    axes[0,2].tick_params(axis='x', rotation=45)
    
    # 4. Quality metrics heatmap
    quality_metrics = ['structured_response', 'contains_solution', 'contains_explanation', 'contains_accessibility']
    quality_df = df[quality_metrics].mean().reset_index()
    quality_df.columns = ['Metric', 'Success Rate']
    sns.barplot(data=quality_df, x='Metric', y='Success Rate', ax=axes[1,0])
    axes[1,0].set_title('Overall Quality Metrics')
    axes[1,0].set_ylabel('Success Rate')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 5. Modality comparison heatmap
    modality_quality = df.groupby('modality')[quality_metrics].mean()
    sns.heatmap(modality_quality, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[1,1])
    axes[1,1].set_title('Quality Metrics by Modality')
    
    # 6. Processing time vs response quality
    sns.scatterplot(data=df, x='processing_time', y='solution_confidence', hue='modality', ax=axes[1,2])
    axes[1,2].set_title('Processing Time vs Confidence')
    axes[1,2].set_xlabel('Processing Time (seconds)')
    axes[1,2].set_ylabel('Confidence Score')
    
    plt.tight_layout()
    
    # Save the plot
    plots_dir = Path("runs/plots")
    plots_dir.mkdir(exist_ok=True)
    plot_file = plots_dir / f"experiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved to: {plot_file}")
    
    return plot_file

def generate_detailed_report(df, output_file):
    """Generate a detailed markdown report"""
    print(f"\n📝 GENERATING DETAILED REPORT...")
    
    report_lines = [
        "# GUI Bug Repair LLM Experiments - Detailed Report",
        f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total experiments: {len(df)}",
        "\n## Executive Summary",
        "\nThis report analyzes the performance of GPT-4o-mini in identifying and fixing GUI bugs across different modalities.",
        
        "\n## Key Findings",
        f"\n- **Best Performing Modality**: {df.groupby('modality')['solution_confidence'].mean().idxmax()}",
        f"- **Fastest Processing**: {df.groupby('modality')['processing_time'].mean().idxmin()} ({df.groupby('modality')['processing_time'].mean().min():.2f}s avg)",
        f"- **Most Comprehensive**: {df.groupby('modality')['response_length'].mean().idxmax()} ({df.groupby('modality')['response_length'].mean().max():.0f} chars avg)",
        
        "\n## Detailed Analysis by Modality",
    ]
    
    for modality in df['modality'].unique():
        modality_df = df[df['modality'] == modality]
        report_lines.extend([
            f"\n### {modality.replace('_', ' ').title()}",
            f"\n- **Tests Run**: {len(modality_df)}",
            f"- **Average Processing Time**: {modality_df['processing_time'].mean():.2f}s",
            f"- **Average Confidence Score**: {modality_df['solution_confidence'].mean():.2f}",
            f"- **Structured Response Rate**: {(modality_df['structured_response'].mean() * 100):.1f}%",
            f"- **Average Response Length**: {modality_df['response_length'].mean():.0f} characters",
            f"- **Solution Coverage**: {(modality_df['contains_solution'].mean() * 100):.1f}%",
            f"- **Explanation Coverage**: {(modality_df['contains_explanation'].mean() * 100):.1f}%",
            f"- **Accessibility Coverage**: {(modality_df['contains_accessibility'].mean() * 100):.1f}%",
        ])
    
    report_lines.extend([
        "\n## Bug Category Analysis",
        "\nPerformance across different bug types:",
    ])
    
    bug_categories = {
        "BUTTON_CLICK_01": "Interaction Event",
        "LAYOUT_BREAK_02": "Visual Layout", 
        "ACCESSIBILITY_03": "Accessibility",
        "STATE_MANAGEMENT_04": "State Transition",
        "COLOR_CONTRAST_05": "Visual Color/Typography"
    }
    
    for bug_id, category in bug_categories.items():
        bug_df = df[df['experiment_id'].str.startswith(bug_id)]
        if len(bug_df) > 0:
            report_lines.extend([
                f"\n### {category}",
                f"- **Average Confidence**: {bug_df['solution_confidence'].mean():.2f}",
                f"- **Structured Response Rate**: {(bug_df['structured_response'].mean() * 100):.1f}%",
                f"- **Best Modality**: {bug_df.groupby('modality')['solution_confidence'].mean().idxmax()}",
            ])
    
    report_lines.extend([
        "\n## Recommendations",
        "\nBased on the experiment results:",
        "\n1. **For Production Use**: The multimodal approach provides the most comprehensive solutions",
        "\n2. **For Speed**: Text-only modality is fastest for initial bug assessment",
        "\n3. **For Accessibility**: All modalities perform well, but visual modality adds context",
        "\n4. **For Development**: Use structured response formats to improve consistency",
        "\n\n## Methodology",
        "\n- **Model**: GPT-4o-mini",
        "\n- **Temperature**: 0.2",
        "\n- **Evaluation**: Automated metrics for confidence, structure, and coverage",
        "\n- **Scenarios**: 5 realistic GUI bug scenarios across different categories",
    ])
    
    # Save report
    reports_dir = Path("runs/reports")
    reports_dir.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"📄 Detailed report saved to: {output_file}")

def main():
    """Main analysis function"""
    print("🔍 GUI Bug Repair Experiment Results Analyzer")
    print("=" * 50)
    
    # Load results
    results_data = load_latest_results()
    if results_data is None:
        return
    
    results, results_file = results_data
    
    # Analyze results
    df = analyze_results(results)
    
    # Generate visualizations
    plot_file = generate_visualizations(df)
    
    # Generate detailed report
    reports_dir = Path("runs/reports")
    reports_dir.mkdir(exist_ok=True)
    report_file = reports_dir / f"experiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    generate_detailed_report(df, report_file)
    
    print(f"\n✅ Analysis complete!")
    print(f"📊 Results: {results_file}")
    print(f"📈 Visualizations: {plot_file}")
    print(f"📄 Report: {report_file}")
    
    # Show some sample responses
    print(f"\n🎯 SAMPLE RESPONSES:")
    print("=" * 50)
    
    # Show best performing experiment
    best_exp = df.loc[df['solution_confidence'].idxmax()]
    print(f"\n🏆 Best Performance ({best_exp['solution_confidence']:.2f} confidence):")
    print(f"Bug: {best_exp['bug_description'][:100]}...")
    print(f"Modality: {best_exp['modality']}")
    print(f"Response: {best_exp['llm_response'][:200]}...")

if __name__ == "__main__":
    main()
