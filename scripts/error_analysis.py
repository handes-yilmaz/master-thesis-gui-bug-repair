#!/usr/bin/env python
"""
Error Analysis for LLM Bug Repair Performance
============================================

This script analyzes when and why LLMs fail in GUI bug repair tasks,
providing insights for methodology improvement.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import re

class LLMErrorAnalyzer:
    def __init__(self, results_file: str):
        """Initialize the error analyzer with results data"""
        self.results_file = results_file
        self.results_data = self._load_results()
        self.df = self._create_dataframe()
        
    def _load_results(self):
        """Load the experiment results"""
        with open(self.results_file, 'r') as f:
            return json.load(f)
    
    def _create_dataframe(self):
        """Convert results to pandas DataFrame for analysis"""
        experiments = []
        for result in self.results_data:
            if isinstance(result, dict) and 'bug_id' in result:
                experiments.append(result)
        
        df = pd.DataFrame(experiments)
        
        # Convert processing time to numeric
        df['processing_time'] = pd.to_numeric(df['processing_time'], errors='coerce')
        
        # Extract model name from model field
        df['model_name'] = df['model'].apply(lambda x: x.split('-')[0] if isinstance(x, str) else x)
        
        return df
    
    def analyze_response_quality(self):
        """Analyze the quality and structure of LLM responses"""
        print("\n" + "🔍 RESPONSE QUALITY ANALYSIS".center(80, "="))
        print()
        
        # Response length analysis
        df_with_responses = self.df[self.df['response'].notna()].copy()
        df_with_responses['response_length'] = df_with_responses['response'].str.len()
        df_with_responses['word_count'] = df_with_responses['response'].str.split().str.len()
        
        print(f"📊 Total Responses with Content: {len(df_with_responses)}")
        print(f"❌ Missing Responses: {len(self.df) - len(df_with_responses)}")
        print()
        
        # Response length statistics by model
        length_stats = df_with_responses.groupby('model_name')['response_length'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(0)
        
        print("📏 RESPONSE LENGTH ANALYSIS (Characters)")
        print("-" * 60)
        print(f"{'Model':<15} {'Count':<8} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
        print("-" * 60)
        
        for model, row in length_stats.iterrows():
            model_name = model.upper()
            print(f"{model_name:<15} {row['count']:<8.0f} {row['mean']:<8.0f} {row['std']:<8.0f} {row['min']:<8.0f} {row['max']:<8.0f}")
        
        print()
        
        # Word count statistics by model
        word_stats = df_with_responses.groupby('model_name')['word_count'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(0)
        
        print("📝 WORD COUNT ANALYSIS")
        print("-" * 60)
        print(f"{'Model':<15} {'Count':<8} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
        print("-" * 60)
        
        for model, row in word_stats.iterrows():
            model_name = model.upper()
            print(f"{model_name:<15} {row['count']:<8.0f} {row['mean']:<8.0f} {row['std']:<8.0f} {row['min']:<8.0f} {row['max']:<8.0f}")
        
        print()
        
        return df_with_responses, length_stats, word_stats
    
    def analyze_response_structure(self):
        """Analyze the structure and format of responses"""
        print("\n" + "📋 RESPONSE STRUCTURE ANALYSIS".center(80, "="))
        print()
        
        df_with_responses = self.df[self.df['response'].notna()].copy()
        
        # Check for structured sections
        structure_analysis = {}
        
        # Common section headers to look for
        section_headers = [
            'root cause', 'solution', 'explanation', 'testing', 'code', 'fix',
            'problem', 'issue', 'cause', 'resolve', 'implement', 'steps'
        ]
        
        for model in df_with_responses['model_name'].unique():
            model_responses = df_with_responses[df_with_responses['model_name'] == model]
            
            structured_count = 0
            total_count = len(model_responses)
            
            for response in model_responses['response']:
                response_lower = response.lower()
                
                # Check if response has multiple sections
                section_count = sum(1 for header in section_headers if header in response_lower)
                
                # Check for markdown formatting
                has_markdown = bool(re.search(r'[#*`]', response))
                
                # Check for code blocks
                has_code_blocks = bool(re.search(r'```[\s\S]*```', response))
                
                # Consider structured if it has multiple sections or good formatting
                if section_count >= 2 or has_markdown or has_code_blocks:
                    structured_count += 1
            
            structure_analysis[model] = {
                'total': total_count,
                'structured': structured_count,
                'unstructured': total_count - structured_count,
                'structure_rate': (structured_count / total_count) * 100
            }
        
        print("🏗️  STRUCTURE ANALYSIS RESULTS")
        print("-" * 60)
        print(f"{'Model':<15} {'Total':<8} {'Structured':<12} {'Unstructured':<12} {'Rate':<8}")
        print("-" * 60)
        
        for model, stats in structure_analysis.items():
            model_name = model.upper()
            print(f"{model_name:<15} {stats['total']:<8} {stats['structured']:<12} {stats['unstructured']:<12} {stats['structure_rate']:<8.1f}%")
        
        print()
        
        # Summary
        total_structured = sum(stats['structured'] for stats in structure_analysis.values())
        total_responses = sum(stats['total'] for stats in structure_analysis.values())
        overall_rate = (total_structured / total_responses) * 100
        
        print(f"📊 OVERALL STRUCTURE RATE: {overall_rate:.1f}% ({total_structured}/{total_responses})")
        print()
        
        return structure_analysis
    
    def analyze_bug_complexity_impact(self):
        """Analyze how bug complexity affects LLM performance"""
        print("\n" + "🐛 BUG COMPLEXITY IMPACT ANALYSIS".center(80, "="))
        print()
        
        # Analyze performance by repository (proxy for complexity)
        repo_analysis = self.df.groupby(['repo', 'model_name'])['processing_time'].agg([
            'count', 'mean', 'std'
        ]).round(3)
        
        print("📊 PERFORMANCE BY REPOSITORY (Complexity Proxy)")
        print("-" * 80)
        print(f"{'Repository':<30} {'Model':<15} {'Count':<8} {'Mean (s)':<10} {'Std (s)':<10}")
        print("-" * 80)
        
        for (repo, model), row in repo_analysis.iterrows():
            print(f"{repo:<30} {model.upper():<15} {row['count']:<8} {row['mean']:<10.3f} {row['std']:<10.3f}")
        
        print()
        
        # Calculate complexity impact
        complexity_impact = {}
        
        for repo in self.df['repo'].unique():
            repo_data = self.df[self.df['repo'] == repo]
            
            if len(repo_data) >= 2:  # Need at least 2 models for comparison
                openai_time = repo_data[repo_data['model_name'] == 'gpt']['processing_time'].mean()
                claude_time = repo_data[repo_data['model_name'] == 'claude']['processing_time'].mean()
                
                if not pd.isna(openai_time) and not pd.isna(claude_time):
                    performance_gap = abs(openai_time - claude_time)
                    relative_gap = (performance_gap / max(openai_time, claude_time)) * 100
                    
                    complexity_impact[repo] = {
                        'openai_time': openai_time,
                        'claude_time': claude_time,
                        'absolute_gap': performance_gap,
                        'relative_gap': relative_gap,
                        'complexity_level': 'High' if relative_gap > 20 else 'Medium' if relative_gap > 10 else 'Low'
                    }
        
        print("🎯 COMPLEXITY IMPACT ANALYSIS")
        print("-" * 80)
        print(f"{'Repository':<30} {'OpenAI (s)':<12} {'Claude (s)':<12} {'Gap (s)':<10} {'Gap (%)':<10} {'Level':<8}")
        print("-" * 80)
        
        for repo, impact in complexity_impact.items():
            repo_short = repo.split('/')[-1] if '/' in repo else repo
            print(f"{repo_short:<30} {impact['openai_time']:<12.2f} {impact['claude_time']:<12.2f} {impact['absolute_gap']:<10.2f} {impact['relative_gap']:<10.1f} {impact['complexity_level']:<8}")
        
        print()
        
        # Summary
        high_complexity = sum(1 for impact in complexity_impact.values() if impact['complexity_level'] == 'High')
        medium_complexity = sum(1 for impact in complexity_impact.values() if impact['complexity_level'] == 'Medium')
        low_complexity = sum(1 for impact in complexity_impact.values() if impact['complexity_level'] == 'Low')
        
        print(f"📊 COMPLEXITY DISTRIBUTION:")
        print(f"   🔴 High Complexity: {high_complexity} repositories")
        print(f"   🟡 Medium Complexity: {medium_complexity} repositories")
        print(f"   🟢 Low Complexity: {low_complexity} repositories")
        print()
        
        return complexity_impact
    
    def analyze_failure_patterns(self):
        """Analyze patterns in LLM failures or poor performance"""
        print("\n" + "❌ FAILURE PATTERN ANALYSIS".center(80, "="))
        print()
        
        # Identify potential failure indicators
        df_with_responses = self.df[self.df['response'].notna()].copy()
        
        # Analyze response quality indicators
        failure_indicators = {}
        
        for model in df_with_responses['model_name'].unique():
            model_responses = df_with_responses[df_with_responses['model_name'] == model]
            
            # Check for short responses (potential incomplete answers)
            short_responses = model_responses[model_responses['response'].str.len() < 500]
            
            # Check for generic responses
            generic_indicators = [
                'i cannot', 'i am unable', 'i don\'t have', 'i cannot provide',
                'this is beyond', 'i cannot determine', 'i need more information'
            ]
            
            generic_count = 0
            for response in model_responses['response']:
                if any(indicator in response.lower() for indicator in generic_indicators):
                    generic_count += 1
            
            # Check for response time outliers
            response_times = model_responses['processing_time']
            outliers = response_times[
                (response_times < response_times.quantile(0.25) - 1.5 * (response_times.quantile(0.75) - response_times.quantile(0.25))) |
                (response_times > response_times.quantile(0.75) + 1.5 * (response_times.quantile(0.75) - response_times.quantile(0.25)))
            ]
            
            failure_indicators[model] = {
                'total_responses': len(model_responses),
                'short_responses': len(short_responses),
                'short_response_rate': (len(short_responses) / len(model_responses)) * 100,
                'generic_responses': generic_count,
                'generic_response_rate': (generic_count / len(model_responses)) * 100,
                'time_outliers': len(outliers),
                'outlier_rate': (len(outliers) / len(model_responses)) * 100
            }
        
        print("🚨 FAILURE PATTERN ANALYSIS RESULTS")
        print("-" * 80)
        print(f"{'Model':<15} {'Total':<8} {'Short (%)':<12} {'Generic (%)':<12} {'Outliers (%)':<12}")
        print("-" * 80)
        
        for model, indicators in failure_indicators.items():
            model_name = model.upper()
            print(f"{model_name:<15} {indicators['total_responses']:<8} {indicators['short_response_rate']:<12.1f} {indicators['generic_response_rate']:<12.1f} {indicators['outlier_rate']:<12.1f}")
        
        print()
        
        # Summary
        total_short = sum(indicators['short_responses'] for indicators in failure_indicators.values())
        total_generic = sum(indicators['generic_responses'] for indicators in failure_indicators.values())
        total_outliers = sum(indicators['time_outliers'] for indicators in failure_indicators.values())
        total_responses = sum(indicators['total_responses'] for indicators in failure_indicators.values())
        
        print(f"📊 OVERALL FAILURE SUMMARY:")
        print(f"   📉 Short Responses: {total_short}/{total_responses} ({total_short/total_responses*100:.1f}%)")
        print(f"   🤖 Generic Responses: {total_generic}/{total_responses} ({total_generic/total_responses*100:.1f}%)")
        print(f"   ⏱️  Time Outliers: {total_outliers}/{total_responses} ({total_outliers/total_responses*100:.1f}%)")
        print()
        
        return failure_indicators
    
    def generate_error_visualizations(self, output_dir: str = "runs/error_analysis"):
        """Generate visualizations for error analysis"""
        print("\n📊 GENERATING ERROR ANALYSIS VISUALIZATIONS")
        print("=" * 50)
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Response Length Comparison
        df_with_responses = self.df[self.df['response'].notna()].copy()
        df_with_responses['response_length'] = df_with_responses['response'].str.len()
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        df_with_responses.boxplot(column='response_length', by='model_name', ax=plt.gca())
        plt.title('Response Length Comparison')
        plt.ylabel('Length (characters)')
        plt.suptitle('')
        
        # 2. Response Length Distribution
        plt.subplot(2, 3, 2)
        for model in df_with_responses['model_name'].unique():
            model_data = df_with_responses[df_with_responses['model_name'] == model]['response_length']
            plt.hist(model_data, alpha=0.7, label=model, bins=15)
        plt.title('Response Length Distribution')
        plt.xlabel('Length (characters)')
        plt.ylabel('Frequency')
        plt.legend()
        
        # 3. Performance vs Response Length
        plt.subplot(2, 3, 3)
        for model in df_with_responses['model_name'].unique():
            model_data = df_with_responses[df_with_responses['model_name'] == model]
            plt.scatter(model_data['response_length'], model_data['processing_time'], 
                       alpha=0.7, label=model, s=50)
        plt.title('Performance vs Response Length')
        plt.xlabel('Response Length (characters)')
        plt.ylabel('Processing Time (seconds)')
        plt.legend()
        
        # 4. Repository Performance Comparison
        plt.subplot(2, 3, 4)
        repo_perf = self.df.groupby(['repo', 'model_name'])['processing_time'].mean().unstack()
        repo_perf.plot(kind='bar', ax=plt.gca())
        plt.title('Performance by Repository')
        plt.xlabel('Repository')
        plt.ylabel('Average Time (seconds)')
        plt.xticks(rotation=45)
        plt.legend(title='Model')
        
        # 5. Failure Indicators
        plt.subplot(2, 3, 5)
        failure_indicators = self.analyze_failure_patterns()
        
        models = list(failure_indicators.keys())
        short_rates = [failure_indicators[model]['short_response_rate'] for model in models]
        generic_rates = [failure_indicators[model]['generic_response_rate'] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        plt.bar(x - width/2, short_rates, width, label='Short Responses', alpha=0.7)
        plt.bar(x + width/2, generic_rates, width, label='Generic Responses', alpha=0.7)
        plt.title('Failure Indicators by Model')
        plt.xlabel('Model')
        plt.ylabel('Rate (%)')
        plt.xticks(x, [model.upper() for model in models])
        plt.legend()
        
        # 6. Complexity Impact
        plt.subplot(2, 3, 6)
        complexity_impact = self.analyze_bug_complexity_impact()
        
        if complexity_impact:
            repos = list(complexity_impact.keys())
            relative_gaps = [complexity_impact[repo]['relative_gap'] for repo in repos]
            
            colors = ['red' if gap > 20 else 'orange' if gap > 10 else 'green' for gap in relative_gaps]
            bars = plt.bar(repos, relative_gaps, color=colors, alpha=0.7)
            plt.title('Performance Gap by Repository (Complexity)')
            plt.xlabel('Repository')
            plt.ylabel('Relative Performance Gap (%)')
            plt.xticks(rotation=45)
            
            # Add value labels
            for bar, gap in zip(bars, relative_gaps):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        f'{gap:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = f"{output_dir}/error_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Error analysis visualization saved to: {plot_path}")
        
        plt.show()
        
        return plot_path
    
    def generate_error_report(self, output_dir: str = "runs/error_analysis"):
        """Generate comprehensive error analysis report"""
        print("\n📝 GENERATING ERROR ANALYSIS REPORT")
        print("=" * 50)
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Run all analyses
        response_quality = self.analyze_response_quality()
        structure_analysis = self.analyze_response_structure()
        complexity_impact = self.analyze_bug_complexity_impact()
        failure_patterns = self.analyze_failure_patterns()
        
        # Compile report
        error_report = {
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_experiments': len(self.df),
                'models_tested': list(self.df['model_name'].unique()),
                'repositories_tested': list(self.df['repo'].unique())
            },
            'response_quality': {
                'response_length_stats': response_quality[1].to_dict() if hasattr(response_quality[1], 'to_dict') else str(response_quality[1]),
                'word_count_stats': response_quality[2].to_dict() if hasattr(response_quality[2], 'to_dict') else str(response_quality[2])
            },
            'structure_analysis': structure_analysis,
            'complexity_impact': complexity_impact,
            'failure_patterns': failure_patterns,
            'recommendations': self._generate_recommendations()
        }
        
        # Save report
        report_file = f"{output_dir}/error_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(error_report, f, indent=2)
        
        print(f"📝 Error analysis report saved to: {report_file}")
        
        return report_file
    
    def _generate_recommendations(self):
        """Generate recommendations based on error analysis"""
        recommendations = {
            'methodology_improvements': [
                "Implement response quality scoring based on length and structure",
                "Add complexity classification for bugs before LLM processing",
                "Set minimum response length thresholds for quality assurance",
                "Implement retry mechanisms for failed or poor-quality responses"
            ],
            'prompt_optimization': [
                "Add explicit structure requirements to prompts",
                "Include complexity indicators in bug descriptions",
                "Request specific output formats (markdown, code blocks)",
                "Add validation criteria to prompts"
            ],
            'quality_metrics': [
                "Response completeness score",
                "Structural adherence score",
                "Technical accuracy validation",
                "Solution implementability assessment"
            ]
        }
        
        return recommendations
    
    def run_complete_error_analysis(self, output_dir: str = "runs/error_analysis"):
        """Run the complete error analysis pipeline"""
        print("\n" + "🚀 RUNNING COMPLETE ERROR ANALYSIS".center(80, "="))
        print()
        print(f"📁 Results file: {self.results_file}")
        print(f"📊 Total experiments: {len(self.df)}")
        print(f"🤖 Models tested: {', '.join(self.df['model_name'].unique())}")
        print(f"🐛 Repositories tested: {', '.join(self.df['repo'].unique())}")
        print()
        print("=" * 80)
        
        # Run all analyses
        response_quality = self.analyze_response_quality()
        structure_analysis = self.analyze_response_structure()
        complexity_impact = self.analyze_bug_complexity_impact()
        failure_patterns = self.analyze_failure_patterns()
        
        # Generate visualizations and report
        plot_path = self.generate_error_visualizations(output_dir)
        report_file = self.generate_error_report(output_dir)
        
        print("\n" + "🎉 ERROR ANALYSIS COMPLETE!".center(80, "="))
        print()
        
        # Generate summary
        print("📋 EXECUTIVE SUMMARY")
        print("-" * 80)
        
        # Response quality summary
        total_responses = len(self.df[self.df['response'].notna()])
        avg_length = self.df[self.df['response'].notna()]['response'].str.len().mean()
        print(f"📊 Total Valid Responses: {total_responses}")
        print(f"📏 Average Response Length: {avg_length:.0f} characters")
        
        # Structure summary
        total_structured = sum(stats['structured'] for stats in structure_analysis.values())
        structure_rate = (total_structured / total_responses) * 100
        print(f"🏗️  Structured Response Rate: {structure_rate:.1f}%")
        
        # Complexity summary
        if complexity_impact:
            high_complex = sum(1 for impact in complexity_impact.values() if impact['complexity_level'] == 'High')
            print(f"🔴 High Complexity Repositories: {high_complex}")
        
        # Failure summary
        total_failures = sum(indicators['short_responses'] + indicators['generic_responses'] for indicators in failure_patterns.values())
        failure_rate = (total_failures / total_responses) * 100
        print(f"❌ Overall Failure Rate: {failure_rate:.1f}%")
        
        print("-" * 80)
        print(f"📊 Visualizations: {plot_path}")
        print(f"📝 Error report: {report_file}")
        
        return {
            'response_quality': response_quality,
            'structure_analysis': structure_analysis,
            'complexity_impact': complexity_impact,
            'failure_patterns': failure_patterns,
            'plot_path': plot_path,
            'report_file': report_file
        }

def main():
    """Main function to run the error analysis"""
    # Find the most recent results file
    results_dir = Path("data/results")
    results_files = list(results_dir.glob("cloud_processing_results_*.json"))
    
    if not results_files:
        print("❌ No results files found in data/results/")
        return
    
    # Use the most recent file
    latest_file = max(results_files, key=lambda x: x.stat().st_mtime)
    print(f"📁 Using results file: {latest_file}")
    
    # Run error analysis
    analyzer = LLMErrorAnalyzer(str(latest_file))
    results = analyzer.run_complete_error_analysis()
    
    print("\n✅ Error analysis complete! Check the runs/error_analysis/ directory for outputs.")

if __name__ == "__main__":
    main()
