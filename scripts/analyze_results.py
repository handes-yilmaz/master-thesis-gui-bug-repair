#!/usr/bin/env python
"""
Comprehensive Analysis of LLM Performance Results
================================================

This script performs detailed statistical analysis on the LLM experiment results
to support research writing and academic analysis.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class LLMResultsAnalyzer:
    def __init__(self, results_file: str):
        """Initialize the analyzer with results data"""
        self.results_file = results_file
        self.results_data = self._load_results()
        self.df = self._create_dataframe()
        
    def _load_results(self):
        """Load the experiment results"""
        with open(self.results_file, 'r') as f:
            return json.load(f)
    
    def _create_dataframe(self):
        """Convert results to pandas DataFrame for analysis"""
        # Extract individual experiment results
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
    
    def basic_statistics(self):
        """Generate basic statistical summary"""
        print("📊 BASIC STATISTICAL SUMMARY")
        print("=" * 50)
        
        # Overall statistics
        print(f"Total Experiments: {len(self.df)}")
        print(f"Unique Bugs Tested: {self.df['bug_id'].nunique()}")
        print(f"Models Tested: {', '.join(self.df['model_name'].unique())}")
        print()
        
        # Performance by model
        model_stats = self.df.groupby('model_name')['processing_time'].agg([
            'count', 'mean', 'std', 'min', 'max', 'median'
        ]).round(3)
        
        print("Performance by Model:")
        print(model_stats)
        print()
        
        # Overall performance
        overall_stats = self.df['processing_time'].describe()
        print("Overall Performance Statistics:")
        print(overall_stats.round(3))
        
        return model_stats, overall_stats
    
    def statistical_significance_test(self):
        """Perform statistical significance tests"""
        print("\n🔬 STATISTICAL SIGNIFICANCE TESTS")
        print("=" * 50)
        
        # Separate data by model
        openai_times = self.df[self.df['model_name'] == 'gpt']['processing_time'].dropna()
        claude_times = self.df[self.df['model_name'] == 'claude']['processing_time'].dropna()
        
        if len(openai_times) == 0 or len(claude_times) == 0:
            print("❌ Insufficient data for statistical testing")
            return None
        
        # Shapiro-Wilk test for normality
        print("1. Normality Test (Shapiro-Wilk):")
        openai_normality = stats.shapiro(openai_times)
        claude_normality = stats.shapiro(claude_times)
        
        print(f"   OpenAI: W={openai_normality.statistic:.4f}, p={openai_normality.pvalue:.4f}")
        print(f"   Claude: W={claude_normality.statistic:.4f}, p={claude_normality.pvalue:.4f}")
        
        # Levene's test for homogeneity of variance
        print("\n2. Homogeneity of Variance Test (Levene):")
        levene_result = stats.levene(openai_times, claude_times)
        print(f"   Statistic: {levene_result.statistic:.4f}")
        print(f"   p-value: {levene_result.pvalue:.4f}")
        
        # T-test (assuming normal distribution)
        print("\n3. Independent T-Test:")
        ttest_result = stats.ttest_ind(openai_times, claude_times)
        print(f"   t-statistic: {ttest_result.statistic:.4f}")
        print(f"   p-value: {ttest_result.pvalue:.4f}")
        print(f"   Degrees of freedom: {ttest_result.df:.1f}")
        
        # Mann-Whitney U test (non-parametric)
        print("\n4. Mann-Whitney U Test (Non-parametric):")
        mw_result = stats.mannwhitneyu(openai_times, claude_times, alternative='two-sided')
        print(f"   U-statistic: {mw_result.statistic:.4f}")
        print(f"   p-value: {mw_result.pvalue:.4f}")
        
        # Effect size (Cohen's d)
        print("\n5. Effect Size (Cohen's d):")
        pooled_std = np.sqrt(((len(openai_times) - 1) * openai_times.var() + 
                             (len(claude_times) - 1) * claude_times.var()) / 
                            (len(openai_times) + len(claude_times) - 2))
        cohens_d = (openai_times.mean() - claude_times.mean()) / pooled_std
        print(f"   Cohen's d: {cohens_d:.4f}")
        
        # Interpret effect size
        if abs(cohens_d) < 0.2:
            effect_size = "Small"
        elif abs(cohens_d) < 0.5:
            effect_size = "Small to Medium"
        elif abs(cohens_d) < 0.8:
            effect_size = "Medium"
        elif abs(cohens_d) < 1.2:
            effect_size = "Large"
        else:
            effect_size = "Very Large"
        
        print(f"   Effect Size Interpretation: {effect_size}")
        
        return {
            'ttest': ttest_result,
            'mannwhitney': mw_result,
            'cohens_d': cohens_d,
            'effect_size': effect_size
        }
    
    def performance_analysis(self):
        """Detailed performance analysis"""
        print("\n📈 DETAILED PERFORMANCE ANALYSIS")
        print("=" * 50)
        
        # Performance comparison
        openai_avg = self.df[self.df['model_name'] == 'gpt']['processing_time'].mean()
        claude_avg = self.df[self.df['model_name'] == 'claude']['processing_time'].mean()
        
        print(f"OpenAI Average Time: {openai_avg:.3f} seconds")
        print(f"Claude Average Time: {claude_avg:.3f} seconds")
        
        # Calculate differences
        absolute_diff = abs(openai_avg - claude_avg)
        relative_diff = (absolute_diff / max(openai_avg, claude_avg)) * 100
        
        print(f"Absolute Difference: {absolute_diff:.3f} seconds")
        print(f"Relative Difference: {relative_diff:.1f}%")
        
        # Performance ratio
        if openai_avg < claude_avg:
            faster_model = "OpenAI"
            slower_model = "Claude"
            ratio = claude_avg / openai_avg
        else:
            faster_model = "Claude"
            slower_model = "OpenAI"
            ratio = openai_avg / claude_avg
        
        print(f"{faster_model} is {ratio:.2f}x faster than {slower_model}")
        
        # Consistency analysis
        openai_std = self.df[self.df['model_name'] == 'gpt']['processing_time'].std()
        claude_std = self.df[self.df['model_name'] == 'claude']['processing_time'].std()
        
        print(f"\nConsistency Analysis:")
        print(f"OpenAI Standard Deviation: {openai_std:.3f} seconds")
        print(f"Claude Standard Deviation: {claude_std:.3f} seconds")
        
        openai_cv = (openai_std / openai_avg) * 100
        claude_cv = (claude_std / claude_avg) * 100
        
        print(f"OpenAI Coefficient of Variation: {openai_cv:.1f}%")
        print(f"Claude Coefficient of Variation: {claude_cv:.1f}%")
        
        more_consistent = "OpenAI" if openai_cv < claude_cv else "Claude"
        print(f"More Consistent Model: {more_consistent}")
        
        return {
            'absolute_difference': absolute_diff,
            'relative_difference': relative_diff,
            'performance_ratio': ratio,
            'faster_model': faster_model,
            'consistency_analysis': {
                'openai_cv': openai_cv,
                'claude_cv': claude_cv,
                'more_consistent': more_consistent
            }
        }
    
    def bug_type_analysis(self):
        """Analyze performance by bug type and repository"""
        print("\n🐛 BUG TYPE AND REPOSITORY ANALYSIS")
        print("=" * 50)
        
        # Performance by repository
        repo_performance = self.df.groupby(['repo', 'model_name'])['processing_time'].agg([
            'count', 'mean', 'std'
        ]).round(3)
        
        print("Performance by Repository and Model:")
        print(repo_performance)
        print()
        
        # Bug difficulty analysis (if available)
        if 'difficulty' in self.df.columns:
            difficulty_performance = self.df.groupby(['difficulty', 'model_name'])['processing_time'].agg([
                'count', 'mean', 'std'
            ]).round(3)
            
            print("Performance by Bug Difficulty:")
            print(difficulty_performance)
            print()
        
        return repo_performance
    
    def generate_visualizations(self, output_dir: str = "runs/analysis"):
        """Generate comprehensive visualizations"""
        print("\n📊 GENERATING VISUALIZATIONS")
        print("=" * 50)
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Performance Comparison Box Plot
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 3, 1)
        self.df.boxplot(column='processing_time', by='model_name', ax=plt.gca())
        plt.title('Processing Time Comparison')
        plt.ylabel('Time (seconds)')
        plt.suptitle('')
        
        # 2. Performance Distribution
        plt.subplot(2, 3, 2)
        for model in self.df['model_name'].unique():
            model_data = self.df[self.df['model_name'] == model]['processing_time']
            plt.hist(model_data, alpha=0.7, label=model, bins=15)
        plt.title('Processing Time Distribution')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency')
        plt.legend()
        
        # 3. Performance by Repository
        plt.subplot(2, 3, 3)
        repo_perf = self.df.groupby(['repo', 'model_name'])['processing_time'].mean().unstack()
        repo_perf.plot(kind='bar', ax=plt.gca())
        plt.title('Performance by Repository')
        plt.xlabel('Repository')
        plt.ylabel('Average Time (seconds)')
        plt.xticks(rotation=45)
        plt.legend(title='Model')
        
        # 4. Time Series Analysis
        plt.subplot(2, 3, 4)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.sort_values('timestamp')
        
        for model in self.df['model_name'].unique():
            model_data = self.df[self.df['model_name'] == model]
            plt.plot(range(len(model_data)), model_data['processing_time'], 
                    marker='o', label=model, alpha=0.7)
        plt.title('Processing Time Over Experiments')
        plt.xlabel('Experiment Number')
        plt.ylabel('Time (seconds)')
        plt.legend()
        
        # 5. Statistical Summary
        plt.subplot(2, 3, 5)
        model_stats = self.df.groupby('model_name')['processing_time'].agg(['mean', 'std'])
        x_pos = np.arange(len(model_stats))
        plt.bar(x_pos, model_stats['mean'], yerr=model_stats['std'], 
               capsize=5, alpha=0.7)
        plt.title('Mean Processing Time with Standard Deviation')
        plt.xlabel('Model')
        plt.ylabel('Time (seconds)')
        plt.xticks(x_pos, model_stats.index)
        
        # 6. Performance Ratio
        plt.subplot(2, 3, 6)
        openai_avg = self.df[self.df['model_name'] == 'gpt']['processing_time'].mean()
        claude_avg = self.df[self.df['model_name'] == 'claude']['processing_time'].mean()
        
        models = ['OpenAI', 'Claude']
        times = [openai_avg, claude_avg]
        colors = ['#10a37f' if x == 'OpenAI' else '#d97706' for x in models]
        
        bars = plt.bar(models, times, color=colors, alpha=0.7)
        plt.title('Average Processing Time Comparison')
        plt.ylabel('Time (seconds)')
        
        # Add value labels on bars
        for bar, time in zip(bars, times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{time:.2f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = f"{output_dir}/performance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved to: {plot_path}")
        
        plt.show()
        
        return plot_path
    
    def generate_research_data(self, output_dir: str = "runs/analysis"):
        """Generate research-ready data tables and summaries"""
        print("\n📝 GENERATING RESEARCH DATA")
        print("=" * 50)
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 1. Performance Summary Table
        performance_table = self.df.groupby('model_name')['processing_time'].agg([
            'count', 'mean', 'std', 'min', 'max', 'median'
        ]).round(3)
        
        # Add relative performance
        fastest_time = performance_table['mean'].min()
        performance_table['relative_performance'] = (performance_table['mean'] / fastest_time).round(2)
        
        print("Performance Summary Table for Research:")
        print(performance_table)
        print()
        
        # 2. Statistical Test Results
        stats_results = self.statistical_significance_test()
        
        # 3. Save research data
        research_data = {
            'generated_at': datetime.now().isoformat(),
            'performance_summary': performance_table.to_dict(),
            'statistical_tests': {
                't_test_p_value': float(stats_results['ttest'].pvalue),
                'mann_whitney_p_value': float(stats_results['mannwhitney'].pvalue),
                'cohens_d': float(stats_results['cohens_d']),
                'effect_size': stats_results['effect_size']
            },
            'performance_analysis': self.performance_analysis(),
            'bug_analysis': str(self.bug_type_analysis())
        }
        
        # Save research data
        research_file = f"{output_dir}/research_analysis_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(research_file, 'w') as f:
            json.dump(research_data, f, indent=2)
        
        print(f"📝 Research data saved to: {research_file}")
        
        # 4. Generate LaTeX table
        latex_table = self._generate_latex_table(performance_table)
        latex_file = f"{output_dir}/performance_table_latex_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex"
        with open(latex_file, 'w') as f:
            f.write(latex_table)
        
        print(f"📝 LaTeX table saved to: {latex_file}")
        
        return research_file, latex_file
    
    def _generate_latex_table(self, performance_table):
        """Generate LaTeX table for research"""
        latex = r"""\begin{table}[h]
\centering
\caption{LLM Performance Comparison Results}
\label{tab:llm-performance}
\begin{tabular}{lcccccc}
\toprule
\textbf{Model} & \textbf{Count} & \textbf{Mean (s)} & \textbf{Std (s)} & \textbf{Min (s)} & \textbf{Max (s)} & \textbf{Relative} \\
\midrule
"""
        
        for model, row in performance_table.iterrows():
            latex += f"{model} & {row['count']} & {row['mean']:.3f} & {row['std']:.3f} & {row['min']:.3f} & {row['max']:.3f} & {row['relative_performance']:.2f}x \\\\\n"
        
        latex += r"""\bottomrule
\end{tabular}
\end{table}"""
        
        return latex
    
    def run_complete_analysis(self, output_dir: str = "runs/analysis"):
        """Run the complete analysis pipeline"""
        print("🚀 RUNNING COMPLETE LLM RESULTS ANALYSIS")
        print("=" * 60)
        print(f"📁 Results file: {self.results_file}")
        print(f"📊 Total experiments: {len(self.df)}")
        print(f"🤖 Models tested: {', '.join(self.df['model_name'].unique())}")
        print("=" * 60)
        
        # Run all analyses
        basic_stats = self.basic_statistics()
        significance_tests = self.statistical_significance_test()
        performance_analysis = self.performance_analysis()
        bug_analysis = self.bug_type_analysis()
        
        # Generate visualizations and research data
        plot_path = self.generate_visualizations(output_dir)
        research_file, latex_file = self.generate_research_data(output_dir)
        
        print("\n🎉 ANALYSIS COMPLETE!")
        print("=" * 60)
        print(f"📊 Visualizations: {plot_path}")
        print(f"📝 Research data: {research_file}")
        print(f"📋 LaTeX table: {latex_file}")
        
        return {
            'basic_stats': basic_stats,
            'significance_tests': significance_tests,
            'performance_analysis': performance_analysis,
            'bug_analysis': bug_analysis,
            'plot_path': plot_path,
            'research_file': research_file,
            'latex_file': latex_file
        }

def main():
    """Main function to run the analysis"""
    # Find the most recent results file
    results_dir = Path("data/results")
    results_files = list(results_dir.glob("cloud_processing_results_*.json"))
    
    if not results_files:
        print("❌ No results files found in data/results/")
        return
    
    # Use the most recent file
    latest_file = max(results_files, key=lambda x: x.stat().st_mtime)
    print(f"📁 Using results file: {latest_file}")
    
    # Run analysis
    analyzer = LLMResultsAnalyzer(str(latest_file))
    results = analyzer.run_complete_analysis()
    
    print("\n✅ Analysis complete! Check the runs/analysis/ directory for outputs.")

if __name__ == "__main__":
    main()
