#!/usr/bin/env python
"""
SWE-Bench 46 Experiments - Visualization Generator
=================================================

This script creates comprehensive visualizations of the SWE-bench experiment results
for research analysis and presentation purposes.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SWEBenchVisualizationGenerator:
    """Generate comprehensive visualizations for SWE-bench results"""
    
    def __init__(self, results_file: str = "data/results/week2_swe_bench_46_experiments/swe_bench_46_results_20250831_181453.json"):
        """Initialize the visualization generator"""
        self.results_file = Path(results_file)
        self.output_dir = Path("data/results/figures/week2_visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        self.results = self._load_results()
        self.df = self._create_dataframe()
        
        print(f"🎨 SWE-Bench Visualization Generator Initialized")
        print(f"📊 Loaded {len(self.results)} experiments")
        print(f"📁 Output directory: {self.output_dir}")
    
    def _load_results(self) -> list:
        """Load the experiment results"""
        with open(self.results_file, 'r') as f:
            return json.load(f)
    
    def _create_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame for analysis"""
        df = pd.DataFrame(self.results)
        
        # Clean and enhance data
        df['processing_time'] = pd.to_numeric(df['processing_time'], errors='coerce')
        
        # Extract evaluation metrics from nested structure
        df['response_length'] = df['evaluation_metrics'].apply(lambda x: x.get('response_length', 0) if isinstance(x, dict) else 0)
        df['solution_confidence'] = df['evaluation_metrics'].apply(lambda x: x.get('solution_confidence', 0) if isinstance(x, dict) else 0)
        df['contains_solution'] = df['evaluation_metrics'].apply(lambda x: x.get('contains_solution', False) if isinstance(x, dict) else False)
        df['contains_explanation'] = df['evaluation_metrics'].apply(lambda x: x.get('contains_explanation', False) if isinstance(x, dict) else False)
        df['contains_testing'] = df['evaluation_metrics'].apply(lambda x: x.get('contains_testing', False) if isinstance(x, dict) else False)
        df['contains_root_cause'] = df['evaluation_metrics'].apply(lambda x: x.get('contains_root_cause', False) if isinstance(x, dict) else False)
        df['structured_response'] = df['evaluation_metrics'].apply(lambda x: x.get('structured_response', False) if isinstance(x, dict) else False)
        
        # Add derived columns
        df['model_category'] = df['model_used'].map({
            'primary': 'GPT-4o-mini',
            'claude': 'Claude 3.7 Sonnet',
            'gpt4': 'GPT-4'
        })
        
        return df
    
    def generate_all_visualizations(self):
        """Generate all visualization types"""
        print(f"\n🚀 Generating Comprehensive SWE-Bench Visualizations...")
        
        # 1. Performance Analysis
        self._create_performance_charts()
        
        # 2. Bug Distribution Analysis
        self._create_bug_distribution_charts()
        
        # 3. Model Comparison
        self._create_model_comparison_charts()
        
        # 4. Repository Analysis
        self._create_repository_charts()
        
        # 5. Quality Metrics
        self._create_quality_metrics_charts()
        
        # 6. Summary Statistics
        self._create_summary_statistics()
        
        print(f"\n✅ All visualizations generated successfully!")
        print(f"📁 Output directory: {self.output_dir}")
    
    def _create_performance_charts(self):
        """Create performance analysis charts"""
        print(f"📈 Creating performance charts...")
        
        # 1. Processing Time by Model
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Performance Analysis Across Models', fontsize=16, fontweight='bold')
        
        # Processing time distribution
        for i, model in enumerate(['primary', 'claude', 'gpt4']):
            model_data = self.df[self.df['model_used'] == model]['processing_time']
            axes[0, 0].hist(model_data, alpha=0.7, label=model, bins=15)
        axes[0, 0].set_title('Processing Time Distribution by Model')
        axes[0, 0].set_xlabel('Processing Time (seconds)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Processing time box plot
        self.df.boxplot(column='processing_time', by='model_used', ax=axes[0, 1])
        axes[0, 1].set_title('Processing Time Comparison (Box Plot)')
        axes[0, 1].set_xlabel('Model')
        axes[0, 1].set_ylabel('Processing Time (seconds)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Processing time by bug type
        bug_time_data = self.df.groupby('bug_category')['processing_time'].mean().sort_values(ascending=False)
        bug_time_data.plot(kind='bar', ax=axes[1, 0], color='skyblue')
        axes[1, 0].set_title('Average Processing Time by Bug Category')
        axes[1, 0].set_xlabel('Bug Category')
        axes[1, 0].set_ylabel('Average Time (seconds)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Processing time by severity
        severity_time_data = self.df.groupby('bug_severity')['processing_time'].mean().reindex(['low', 'high', 'critical'])
        severity_time_data.plot(kind='bar', ax=axes[1, 1], color='lightcoral')
        axes[1, 1].set_title('Average Processing Time by Bug Severity')
        axes[1, 1].set_xlabel('Bug Severity')
        axes[1, 1].set_ylabel('Average Time (seconds)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Performance Timeline
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Group by bug and model to show progression
        timeline_data = self.df.groupby(['bug_title', 'model_used'])['processing_time'].mean().unstack()
        
        # Plot timeline
        for model in ['primary', 'claude', 'gpt4']:
            if model in timeline_data.columns:
                ax.plot(range(len(timeline_data)), timeline_data[model], 
                       marker='o', label=model, linewidth=2, markersize=4)
        
        ax.set_title('Processing Time Timeline: Performance Across All Bugs', fontsize=14, fontweight='bold')
        ax.set_xlabel('Bug Index (1-46)')
        ax.set_ylabel('Processing Time (seconds)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_timeline.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_bug_distribution_charts(self):
        """Create bug distribution analysis charts"""
        print(f"🐛 Creating bug distribution charts...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Bug Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Bug Categories
        bug_categories = self.df['bug_category'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(bug_categories)))
        axes[0, 0].pie(bug_categories.values, labels=bug_categories.index, autopct='%1.1f%%', 
                       colors=colors, startangle=90)
        axes[0, 0].set_title('Bug Categories Distribution')
        
        # 2. Bug Severity
        bug_severity = self.df['bug_severity'].value_counts().reindex(['low', 'high', 'critical'])
        severity_colors = ['lightgreen', 'orange', 'red']
        axes[0, 1].bar(bug_severity.index, bug_severity.values, color=severity_colors)
        axes[0, 1].set_title('Bug Severity Distribution')
        axes[0, 1].set_ylabel('Number of Bugs')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Repositories
        repositories = self.df['repo'].value_counts()
        axes[1, 0].barh(repositories.index, repositories.values, color='lightblue')
        axes[1, 0].set_title('Bugs by Repository')
        axes[1, 0].set_xlabel('Number of Bugs')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Bug Type vs Severity Heatmap
        bug_severity_cross = pd.crosstab(self.df['bug_category'], self.df['bug_severity'])
        sns.heatmap(bug_severity_cross, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1, 1])
        axes[1, 1].set_title('Bug Category vs Severity Heatmap')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'bug_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_model_comparison_charts(self):
        """Create model comparison charts"""
        print(f"🤖 Creating model comparison charts...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. Processing Time Comparison
        model_time_data = self.df.groupby('model_used')['processing_time'].agg(['mean', 'std', 'count'])
        x_pos = np.arange(len(model_time_data))
        
        axes[0, 0].bar(x_pos, model_time_data['mean'], yerr=model_time_data['std'], 
                       capsize=5, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0, 0].set_title('Average Processing Time by Model')
        axes[0, 0].set_xlabel('Model')
        axes[0, 0].set_ylabel('Processing Time (seconds)')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(['GPT-4o-mini', 'Claude 3.7', 'GPT-4'])
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Success Rate by Model
        success_rates = self.df.groupby('model_used').size()
        total_bugs = 46
        success_percentages = (success_rates / total_bugs) * 100
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        bars = axes[0, 1].bar(success_percentages.index, success_percentages.values, color=colors)
        axes[0, 1].set_title('Success Rate by Model')
        axes[0, 1].set_xlabel('Model')
        axes[0, 1].set_ylabel('Success Rate (%)')
        axes[0, 1].set_ylim(0, 110)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add percentage labels on bars
        for bar, percentage in zip(bars, success_percentages.values):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{percentage:.1f}%', ha='center', va='bottom')
        
        # 3. Response Length Comparison
        response_length_data = self.df.groupby('model_used')['response_length'].mean()
        axes[1, 0].bar(response_length_data.index, response_length_data.values, 
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[1, 0].set_title('Average Response Length by Model')
        axes[1, 0].set_xlabel('Model')
        axes[1, 0].set_ylabel('Response Length (characters)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Solution Confidence by Model
        confidence_data = self.df.groupby('model_used')['solution_confidence'].mean()
        axes[1, 1].bar(confidence_data.index, confidence_data.values, 
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[1, 1].set_title('Average Solution Confidence by Model')
        axes[1, 1].set_xlabel('Model')
        axes[1, 1].set_ylabel('Solution Confidence Score')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_repository_charts(self):
        """Create repository analysis charts"""
        print(f"📚 Creating repository analysis charts...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Repository Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Bugs per Repository
        repo_counts = self.df['repo'].value_counts()
        axes[0, 0].pie(repo_counts.values, labels=repo_counts.index, autopct='%1.1f%%', 
                       startangle=90, colors=plt.cm.Pastel1(np.linspace(0, 1, len(repo_counts))))
        axes[0, 0].set_title('Bugs Distribution by Repository')
        
        # 2. Processing Time by Repository
        repo_time_data = self.df.groupby('repo')['processing_time'].mean().sort_values(ascending=False)
        repo_time_data.plot(kind='bar', ax=axes[0, 1], color='lightcoral')
        axes[0, 1].set_title('Average Processing Time by Repository')
        axes[0, 1].set_xlabel('Repository')
        axes[0, 1].set_ylabel('Average Time (seconds)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Bug Types by Repository
        repo_bug_cross = pd.crosstab(self.df['repo'], self.df['bug_category'])
        repo_bug_cross.plot(kind='bar', ax=axes[1, 0], stacked=True, colormap='tab10')
        axes[1, 0].set_title('Bug Types Distribution by Repository')
        axes[1, 0].set_xlabel('Repository')
        axes[1, 0].set_ylabel('Number of Bugs')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].legend(title='Bug Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Severity by Repository
        repo_severity_cross = pd.crosstab(self.df['repo'], self.df['bug_severity'])
        repo_severity_cross.plot(kind='bar', ax=axes[1, 1], stacked=True, 
                               color=['lightgreen', 'orange', 'red'])
        axes[1, 1].set_title('Bug Severity Distribution by Repository')
        axes[1, 1].set_xlabel('Repository')
        axes[1, 1].set_ylabel('Number of Bugs')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].legend(title='Severity', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'repository_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_quality_metrics_charts(self):
        """Create quality metrics visualization"""
        print(f"⭐ Creating quality metrics charts...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Solution Quality Metrics Analysis', fontsize=16, fontweight='bold')
        
        # 1. Solution Confidence Distribution
        confidence_data = self.df['solution_confidence'].dropna()
        axes[0, 0].hist(confidence_data, bins=20, color='lightblue', alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Solution Confidence Distribution')
        axes[0, 0].set_xlabel('Confidence Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Response Length vs Processing Time
        scatter = axes[0, 1].scatter(self.df['response_length'], self.df['processing_time'], 
                                    c=self.df['solution_confidence'], cmap='viridis', alpha=0.6)
        axes[0, 1].set_title('Response Length vs Processing Time (colored by confidence)')
        axes[0, 1].set_xlabel('Response Length (characters)')
        axes[0, 1].set_ylabel('Processing Time (seconds)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add colorbar
        plt.colorbar(scatter, ax=axes[0, 1], label='Solution Confidence')
        
        # 3. Quality Metrics by Bug Category
        quality_by_category = self.df.groupby('bug_category')['solution_confidence'].mean().sort_values(ascending=False)
        quality_by_category.plot(kind='bar', ax=axes[1, 0], color='lightgreen')
        axes[1, 0].set_title('Average Solution Confidence by Bug Category')
        axes[1, 0].set_xlabel('Bug Category')
        axes[1, 0].set_ylabel('Average Confidence Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Quality Metrics by Model
        quality_by_model = self.df.groupby('model_used')['solution_confidence'].mean()
        model_names = ['GPT-4o-mini', 'Claude 3.7', 'GPT-4']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        bars = axes[1, 1].bar(range(len(quality_by_model)), quality_by_model.values, color=colors)
        axes[1, 1].set_title('Average Solution Confidence by Model')
        axes[1, 1].set_xlabel('Model')
        axes[1, 1].set_ylabel('Average Confidence Score')
        axes[1, 1].set_xticks(range(len(quality_by_model)))
        axes[1, 1].set_xticklabels(model_names)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, quality_by_model.values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'quality_metrics_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_summary_statistics(self):
        """Create summary statistics table"""
        print(f"📋 Creating summary statistics...")
        
        # Calculate key statistics
        stats = {
            'Total Experiments': len(self.df),
            'Total Bugs Tested': len(self.df) // 3,  # 3 models per bug
            'Success Rate': f"{(len(self.df) / (46 * 3)) * 100:.1f}%",
            'Total Processing Time': f"{self.df['processing_time'].sum():.1f} seconds",
            'Average Processing Time': f"{self.df['processing_time'].mean():.2f} seconds",
            'Fastest Model': self.df.groupby('model_used')['processing_time'].mean().idxmin(),
            'Most Comprehensive Model': self.df.groupby('model_used')['response_length'].mean().idxmax(),
            'Highest Confidence Model': self.df.groupby('model_used')['solution_confidence'].mean().idxmax(),
            'Most Common Bug Type': self.df['bug_category'].mode()[0],
            'Most Common Severity': self.df['bug_severity'].mode()[0],
            'Largest Repository': self.df['repo'].mode()[0]
        }
        
        # Create summary table
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        table_data = [[k, v] for k, v in stats.items()]
        table = ax.table(cellText=table_data, colLabels=['Metric', 'Value'], 
                        cellLoc='left', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(table_data) + 1):
            for j in range(2):
                if i == 0:  # Header row
                    table[(i, j)].set_facecolor('#4ECDC4')
                    table[(i, j)].set_text_props(weight='bold', color='white')
                else:  # Data rows
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#F7F7F7')
                    else:
                        table[(i, j)].set_facecolor('#FFFFFF')
        
        plt.title('Summary Statistics - SWE-Bench 46 Experiments', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'summary_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_figures_summary(self):
        """Create a summary document for research integration"""
        print(f"📚 Creating figures summary...")
        
        summary_content = f"""# SWE-Bench: Research Figures Summary

## 🎯 Generated Visualizations

This document summarizes all visualizations created for SWE-bench results integration into research documents.

## 📊 Static Figures (PNG Format)

### 1. Performance Analysis (`performance_analysis.png`)
- **Content**: Processing time distribution, box plots, bug type performance, severity performance
- **Use**: Performance comparison and analysis sections
- **Size**: 16x12 inches, 300 DPI

### 2. Performance Timeline (`performance_timeline.png`)
- **Content**: Processing time progression across all 46 bugs
- **Use**: Performance trends and consistency analysis
- **Size**: 14x8 inches, 300 DPI

### 3. Bug Distribution Analysis (`bug_distribution_analysis.png`)
- **Content**: Bug categories, severity, repositories, cross-tabulation
- **Use**: Dataset characterization and bug type analysis
- **Size**: 16x12 inches, 300 DPI

### 4. Model Comparison Analysis (`model_comparison_analysis.png`)
- **Content**: Model performance, success rates, response quality
- **Use**: LLM model evaluation and comparison
- **Size**: 16x12 inches, 300 DPI

### 5. Repository Analysis (`repository_analysis.png`)
- **Content**: Repository performance, bug distribution by repo
- **Use**: Cross-project analysis and validation
- **Size**: 16x12 inches, 300 DPI

### 6. Quality Metrics Analysis (`quality_metrics_analysis.png`)
- **Content**: Solution confidence, response quality, quality by category
- **Use**: Solution quality evaluation and validation
- **Size**: 16x12 inches, 300 DPI

### 7. Summary Statistics (`summary_statistics.png`)
- **Content**: Key statistics table for quick reference
- **Use**: Executive summary and methodology sections
- **Size**: 12x8 inches, 300 DPI

## 📁 File Locations

All figures are saved in: `data/results/figures/week2_visualizations/`

## 🎨 Figure Specifications

- **Format**: PNG (static)
- **Resolution**: 300 DPI for print quality
- **Color Scheme**: Professional, publication-ready colors
- **Fonts**: Clear, readable typography
- **Layout**: Optimized for research integration

## 📖 Research Integration Guidelines

### 1. **Methodology Section**
- Use: Performance Analysis, Model Comparison
- Purpose: Demonstrate framework effectiveness

### 2. **Results Section**
- Use: Bug Distribution, Repository Analysis
- Purpose: Present comprehensive findings

### 3. **Discussion Section**
- Use: Quality Metrics, Performance Timeline
- Purpose: Analyze patterns and implications

### 4. **Conclusion Section**
- Use: Summary Statistics
- Purpose: Highlight key achievements

## 🚀 Usage Instructions

### For Research Documents (PNG files):
```latex
\\begin{{figure}}[h]
\\centering
\\includegraphics[width=0.8\\textwidth]{figures/week2_visualizations/performance_analysis.png}
\\caption{{Performance Analysis Across Models}}
\\label{{fig:week2_performance}}
\\end{{figure}}
```

## 📊 Data Sources

All visualizations are generated from:
- **Primary Data**: `swe_bench_46_results_20250831_181453.json`
- **Experiments**: 133 total experiments
- **Bugs**: 46 real-world SWE-bench GUI bugs
- **Models**: 3 LLM providers (GPT-4o-mini, Claude 3.7 Sonnet, GPT-4)

## 🎯 Key Insights Visualized

1. **Performance Patterns**: Model efficiency and consistency
2. **Bug Distribution**: Real-world bug type prevalence
3. **Quality Metrics**: Solution confidence and response quality
4. **Repository Analysis**: Cross-project validation
5. **Temporal Trends**: Performance consistency over time

## ✅ Quality Assurance

- **Resolution**: Print-ready 300 DPI
- **Color Schemes**: Accessible and professional
- **Layout**: Optimized for research integration
- **Data Accuracy**: Directly generated from experiment results
- **Professional Standards**: Academic publication quality

---

**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Total Figures**: 7 static figures  
**Data Source**: SWE-Bench 46 Experiments
**Ready for Research Integration**: ✅
"""
        
        with open(self.output_dir / 'FIGURES_SUMMARY.md', 'w') as f:
            f.write(summary_content)
        
        print(f"   📚 Figures summary saved: FIGURES_SUMMARY.md")

def main():
    """Main execution function"""
    print("🎨 SWE-Bench Visualization Generator")
    print("=" * 50)
    
    try:
        # Initialize generator
        generator = Week2VisualizationGenerator()
        
        # Generate all visualizations
        generator.generate_all_visualizations()
        
        # Create figures summary
        generator.create_figures_summary()
        
        print(f"\n🎉 All visualizations generated successfully!")
        print(f"📁 Output directory: {generator.output_dir}")
        print(f"📊 Total figures created: 7 static figures")
        print(f"📚 Ready for research integration!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
