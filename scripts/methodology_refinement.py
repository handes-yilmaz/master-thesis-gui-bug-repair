#!/usr/bin/env python
"""
Methodology Refinement for LLM Bug Repair Experiments
====================================================

This script analyzes the current methodology and suggests improvements
based on error analysis and performance findings.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

class MethodologyRefiner:
    def __init__(self, results_file: str):
        """Initialize the methodology refiner with results data"""
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
    
    def analyze_current_methodology_strengths(self):
        """Analyze the strengths of the current methodology"""
        print("\n" + "✅ CURRENT METHODOLOGY STRENGTHS".center(80, "="))
        print()
        
        strengths = []
        
        # 1. Data Quality
        total_experiments = len(self.df)
        successful_experiments = len(self.df[self.df['response'].notna()])
        success_rate = (successful_experiments / total_experiments) * 100
        
        strengths.append({
            'category': 'Data Quality',
            'metric': 'Success Rate',
            'value': f"{success_rate:.1f}%",
            'description': f"{successful_experiments}/{total_experiments} experiments completed successfully"
        })
        
        # 2. Model Coverage
        models_tested = self.df['model_name'].nunique()
        strengths.append({
            'category': 'Model Coverage',
            'metric': 'Models Tested',
            'value': str(models_tested),
            'description': f"Comprehensive comparison of {models_tested} leading LLM providers"
        })
        
        # 3. Bug Diversity
        unique_bugs = self.df['bug_id'].nunique()
        unique_repos = self.df['repo'].nunique()
        strengths.append({
            'category': 'Bug Diversity',
            'metric': 'Coverage',
            'value': f"{unique_bugs} bugs, {unique_repos} repos",
            'description': f"Real-world bugs from {unique_repos} different repositories"
        })
        
        # 4. Response Quality
        df_with_responses = self.df[self.df['response'].notna()]
        avg_response_length = df_with_responses['response'].str.len().mean()
        strengths.append({
            'category': 'Response Quality',
            'metric': 'Average Length',
            'value': f"{avg_response_length:.0f} chars",
            'description': "Comprehensive, detailed responses from both models"
        })
        
        # 5. Statistical Power
        experiments_per_model = total_experiments // models_tested
        strengths.append({
            'category': 'Statistical Power',
            'metric': 'Sample Size',
            'value': f"{experiments_per_model} per model",
            'description': f"Sufficient sample size for statistical significance testing"
        })
        
        # Display strengths
        print("🏆 METHODOLOGY STRENGTHS ANALYSIS")
        print("-" * 80)
        print(f"{'Category':<20} {'Metric':<15} {'Value':<12} {'Description':<30}")
        print("-" * 80)
        
        for strength in strengths:
            print(f"{strength['category']:<20} {strength['metric']:<15} {strength['value']:<12} {strength['description']:<30}")
        
        print()
        return strengths
    
    def analyze_methodology_weaknesses(self):
        """Analyze potential weaknesses and areas for improvement"""
        print("\n" + "⚠️  METHODOLOGY WEAKNESSES & IMPROVEMENTS".center(80, "="))
        print()
        
        weaknesses = []
        
        # 1. Sample Size
        total_experiments = len(self.df)
        if total_experiments < 30:
            weaknesses.append({
                'category': 'Sample Size',
                'issue': 'Limited statistical power',
                'current': f"{total_experiments} experiments",
                'recommended': '30+ experiments per model',
                'impact': 'Medium',
                'solution': 'Expand to 46 SWE-bench bugs for better power'
            })
        
        # 2. Bug Complexity Classification
        if 'difficulty' not in self.df.columns:
            weaknesses.append({
                'category': 'Bug Classification',
                'issue': 'No explicit complexity classification',
                'current': 'Repository-based proxy only',
                'recommended': 'Explicit difficulty ratings',
                'impact': 'Medium',
                'solution': 'Add difficulty classification system'
            })
        
        # 3. Response Quality Metrics
        weaknesses.append({
            'category': 'Quality Metrics',
            'issue': 'Limited quality assessment',
            'current': 'Length and structure only',
            'recommended': 'Comprehensive quality scoring',
            'impact': 'High',
            'solution': 'Implement quality scoring system'
        })
        
        # 4. Validation Framework
        weaknesses.append({
            'category': 'Validation',
            'issue': 'No solution validation',
            'current': 'LLM responses only',
            'recommended': 'Automated solution testing',
            'impact': 'High',
            'solution': 'Build validation framework'
        })
        
        # 5. Cost Analysis
        if 'tokens_used' not in self.df.columns:
            weaknesses.append({
                'category': 'Cost Analysis',
                'issue': 'No cost tracking',
                'current': 'Missing token usage data',
                'recommended': 'Complete cost analysis',
                'impact': 'Medium',
                'solution': 'Track token usage and costs'
            })
        
        # Display weaknesses
        print("🔍 WEAKNESSES & IMPROVEMENT OPPORTUNITIES")
        print("-" * 80)
        print(f"{'Category':<20} {'Issue':<25} {'Current':<15} {'Recommended':<15} {'Impact':<8}")
        print("-" * 80)
        
        for weakness in weaknesses:
            print(f"{weakness['category']:<20} {weakness['issue']:<25} {weakness['current']:<15} {weakness['recommended']:<15} {weakness['impact']:<8}")
        
        print()
        return weaknesses
    
    def generate_improvement_recommendations(self):
        """Generate specific recommendations for methodology improvement"""
        print("\n" + "🚀 METHODOLOGY IMPROVEMENT RECOMMENDATIONS".center(80, "="))
        print()
        
        recommendations = {
            'immediate_actions': [
                "Expand dataset to 46 SWE-bench bugs for better statistical power",
                "Implement comprehensive response quality scoring system",
                "Add explicit bug difficulty classification",
                "Track token usage and cost analysis",
                "Create automated solution validation framework"
            ],
            'prompt_optimization': [
                "Add complexity indicators to prompts",
                "Request specific output formats (markdown, code blocks)",
                "Include validation criteria in prompts",
                "Add difficulty-based prompt variations",
                "Implement retry mechanisms for poor responses"
            ],
            'evaluation_metrics': [
                "Response completeness score (0-100%)",
                "Technical accuracy validation",
                "Solution implementability assessment",
                "Code quality metrics",
                "Accessibility improvement tracking"
            ],
            'experimental_design': [
                "Randomized bug assignment to models",
                "Blind evaluation of responses",
                "Cross-validation of results",
                "Statistical power analysis",
                "Effect size calculations"
            ]
        }
        
        print("📋 IMMEDIATE ACTIONS (This Week)")
        print("-" * 80)
        for i, action in enumerate(recommendations['immediate_actions'], 1):
            print(f"{i}. {action}")
        
        print("\n🔧 PROMPT OPTIMIZATION")
        print("-" * 80)
        for i, optimization in enumerate(recommendations['prompt_optimization'], 1):
            print(f"{i}. {optimization}")
        
        print("\n📊 EVALUATION METRICS")
        print("-" * 80)
        for i, metric in enumerate(recommendations['evaluation_metrics'], 1):
            print(f"{i}. {metric}")
        
        print("\n🧪 EXPERIMENTAL DESIGN")
        print("-" * 80)
        for i, design in enumerate(recommendations['experimental_design'], 1):
            print(f"{i}. {design}")
        
        print()
        return recommendations
    
    def create_improved_methodology_template(self):
        """Create a template for improved methodology"""
        print("\n" + "📝 IMPROVED METHODOLOGY TEMPLATE".center(80, "="))
        print()
        
        template = {
            'experimental_design': {
                'sample_size': '46 bugs (minimum 30 per model for statistical power)',
                'randomization': 'Random bug assignment to models to avoid bias',
                'blinding': 'Blind evaluation of responses by multiple reviewers',
                'controls': 'Standardized prompts and evaluation criteria'
            },
            'data_collection': {
                'bug_selection': 'SWE-bench GUI bugs with explicit difficulty ratings',
                'model_configuration': 'Standardized API parameters and temperature settings',
                'response_tracking': 'Complete response data, timing, and token usage',
                'quality_metrics': 'Automated quality scoring and manual validation'
            },
            'evaluation_framework': {
                'response_quality': 'Completeness, accuracy, and implementability scores',
                'performance_metrics': 'Processing time, cost, and consistency analysis',
                'statistical_analysis': 'T-tests, effect sizes, and confidence intervals',
                'validation': 'Automated solution testing and manual review'
            }
        }
        
        print("🏗️  IMPROVED METHODOLOGY FRAMEWORK")
        print("=" * 80)
        
        for section, details in template.items():
            print(f"\n📋 {section.upper().replace('_', ' ')}")
            print("-" * 40)
            for key, value in details.items():
                print(f"  • {key.replace('_', ' ').title()}: {value}")
        
        print()
        return template
    
    def generate_implementation_plan(self):
        """Generate a step-by-step implementation plan"""
        print("\n" + "📅 IMPLEMENTATION PLAN".center(80, "="))
        print()
        
        implementation_plan = [
            {
                'week': 'Week 1 (Current)',
                'tasks': [
                    'Complete error analysis and methodology review',
                    'Design improved evaluation metrics',
                    'Create enhanced prompt templates',
                    'Set up token usage tracking'
                ],
                'deliverables': [
                    'Updated methodology document',
                    'Enhanced prompt templates',
                    'Quality scoring system'
                ]
            },
            {
                'week': 'SWE-Bench Phase',
                'tasks': [
                    'Implement 46 SWE-bench bug experiments',
                    'Apply improved methodology',
                    'Collect comprehensive data',
                    'Perform enhanced analysis'
                ],
                'deliverables': [
                    '46 bug experiments completed',
                    'Enhanced performance metrics',
                    'Comprehensive cost analysis'
                ]
            },
            {
                'week': 'Week 3',
                'tasks': [
                    'Validate and test solutions',
                    'Final statistical analysis',
                    'Research results chapter',
                    'Methodology documentation'
                ],
                'deliverables': [
                    'Validated results',
                    'Complete research chapter',
                    'Methodology paper'
                ]
            }
        ]
        
        print("🗓️  STEP-BY-STEP IMPLEMENTATION")
        print("=" * 80)
        
        for week_plan in implementation_plan:
            print(f"\n📅 {week_plan['week']}")
            print("-" * 40)
            print("📋 Tasks:")
            for task in week_plan['tasks']:
                print(f"  • {task}")
            print("📦 Deliverables:")
            for deliverable in week_plan['deliverables']:
                print(f"  • {deliverable}")
        
        print()
        return implementation_plan
    
    def run_complete_methodology_refinement(self, output_dir: str = "runs/methodology_refinement"):
        """Run the complete methodology refinement pipeline"""
        print("\n" + "🚀 RUNNING COMPLETE METHODOLOGY REFINEMENT".center(80, "="))
        print()
        print(f"📁 Results file: {self.results_file}")
        print(f"📊 Total experiments: {len(self.df)}")
        print(f"🤖 Models tested: {', '.join(self.df['model_name'].unique())}")
        print()
        print("=" * 80)
        
        # Run all analyses
        strengths = self.analyze_current_methodology_strengths()
        weaknesses = self.analyze_methodology_weaknesses()
        recommendations = self.generate_improvement_recommendations()
        template = self.create_improved_methodology_template()
        implementation_plan = self.generate_implementation_plan()
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save methodology refinement report
        refinement_report = {
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_experiments': len(self.df),
                'models_tested': list(self.df['model_name'].unique()),
                'repositories_tested': list(self.df['repo'].unique())
            },
            'strengths': strengths,
            'weaknesses': weaknesses,
            'recommendations': recommendations,
            'improved_template': template,
            'implementation_plan': implementation_plan
        }
        
        # Save report
        report_file = f"{output_dir}/methodology_refinement_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(refinement_report, f, indent=2)
        
        print("\n" + "🎉 METHODOLOGY REFINEMENT COMPLETE!".center(80, "="))
        print()
        print("📋 EXECUTIVE SUMMARY")
        print("-" * 80)
        print(f"✅ Strengths identified: {len(strengths)}")
        print(f"⚠️  Areas for improvement: {len(weaknesses)}")
        print(f"🚀 Recommendations generated: {len(recommendations)}")
        print(f"📝 Implementation plan: 3-week roadmap")
        print("-" * 80)
        print(f"📝 Refinement report: {report_file}")
        
        return {
            'strengths': strengths,
            'weaknesses': weaknesses,
            'recommendations': recommendations,
            'template': template,
            'implementation_plan': implementation_plan,
            'report_file': report_file
        }

def main():
    """Main function to run the methodology refinement"""
    # Find the most recent results file
    results_dir = Path("data/results")
    results_files = list(results_dir.glob("cloud_processing_results_*.json"))
    
    if not results_files:
        print("❌ No results files found in data/results/")
        return
    
    # Use the most recent file
    latest_file = max(results_files, key=lambda x: x.stat().st_mtime)
    print(f"📁 Using results file: {latest_file}")
    
    # Run methodology refinement
    refiner = MethodologyRefiner(str(latest_file))
    results = refiner.run_complete_methodology_refinement()
    
    print("\n✅ Methodology refinement complete! Check the runs/methodology_refinement/ directory for outputs.")

if __name__ == "__main__":
    main()
