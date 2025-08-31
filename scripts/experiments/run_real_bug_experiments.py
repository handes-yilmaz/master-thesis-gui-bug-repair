#!/usr/bin/env python
"""
Real SWE-bench GUI Bug Experiment Runner
========================================

This script runs experiments on real SWE-bench GUI bugs,
providing authentic research results.
"""

import json
from pathlib import Path
from scripts.experiments.enhanced_experiment_framework import EnhancedExperimentFramework

def run_real_bug_experiments():
    """Run experiments on real SWE-bench GUI bugs"""
    
    print("🔬 Real SWE-bench GUI Bug Experiments")
    print("=" * 50)
    
    # Initialize framework with real bugs
    framework = EnhancedExperimentFramework()
    
    print(f"✓ Framework initialized with {len(framework.llm_clients)} models")
    print(f"✓ Loaded {len(framework.advanced_scenarios)} real SWE-bench GUI bugs")
    
    # Show bug distribution
    show_bug_distribution(framework.advanced_scenarios)
    
    # Run experiments on real bugs
    all_results = []
    
    try:
        print("\n🚀 Starting experiments on real GUI bugs...")
        
        # Test different bug categories
        bug_categories = ['form', 'responsive', 'styling', 'ui_layout', 'interaction']
        
        for category in bug_categories:
            category_bugs = [bug for bug in framework.advanced_scenarios if bug.get('bug_category') == category]
            if category_bugs:
                print(f"\n📊 Testing {len(category_bugs)} {category} bugs...")
                
                # Test first 3 bugs in each category to start
                test_bugs = category_bugs[:3]
                for bug in test_bugs:
                    print(f"  Testing: {bug.get('title', 'No title')[:60]}...")
                    
                    # Run experiment on this bug
                    result = framework.run_advanced_experiment(
                        bug_description=bug.get('description', ''),
                        modality='text_only',
                        prompt_template='technical_detailed',
                        bug_category=bug.get('bug_category', ''),
                        bug_severity=bug.get('severity', ''),
                        bug_difficulty=bug.get('difficulty', '')
                    )
                    
                    if result:
                        all_results.append(result)
                        print(f"    ✅ Experiment completed")
                    else:
                        print(f"    ❌ Experiment failed")
        
        # Save results
        if all_results:
            print(f"\n💾 Saving {len(all_results)} experiment results...")
            filename = framework.save_results(all_results)
            
            print("\n📊 Generating experiment report...")
            report = framework.generate_experiment_report(all_results)
            print(report)
            
            report_file = Path(framework.results_dir) / f"real_bugs_report_{filename.replace('.json', '.txt')}"
            with open(report_file, 'w') as f:
                f.write(report)
            print(f"✓ Report saved to: {report_file}")
            
            print(f"\n🎉 Real bug experiments completed successfully!")
            print(f"Total experiments: {len(all_results)}")
            print(f"Results saved to: {filename}")
        else:
            print("\n❌ No results to save")
            
    except KeyboardInterrupt:
        print("\n⚠️ Experiments interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during experiments: {e}")

def show_bug_distribution(bugs):
    """Show the distribution of real bugs"""
    print("\n📊 Real SWE-bench GUI Bug Distribution:")
    
    # Count by category
    categories = {}
    for bug in bugs:
        category = bug.get('bug_category', 'unknown')
        if category not in categories:
            categories[category] = 0
        categories[category] += 1
    
    for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  {category}: {count} bugs")
    
    # Count by severity
    severities = {}
    for bug in bugs:
        severity = bug.get('severity', 'unknown')
        if severity not in severities:
            severities[severity] = 0
        severities[severity] += 1
    
    print("\nSeverity Distribution:")
    for severity, count in sorted(severities.items(), key=lambda x: x[1], reverse=True):
        print(f"  {severity}: {count} bugs")
    
    # Show sample bugs
    print("\nSample Real Bugs:")
    for i, bug in enumerate(bugs[:5]):
        title = bug.get('title', 'No title')[:60]
        category = bug.get('bug_category', 'unknown')
        severity = bug.get('severity', 'unknown')
        print(f"  {i+1}. [{category}/{severity}] {title}")

def main():
    """Main function to run real bug experiments"""
    print("🚀 Real SWE-bench GUI Bug Experiment Runner")
    print("=" * 50)
    
    # Check if real bugs are available
    real_bugs_file = Path("data/processed/real_swe_bench_scenarios.json")
    if not real_bugs_file.exists():
        print("❌ Real SWE-bench scenarios not found!")
        print("Run the update_experiment_framework.py script first.")
        return
    
    # Run experiments
    run_real_bug_experiments()

if __name__ == "__main__":
    main()
