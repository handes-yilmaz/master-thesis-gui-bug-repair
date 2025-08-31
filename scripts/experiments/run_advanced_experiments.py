#!/usr/bin/env python
"""
Run Advanced Bug Repair Experiments
==================================

This script runs comprehensive experiments using the enhanced framework
to test LLM capabilities on advanced bug scenarios.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_experiment_framework import EnhancedExperimentFramework

def run_security_experiments(framework):
    """Run experiments on security-related bug scenarios"""
    print("\n🔒 Running Security Bug Experiments...")
    
    security_scenarios = [s for s in framework.advanced_scenarios if s.bug_category == "security"]
    
    results = []
    for scenario in security_scenarios:
        print(f"\nTesting: {scenario.title}")
        print(f"Severity: {scenario.severity}, Difficulty: {scenario.difficulty}")
        
        # Test with text-only modality across models
        for model in framework.llm_clients.keys():
            try:
                result = framework.run_advanced_experiment(scenario, "text_only", model)
                results.append(result)
                confidence = result.evaluation_metrics["solution_confidence"]
                security_score = result.evaluation_metrics["security_awareness"]
                print(f"  ✓ {model}: {confidence:.3f} confidence, {security_score:.3f} security awareness")
            except Exception as e:
                print(f"  ✗ {model}: Failed - {e}")
    
    return results

def run_performance_experiments(framework):
    """Run experiments on performance-related bug scenarios"""
    print("\n⚡ Running Performance Bug Experiments...")
    
    perf_scenarios = [s for s in framework.advanced_scenarios if s.bug_category == "performance"]
    
    results = []
    for scenario in perf_scenarios:
        print(f"\nTesting: {scenario.title}")
        print(f"Severity: {scenario.severity}, Difficulty: {scenario.difficulty}")
        
        # Test with multimodal modality across models
        for model in framework.llm_clients.keys():
            try:
                result = framework.run_advanced_experiment(scenario, "multimodal", model)
                results.append(result)
                confidence = result.evaluation_metrics["solution_confidence"]
                perf_score = result.evaluation_metrics["performance_awareness"]
                print(f"  ✓ {model}: {confidence:.3f} confidence, {perf_score:.3f} performance awareness")
            except Exception as e:
                print(f"  ✗ {model}: Failed - {e}")
    
    return results

def run_cross_browser_experiments(framework):
    """Run experiments on cross-browser compatibility bug scenarios"""
    print("\n🌐 Running Cross-Browser Bug Experiments...")
    
    browser_scenarios = [s for s in framework.advanced_scenarios if s.bug_category == "cross_browser"]
    
    results = []
    for scenario in browser_scenarios:
        print(f"\nTesting: {scenario.title}")
        print(f"Severity: {scenario.severity}, Difficulty: {scenario.difficulty}")
        
        # Test with visual modality across models
        for model in framework.llm_clients.keys():
            try:
                result = framework.run_advanced_experiment(scenario, "visual", model)
                results.append(result)
                confidence = result.evaluation_metrics["solution_confidence"]
                browser_score = result.evaluation_metrics["browser_compatibility"]
                print(f"  ✓ {model}: {confidence:.3f} confidence, {browser_score:.3f} browser awareness")
            except Exception as e:
                print(f"  ✗ {model}: Failed - {e}")
    
    return results

def run_mobile_experiments(framework):
    """Run experiments on mobile-specific bug scenarios"""
    print("\n📱 Running Mobile Bug Experiments...")
    
    mobile_scenarios = [s for s in framework.advanced_scenarios if s.mobile_affected]
    
    results = []
    for scenario in mobile_scenarios:
        print(f"\nTesting: {scenario.title}")
        print(f"Severity: {scenario.severity}, Difficulty: {scenario.difficulty}")
        
        # Test with text-only modality across models
        for model in framework.llm_clients.keys():
            try:
                result = framework.run_advanced_experiment(scenario, "text_only", model)
                results.append(result)
                confidence = result.evaluation_metrics["solution_confidence"]
                print(f"  ✓ {model}: {confidence:.3f} confidence")
            except Exception as e:
                print(f"  ✗ {model}: Failed - {e}")
    
    return results

def run_model_comparison_experiments(framework):
    """Run model comparison experiments on selected scenarios"""
    print("\n🔍 Running Model Comparison Experiments...")
    
    # Select representative scenarios from each category
    representative_scenarios = [
        framework.advanced_scenarios[0],   # SEC_XSS_01
        framework.advanced_scenarios[3],   # PERF_MEMORY_04
        framework.advanced_scenarios[6],   # BROWSER_FLEXBOX_07
    ]
    
    results = []
    for scenario in representative_scenarios:
        print(f"\nComparing models on: {scenario.title}")
        print(f"Category: {scenario.bug_category}, Severity: {scenario.severity}")
        
        # Run with text-only modality across all models
        model_results = framework.run_model_comparison_experiment(scenario, "text_only")
        results.extend(model_results)
        
        # Print comparison
        for result in model_results:
            confidence = result.evaluation_metrics["solution_confidence"]
            time_taken = result.processing_time
            print(f"  {result.model_used}: {confidence:.3f} confidence, {time_taken:.2f}s")
    
    return results

def run_comprehensive_suite(framework):
    """Run comprehensive test suite across all scenarios and modalities"""
    print("\n🚀 Running Comprehensive Test Suite...")
    
    # Run with a subset of scenarios to avoid excessive API calls
    test_scenarios = framework.advanced_scenarios[:5]  # First 5 scenarios
    
    print(f"Testing {len(test_scenarios)} scenarios with text-only modality")
    results = []
    
    for scenario in test_scenarios:
        print(f"\nTesting: {scenario.title}")
        for model in framework.llm_clients.keys():
            try:
                result = framework.run_advanced_experiment(scenario, "text_only", model)
                results.append(result)
                confidence = result.evaluation_metrics["solution_confidence"]
                print(f"  ✓ {model}: {confidence:.3f} confidence")
            except Exception as e:
                print(f"  ✗ {model}: Failed - {e}")
    
    return results

def main():
    """Main function to run all experiments"""
    print("🔬 Advanced Bug Repair LLM Experiments")
    print("=" * 50)
    
    # Initialize framework
    framework = EnhancedExperimentFramework()
    
    print(f"✓ Framework initialized with {len(framework.llm_clients)} models")
    print(f"✓ Loaded {len(framework.advanced_scenarios)} advanced scenarios")
    
    all_results = []
    
    # Run different types of experiments
    try:
        # 1. Security experiments
        security_results = run_security_experiments(framework)
        all_results.extend(security_results)
        
        # 2. Performance experiments
        perf_results = run_performance_experiments(framework)
        all_results.extend(perf_results)
        
        # 3. Cross-browser experiments
        browser_results = run_cross_browser_experiments(framework)
        all_results.extend(browser_results)
        
        # 4. Mobile experiments
        mobile_results = run_mobile_experiments(framework)
        all_results.extend(mobile_results)
        
        # 5. Model comparison experiments
        comparison_results = run_model_comparison_experiments(framework)
        all_results.extend(comparison_results)
        
        # 6. Comprehensive test suite
        comprehensive_results = run_comprehensive_suite(framework)
        all_results.extend(comprehensive_results)
        
    except KeyboardInterrupt:
        print("\n⚠️ Experiments interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during experiments: {e}")
    
    # Save results
    if all_results:
        print(f"\n💾 Saving {len(all_results)} experiment results...")
        filename = framework.save_results(all_results)
        
        # Generate and display report
        print("\n📊 Generating experiment report...")
        report = framework.generate_experiment_report(all_results)
        print(report)
        
        # Save report to file
        report_file = Path(framework.results_dir) / f"report_{filename.replace('.json', '.txt')}"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"✓ Report saved to: {report_file}")
        
        print(f"\n🎉 Experiments completed successfully!")
        print(f"Total experiments: {len(all_results)}")
        print(f"Results saved to: {filename}")
    else:
        print("\n❌ No results to save")

if __name__ == "__main__":
    main()
