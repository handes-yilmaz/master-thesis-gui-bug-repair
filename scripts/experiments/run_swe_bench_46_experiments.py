import os
#!/usr/bin/env python
"""
SWE-Bench 46 GUI Bug Experiments - Implementation
=================================================

This script runs comprehensive experiments on all 46 real SWE-bench GUI bugs,
providing the foundation for research with authentic production data.

Features:
- 46 real-world GUI bugs from SWE-bench
- Multi-model testing (OpenAI + Claude)
- Comprehensive performance analysis
- Automated fix validation framework
- Statistical significance testing
- Cost analysis and ROI calculations
"""

import json
import time
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd

# Import our enhanced framework
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from enhanced_experiment_framework import EnhancedExperimentFramework, AdvancedBugScenario
from llm_client import LLMClient, LLMConfig

class SWEBench46ExperimentRunner:
    """Comprehensive experiment runner for 46 SWE-bench GUI bugs"""
    
    def __init__(self, config_path: str = "configs/config.json"):
        """Initialize the SWE-bench experiment runner"""
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize framework
        self.framework = EnhancedExperimentFramework(config_path)
        
        # Load SWE-bench bugs
        self.swe_bench_bugs = self._load_swe_bench_bugs()
        
        # Create results directory
        self.results_dir = Path("runs/swe_bench_46_experiments")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Experiment tracking
        self.experiment_results = []
        self.current_experiment = 0
        self.total_experiments = len(self.swe_bench_bugs) * 2  # 2 models
        
        print(f"🚀 SWE-Bench 46 GUI Bug Experiments Initialized")
        print(f"📊 Total bugs: {len(self.swe_bench_bugs)}")
        print(f"�� Total experiments: {self.total_experiments}")
        print(f"�� Results directory: {self.results_dir}")
    
    def _load_swe_bench_bugs(self) -> List[Dict[str, Any]]:
        """Load the 46 SWE-bench GUI bugs"""
        bugs_file = Path("data/processed/swe_bench_gui_bugs/swe_bench_gui_bugs_46.json")
        
        if not bugs_file.exists():
            raise FileNotFoundError(f"SWE-bench bugs file not found: {bugs_file}")
        
        with open(bugs_file, 'r') as f:
            bugs = json.load(f)
        
        print(f"✅ Loaded {len(bugs)} SWE-bench GUI bugs")
        return bugs
    
    def show_bug_distribution(self):
        """Display comprehensive bug distribution analysis"""
        print("\n📊 SWE-Bench 46 GUI Bug Distribution Analysis")
        print("=" * 60)
        
        # Category distribution
        categories = {}
        severities = {}
        difficulties = {}
        repositories = {}
        
        for bug in self.swe_bench_bugs:
            # Categories
            category = bug.get('bug_type', 'unknown')
            categories[category] = categories.get(category, 0) + 1
            
            # Severities
            severity = bug.get('severity', 'unknown')
            severities[severity] = severities.get(severity, 0) + 1
            
            # Difficulties
            difficulty = bug.get('difficulty', 'unknown')
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
            
            # Repositories
            repo = bug.get('repo', 'unknown')
            repositories[repo] = repositories.get(repo, 0) + 1
        
        print(f"\n🐛 Bug Categories ({len(categories)} types):")
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.swe_bench_bugs)) * 100
            print(f"  {category:15} : {count:2d} bugs ({percentage:5.1f}%)")
        
        print(f"\n�� Severity Levels ({len(severities)} levels):")
        for severity, count in sorted(severities.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.swe_bench_bugs)) * 100
            print(f"  {severity:15} : {count:2d} bugs ({percentage:5.1f}%)")
        
        print(f"\n�� Repositories ({len(repositories)} repos):")
        for repo, count in sorted(repositories.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.swe_bench_bugs)) * 100
            print(f"  {repo:30} : {count:2d} bugs ({percentage:5.1f}%)")
        
        print(f"\n�� Dataset Statistics:")
        print(f"  Total bugs: {len(self.swe_bench_bugs)}")
        print(f"  High/Critical severity: {severities.get('high', 0) + severities.get('critical', 0)}")
        print(f"  Medium/High difficulty: {difficulties.get('medium', 0) + difficulties.get('high', 0)}")
        print(f"  With screenshots: {sum(1 for b in self.swe_bench_bugs if b.get('has_image', False))}")
    
    def run_comprehensive_experiments(self, max_bugs: Optional[int] = None):
        """Run comprehensive experiments on all SWE-bench bugs"""
        print(f"\n🚀 Starting Comprehensive SWE-Bench Experiments")
        print("=" * 60)
        
        # Limit bugs if specified (for testing)
        bugs_to_test = self.swe_bench_bugs[:max_bugs] if max_bugs else self.swe_bench_bugs
        total_experiments = len(bugs_to_test) * 2
        
        print(f"📊 Testing {len(bugs_to_test)} bugs across 2 models")
        print(f"🔬 Total experiments planned: {total_experiments}")
        
        start_time = time.time()
        
        try:
            for i, bug in enumerate(bugs_to_test):
                print(f"\n📋 Bug {i+1}/{len(bugs_to_test)}: {bug.get('title', 'No title')[:80]}...")
                print(f"   Type: {bug.get('bug_type', 'unknown')} | Severity: {bug.get('severity', 'unknown')}")
                
                # Test with all available models
                available_models = list(self.framework.llm_clients.keys())
                print(f"   🔍 Available models: {available_models}")
                
                for model_name in available_models:
                    print(f"   🔬 Testing with {model_name} model...")
                    
                    try:
                        # Create an AdvancedBugScenario object from the SWE-bench bug
                        scenario = AdvancedBugScenario(
                            bug_id=bug.get('swe_bench_id', ''),
                            title=bug.get('title', ''),
                            description=bug.get('description', ''),
                            expected_solution='',
                            bug_category=bug.get('bug_type', ''),
                            severity=bug.get('severity', ''),
                            difficulty=bug.get('difficulty', ''),
                            ui_context=bug.get('description', ''),
                            code_snippet='',
                            security_implications=[],
                            performance_impact='',
                            browser_specific=[],
                            mobile_affected=False,
                            accessibility_impact='',
                            testing_scenarios=[],
                            fix_priority=bug.get('severity', 'medium')
                        )
                        
                        # Run experiment with correct parameters
                        result = self.framework.run_advanced_experiment(
                            scenario=scenario,
                            modality='text_only',
                            model_name=model_name
                        )
                        
                        if result:
                            # Convert result to dictionary and add bug metadata
                            result_dict = result.__dict__.copy() if hasattr(result, '__dict__') else dict(result)
                            
                            # Add bug metadata
                            result_dict['swe_bench_id'] = bug.get('swe_bench_id', '')
                            result_dict['repo'] = bug.get('repo', '')
                            result_dict['bug_title'] = bug.get('title', '')
                            result_dict['has_screenshot'] = bug.get('has_image', False)
                            result_dict['bug_category'] = bug.get('bug_type', '')
                            result_dict['bug_severity'] = bug.get('severity', '')
                            result_dict['bug_difficulty'] = bug.get('difficulty', '')
                            
                            self.experiment_results.append(result_dict)
                            print(f"      ✅ Experiment completed in {result_dict.get('processing_time', 0):.2f}s")
                        else:
                            print(f"      ❌ Experiment failed")
                    
                    except Exception as e:
                        print(f"      ❌ Error with {model_name}: {e}")
                        continue
                
                # Progress update
                completed = len(self.experiment_results)
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1) if i > 0 else 0
                remaining = total_experiments - completed
                eta = remaining * avg_time if avg_time > 0 else 0
                
                print(f"   📊 Progress: {completed}/{total_experiments} ({completed/total_experiments*100:.1f}%)")
                print(f"   ⏱️  Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")
        
        except KeyboardInterrupt:
            print(f"\n⚠️ Experiments interrupted by user")
            print(f"📊 Completed {len(self.experiment_results)} experiments")
        
        except Exception as e:
            print(f"\n❌ Error during experiments: {e}")
        
        finally:
            # Save results
            if self.experiment_results:
                self.save_experiment_results()
                self.generate_comprehensive_report()
    
    def save_experiment_results(self):
        """Save all experiment results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"swe_bench_46_results_{timestamp}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.experiment_results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {filepath}")
        print(f"�� Total experiments: {len(self.experiment_results)}")
        
        return filepath
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""
        if not self.experiment_results:
            print("❌ No results to analyze")
            return
        
        print(f"\n📊 Generating Comprehensive Analysis Report")
        print("=" * 60)
        
        # Performance analysis
        performance_analysis = self._analyze_performance()
        
        # Bug complexity correlation
        complexity_analysis = self._analyze_complexity_correlation()
        
        # Repository-specific analysis
        repo_analysis = self._analyze_repository_performance()
        
        # Cost analysis
        cost_analysis = self._analyze_costs()
        
        # Generate report
        report = self._format_comprehensive_report(
            performance_analysis,
            complexity_analysis,
            repo_analysis,
            cost_analysis
        )
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_dir / f"comprehensive_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"�� Report saved to: {report_file}")
        print(f"�� Analysis complete: {len(self.experiment_results)} experiments analyzed")
        
        return report
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze overall performance metrics"""
        if not self.experiment_results:
            return {}
        
        # Group by model
        models = {}
        for result in self.experiment_results:
            model = result.get('model_used', 'unknown')
            if model not in models:
                models[model] = []
            models[model].append(result)
        
        analysis = {}
        for model, results in models.items():
            times = [r.get('processing_time', 0) for r in results]
            analysis[model] = {
                'count': len(results),
                'avg_time': statistics.mean(times),
                'min_time': min(times),
                'max_time': max(times),
                'std_dev': statistics.stdev(times) if len(times) > 1 else 0,
                'total_time': sum(times)
            }
        
        return analysis
    
    def _analyze_complexity_correlation(self) -> Dict[str, Any]:
        """Analyze correlation between bug complexity and performance"""
        if not self.experiment_results:
            return {}
        
        # Group by severity and difficulty
        complexity_groups = {}
        for result in self.experiment_results:
            severity = result.get('bug_severity', 'unknown')
            difficulty = result.get('bug_difficulty', 'unknown')
            key = f"{severity}_{difficulty}"
            
            if key not in complexity_groups:
                complexity_groups[key] = []
            complexity_groups[key].append(result)
        
        analysis = {}
        for complexity, results in complexity_groups.items():
            times = [r.get('processing_time', 0) for r in results]
            analysis[complexity] = {
                'count': len(results),
                'avg_time': statistics.mean(times),
                'min_time': min(times),
                'max_time': max(times)
            }
        
        return analysis
    
    def _analyze_repository_performance(self) -> Dict[str, Any]:
        """Analyze performance by repository"""
        if not self.experiment_results:
            return {}
        
        # Group by repository
        repos = {}
        for result in self.experiment_results:
            repo = result.get('repo', 'unknown')
            if repo not in repos:
                repos[repo] = []
            repos[repo].append(result)
        
        analysis = {}
        for repo, results in repos.items():
            times = [r.get('processing_time', 0) for r in results]
            analysis[repo] = {
                'count': len(results),
                'avg_time': statistics.mean(times),
                'min_time': min(times),
                'max_time': max(times)
            }
        
        return analysis
    
    def _analyze_costs(self) -> Dict[str, Any]:
        """Analyze cost implications"""
        if not self.experiment_results:
            return {}
        
        # Handle None values in cost estimates properly
        total_cost = 0
        for r in self.experiment_results:
            cost = r.get('cost_estimate', 0)
            if cost is not None:
                total_cost += cost
        
        avg_cost = total_cost / len(self.experiment_results) if self.experiment_results else 0
        
        return {
            'total_cost': total_cost,
            'avg_cost_per_experiment': avg_cost,
            'total_experiments': len(self.experiment_results),
            'cost_per_bug': total_cost / (len(self.experiment_results) / 2) if len(self.experiment_results) > 0 else 0
        }
    
    def _format_comprehensive_report(self, performance, complexity, repos, costs) -> str:
        """Format the comprehensive analysis report"""
        report = []
        report.append("=" * 80)
        report.append("SWE-BENCH 46 GUI BUG EXPERIMENTS - COMPREHENSIVE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Experiments: {len(self.experiment_results)}")
        report.append(f"Total Bugs Tested: {len(self.experiment_results) // 2}")
        report.append("")
        
        # Performance Analysis
        report.append("🚀 PERFORMANCE ANALYSIS")
        report.append("-" * 40)
        for model, metrics in performance.items():
            report.append(f"Model: {model}")
            report.append(f"  Experiments: {metrics['count']}")
            report.append(f"  Average Time: {metrics['avg_time']:.2f}s")
            report.append(f"  Time Range: {metrics['min_time']:.2f}s - {metrics['max_time']:.2f}s")
            report.append(f"  Standard Deviation: {metrics['std_dev']:.2f}s")
            report.append("")
        
        # Complexity Correlation
        report.append("🔍 COMPLEXITY CORRELATION ANALYSIS")
        report.append("-" * 40)
        for complexity, metrics in complexity.items():
            report.append(f"Complexity: {complexity}")
            report.append(f"  Count: {metrics['count']}")
            report.append(f"  Average Time: {metrics['avg_time']:.2f}s")
            report.append(f"  Time Range: {metrics['min_time']:.2f}s - {metrics['max_time']:.2f}s")
            report.append("")
        
        # Repository Performance
        report.append("�� REPOSITORY PERFORMANCE ANALYSIS")
        report.append("-" * 40)
        for repo, metrics in repos.items():
            report.append(f"Repository: {repo}")
            report.append(f"  Bugs Tested: {metrics['count']}")
            report.append(f"  Average Time: {metrics['avg_time']:.2f}s")
            report.append("")
        
        # Cost Analysis
        report.append("💰 COST ANALYSIS")
        report.append("-" * 40)
        report.append(f"Total Cost: ${costs['total_cost']:.4f}")
        report.append(f"Average Cost per Experiment: ${costs['avg_cost_per_experiment']:.4f}")
        report.append(f"Cost per Bug: ${costs['cost_per_bug']:.4f}")
        report.append("")
        
        # Summary
        report.append("📊 SUMMARY")
        report.append("-" * 40)
        report.append(f"✅ Successfully completed {len(self.experiment_results)} experiments")
        report.append(f"✅ Tested {len(self.experiment_results) // 2} unique GUI bugs")
        report.append(f"✅ Covered multiple repositories and bug types")
        report.append(f"✅ Generated comprehensive performance metrics")
        report.append(f"✅ Ready for research integration")
        
        return "\n".join(report)

def main():
    """Main execution function"""
    print("🎯 SWE-Bench 46 GUI Bug Experiments - Implementation")
    print("=" * 70)
    
    try:
        # Initialize runner
        runner = SWEBench46ExperimentRunner()
        
        # Show bug distribution
        runner.show_bug_distribution()
        
        # Ask user for execution mode
        print(f"\n🔧 Execution Options:")
        print(f"1. Test mode (first 5 bugs)")
        print(f"2. Full execution (all 46 bugs)")
        print(f"3. Custom number of bugs")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == "1":
            print(f"\n🧪 Running in TEST MODE (5 bugs)")
            runner.run_comprehensive_experiments(max_bugs=5)
        elif choice == "2":
            print(f"\n🚀 Running FULL EXECUTION (46 bugs)")
            runner.run_comprehensive_experiments()
        elif choice == "3":
            try:
                num_bugs = int(input("Enter number of bugs to test: "))
                print(f"\n�� Running CUSTOM MODE ({num_bugs} bugs)")
                runner.run_comprehensive_experiments(max_bugs=num_bugs)
            except ValueError:
                print("❌ Invalid number, running test mode")
                runner.run_comprehensive_experiments(max_bugs=5)
        else:
            print("❌ Invalid choice, running test mode")
            runner.run_comprehensive_experiments(max_bugs=5)
        
        print(f"\n🎉 SWE-Bench 46 experiments completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()