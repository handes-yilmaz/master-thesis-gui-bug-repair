#!/usr/bin/env python
"""
Prompt Optimization Framework for GUI Bug Repair
===============================================

This module tests different prompting strategies to find optimal approaches
for different types of GUI bugs.
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import json
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_experiment_framework import EnhancedExperimentFramework, AdvancedBugScenario

class PromptOptimizationFramework:
    """Framework for testing and optimizing different prompting strategies"""
    
    def __init__(self):
        self.framework = EnhancedExperimentFramework()
        self.prompt_strategies = self._define_prompt_strategies()
    
    def _define_prompt_strategies(self) -> Dict[str, Dict[str, str]]:
        """Define different prompting strategies to test"""
        return {
            "few_shot": {
                "text_only": """You are a senior frontend engineer. Given this bug report and examples:

Bug Report: {bug_description}
Bug Category: {bug_category}
Severity: {severity}

EXAMPLES:
1. Button not responding to clicks
   SOLUTION: Check event handlers, CSS pointer-events, z-index layering
   EXPLANATION: Event handlers may be missing or CSS properties blocking interaction

2. Layout breaks on mobile
   SOLUTION: Add media queries and mobile-specific CSS
   EXPLANATION: Responsive design requires mobile-first approach

Task: Fix the current bug using the examples as guidance.
Provide your response in this format:
ROOT CAUSE: [brief explanation]
SOLUTION: [code fix or detailed steps]
EXPLANATION: [how this fixes the issue]
TESTING: [steps to verify the fix]""",

                "multimodal": """You are a senior frontend engineer. Given this bug report and examples:

Bug Report: {bug_description}
Bug Category: {bug_category}
Severity: {severity}
Visual Context: {visual_context}
Code Context: {code_context}

EXAMPLES:
1. XSS vulnerability in user input
   SOLUTION: Use DOMPurify library, implement input sanitization
   EXPLANATION: Never trust user input, always sanitize before rendering

2. Memory leak in event listeners
   SOLUTION: Clean up event listeners in useEffect cleanup function
   EXPLANATION: Prevent memory accumulation by removing listeners

Task: Fix the current bug using the examples as guidance.
Provide your response in this format:
ROOT CAUSE: [brief explanation]
SOLUTION: [comprehensive fix]
EXPLANATION: [how this fixes the issue]
TESTING: [steps to verify the fix]"""
            },
            
            "chain_of_thought": {
                "text_only": """You are a senior frontend engineer. Given this bug report:

Bug Report: {bug_description}
Bug Category: {bug_category}
Severity: {severity}

Task: Think through this step by step:
1. First, analyze what type of bug this is
2. Then, identify the likely root cause
3. Next, consider the best approach to fix it
4. Finally, implement the solution

Provide your response in this format:
ANALYSIS: [step-by-step reasoning about the bug]
ROOT CAUSE: [brief explanation]
SOLUTION: [code fix or detailed steps]
EXPLANATION: [how this fixes the issue]
TESTING: [steps to verify the fix]""",

                "multimodal": """You are a senior frontend engineer. Given this bug report:

Bug Report: {bug_description}
Bug Category: {bug_category}
Severity: {severity}
Visual Context: {visual_context}
Code Context: {code_context}

Task: Think through this step by step:
1. First, analyze the visual and code context
2. Then, identify what type of bug this is
3. Next, determine the root cause
4. Finally, implement the best solution

Provide your response in this format:
CONTEXT ANALYSIS: [what you observe from visual and code]
BUG ANALYSIS: [step-by-step reasoning about the bug]
ROOT CAUSE: [brief explanation]
SOLUTION: [comprehensive fix]
EXPLANATION: [how this fixes the issue]
TESTING: [steps to verify the fix]"""
            },
            
            "technical_detailed": {
                "text_only": """You are a senior frontend engineer specializing in bug fixing. Given this bug report:

Bug Report: {bug_description}
Bug Category: {bug_category}
Severity: {severity}

Technical Requirements:
- Provide specific code examples
- Include error handling considerations
- Address security implications if applicable
- Consider performance impact
- Ensure accessibility compliance

Task: Provide a comprehensive technical solution.
Format your response as:
ROOT CAUSE: [detailed technical explanation]
SOLUTION: [specific code with error handling]
EXPLANATION: [technical details of how this works]
SECURITY: [security considerations if applicable]
PERFORMANCE: [performance implications]
TESTING: [comprehensive testing steps]""",

                "multimodal": """You are a senior frontend engineer specializing in bug fixing. Given this bug report:

Bug Report: {bug_description}
Bug Category: {bug_category}
Severity: {severity}
Visual Context: {visual_context}
Code Context: {code_context}

Technical Requirements:
- Analyze all available modalities comprehensively
- Provide specific code examples
- Include error handling considerations
- Address security implications if applicable
- Consider performance impact
- Ensure accessibility compliance

Task: Provide a comprehensive technical solution.
Format your response as:
CONTEXT ANALYSIS: [comprehensive analysis of all modalities]
ROOT CAUSE: [detailed technical explanation]
SOLUTION: [specific code with error handling]
EXPLANATION: [technical details of how this works]
SECURITY: [security considerations if applicable]
PERFORMANCE: [performance implications]
TESTING: [comprehensive testing steps]"""
            },
            
            "user_friendly": {
                "text_only": """You are a helpful frontend developer. A user has reported this bug:

Bug Report: {bug_description}
Bug Category: {bug_category}
Severity: {severity}

Task: Help the user understand and fix this issue in simple terms.
- Explain what's happening in plain language
- Provide a clear, step-by-step solution
- Make sure the explanation is easy to follow

Format your response as:
WHAT'S HAPPENING: [simple explanation of the problem]
WHY THIS HAPPENS: [easy-to-understand root cause]
HOW TO FIX IT: [step-by-step solution]
HOW TO TEST: [simple testing steps]
PREVENTION: [how to avoid this in the future]""",

                "multimodal": """You are a helpful frontend developer. A user has reported this bug:

Bug Report: {bug_description}
Bug Category: {bug_category}
Severity: {severity}
Visual Context: {visual_context}
Code Context: {code_context}

Task: Help the user understand and fix this issue in simple terms.
- Look at the visual and code context
- Explain what's happening in plain language
- Provide a clear, step-by-step solution
- Make sure the explanation is easy to follow

Format your response as:
WHAT I SEE: [simple description of the visual and code context]
WHAT'S HAPPENING: [simple explanation of the problem]
WHY THIS HAPPENS: [easy-to-understand root cause]
HOW TO FIX IT: [step-by-step solution]
HOW TO TEST: [simple testing steps]
PREVENTION: [how to avoid this in the future]"""
            }
        }
    
    def run_prompt_strategy_test(self, strategy_name: str, modality: str, 
                                scenario: AdvancedBugScenario, model: str = "primary") -> Dict[str, Any]:
        """Test a specific prompt strategy on a bug scenario"""
        if strategy_name not in self.prompt_strategies:
            raise ValueError(f"Strategy {strategy_name} not found")
        
        if modality not in self.prompt_strategies[strategy_name]:
            raise ValueError(f"Modality {modality} not supported for strategy {strategy_name}")
        
        # Get the prompt template
        prompt_template = self.prompt_strategies[strategy_name][modality]
        
        # Prepare context variables
        context_vars = {
            "bug_description": scenario.description,
            "bug_category": scenario.bug_category,
            "severity": scenario.severity,
            "visual_context": scenario.ui_context or "No visual context available",
            "code_context": scenario.code_snippet or "No code context available"
        }
        
        # Format the prompt
        prompt = prompt_template.format(**context_vars)
        
        # Run the experiment
        start_time = time.time()
        try:
            result = self.framework.run_advanced_experiment(scenario, modality, model)
            processing_time = time.time() - start_time
            
            # Add strategy information
            strategy_result = {
                "strategy": strategy_name,
                "modality": modality,
                "scenario_id": scenario.bug_id,
                "bug_category": scenario.bug_category,
                "severity": scenario.severity,
                "prompt_template": prompt_template,
                "llm_response": result.llm_response,
                "evaluation_metrics": result.evaluation_metrics,
                "processing_time": processing_time,
                "model_used": model,
                "success": True
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            strategy_result = {
                "strategy": strategy_name,
                "modality": modality,
                "scenario_id": scenario.bug_id,
                "bug_category": scenario.bug_category,
                "severity": scenario.severity,
                "prompt_template": prompt_template,
                "error": str(e),
                "processing_time": processing_time,
                "model_used": model,
                "success": False
            }
        
        return strategy_result
    
    def run_comprehensive_prompt_test(self, scenarios: List[AdvancedBugScenario] = None, 
                                    strategies: List[str] = None, modalities: List[str] = None,
                                    models: List[str] = None) -> List[Dict[str, Any]]:
        """Run comprehensive prompt strategy testing"""
        if scenarios is None:
            scenarios = self.framework.advanced_scenarios[:5]  # Test first 5 scenarios
        
        if strategies is None:
            strategies = list(self.prompt_strategies.keys())
        
        if modalities is None:
            modalities = ["text_only", "multimodal"]
        
        if models is None:
            models = list(self.framework.llm_clients.keys())
        
        all_results = []
        
        print(f"🔬 Running comprehensive prompt optimization test...")
        print(f"Scenarios: {len(scenarios)}, Strategies: {len(strategies)}, Modalities: {len(modalities)}, Models: {len(models)}")
        
        total_experiments = len(scenarios) * len(strategies) * len(modalities) * len(models)
        current_experiment = 0
        
        for scenario in scenarios:
            print(f"\n📋 Testing scenario: {scenario.title}")
            print(f"Category: {scenario.bug_category}, Severity: {scenario.severity}")
            
            for strategy in strategies:
                print(f"  🎯 Strategy: {strategy}")
                
                for modality in modalities:
                    if modality in self.prompt_strategies[strategy]:
                        print(f"    📱 Modality: {modality}")
                        
                        for model in models:
                            current_experiment += 1
                            print(f"      🤖 Model: {model} ({current_experiment}/{total_experiments})")
                            
                            try:
                                result = self.run_prompt_strategy_test(strategy, modality, scenario, model)
                                all_results.append(result)
                                
                                if result["success"]:
                                    confidence = result["evaluation_metrics"]["solution_confidence"]
                                    print(f"        ✅ Success - Confidence: {confidence:.3f}")
                                else:
                                    print(f"        ❌ Failed - {result['error']}")
                                
                            except Exception as e:
                                print(f"        ❌ Error: {e}")
                                all_results.append({
                                    "strategy": strategy,
                                    "modality": modality,
                                    "scenario_id": scenario.bug_id,
                                    "bug_category": scenario.bug_category,
                                    "severity": scenario.severity,
                                    "error": str(e),
                                    "success": False
                                })
        
        return all_results
    
    def analyze_prompt_performance(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the performance of different prompt strategies"""
        if not results:
            return {}
        
        analysis = {
            "total_experiments": len(results),
            "successful_experiments": len([r for r in results if r.get("success", False)]),
            "strategy_performance": {},
            "modality_performance": {},
            "category_performance": {},
            "model_performance": {}
        }
        
        # Strategy performance
        for result in results:
            strategy = result["strategy"]
            if strategy not in analysis["strategy_performance"]:
                analysis["strategy_performance"][strategy] = {
                    "total": 0,
                    "successful": 0,
                    "avg_confidence": 0,
                    "avg_time": 0,
                    "confidences": [],
                    "times": []
                }
            
            analysis["strategy_performance"][strategy]["total"] += 1
            
            if result.get("success", False):
                analysis["strategy_performance"][strategy]["successful"] += 1
                
                if "evaluation_metrics" in result:
                    confidence = result["evaluation_metrics"].get("solution_confidence", 0)
                    analysis["strategy_performance"][strategy]["confidences"].append(confidence)
                
                if "processing_time" in result:
                    time_taken = result["processing_time"]
                    analysis["strategy_performance"][strategy]["times"].append(time_taken)
        
        # Calculate averages
        for strategy, data in analysis["strategy_performance"].items():
            if data["confidences"]:
                data["avg_confidence"] = sum(data["confidences"]) / len(data["confidences"])
            if data["times"]:
                data["avg_time"] = sum(data["times"]) / len(data["times"])
        
        # Similar analysis for modalities, categories, and models
        # (implementation similar to strategy analysis)
        
        return analysis
    
    def save_results(self, results: List[Dict[str, Any]], filename: str = None):
        """Save prompt optimization results to file"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"prompt_optimization_results_{timestamp}.json"
        
        # Ensure results directory exists
        results_dir = Path("runs/prompt_optimization")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Saved {len(results)} results to {filepath}")
        return filepath
    
    def generate_optimization_report(self, results: List[Dict[str, Any]], analysis: Dict[str, Any]) -> str:
        """Generate a comprehensive optimization report"""
        report = []
        report.append("=" * 60)
        report.append("PROMPT OPTIMIZATION RESULTS REPORT")
        report.append("=" * 60)
        report.append(f"Total Experiments: {analysis.get('total_experiments', 0)}")
        report.append(f"Successful Experiments: {analysis.get('successful_experiments', 0)}")
        report.append(f"Success Rate: {analysis.get('successful_experiments', 0) / max(analysis.get('total_experiments', 1), 1) * 100:.1f}%")
        report.append("")
        
        # Strategy performance
        if "strategy_performance" in analysis:
            report.append("STRATEGY PERFORMANCE:")
            report.append("-" * 30)
            for strategy, data in analysis["strategy_performance"].items():
                success_rate = data["successful"] / max(data["total"], 1) * 100
                avg_conf = data.get("avg_confidence", 0)
                avg_time = data.get("avg_time", 0)
                report.append(f"{strategy.upper()}:")
                report.append(f"  - Success Rate: {success_rate:.1f}%")
                report.append(f"  - Avg Confidence: {avg_conf:.3f}")
                report.append(f"  - Avg Time: {avg_time:.2f}s")
                report.append("")
        
        return "\n".join(report)

def main():
    """Main function to run prompt optimization experiments"""
    print("🎯 Prompt Optimization Framework for GUI Bug Repair")
    print("=" * 50)
    
    # Initialize framework
    optimizer = PromptOptimizationFramework()
    
    print(f"✓ Framework initialized with {len(optimizer.prompt_strategies)} strategies")
    print(f"✓ Available strategies: {list(optimizer.prompt_strategies.keys())}")
    
    # Run comprehensive test
    print("\n🚀 Starting comprehensive prompt optimization test...")
    results = optimizer.run_comprehensive_prompt_test()
    
    if results:
        # Analyze results
        print("\n📊 Analyzing results...")
        analysis = optimizer.analyze_prompt_performance(results)
        
        # Save results
        print("\n💾 Saving results...")
        filename = optimizer.save_results(results)
        
        # Generate report
        print("\n📋 Generating report...")
        report = optimizer.generate_optimization_report(results, analysis)
        print(report)
        
        # Save report
        report_file = Path("runs/prompt_optimization") / f"report_{filename.replace('.json', '.txt')}"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"✓ Report saved to: {report_file}")
        
        print(f"\n🎉 Prompt optimization completed!")
        print(f"Total experiments: {len(results)}")
        print(f"Results saved to: {filename}")
    else:
        print("\n❌ No results to analyze")

if __name__ == "__main__":
    main()
