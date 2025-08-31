#!/usr/bin/env python
"""
Enhanced GUI Bug Repair LLM Experiment Framework
===============================================

This enhanced framework integrates advanced bug scenarios and supports
multiple LLM models for comprehensive testing and comparison.
"""

import os
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dotenv import load_dotenv

# Import our LLM client
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_client import LLMClient, LLMConfig

load_dotenv()

@dataclass
class EnhancedExperimentResult:
    """Enhanced experiment result with additional metrics"""
    experiment_id: str
    timestamp: str
    bug_description: str
    modality: str  # text_only, visual, multimodal
    prompt_template: str
    llm_response: str
    evaluation_metrics: Dict[str, Any]
    processing_time: float
    model_used: str
    temperature: float
    bug_category: str
    bug_severity: str
    bug_difficulty: str
    fix_priority: str
    cost_estimate: Optional[float] = None

@dataclass
class AdvancedBugScenario:
    """Advanced bug scenario with comprehensive details"""
    bug_id: str
    title: str
    description: str
    expected_solution: str
    bug_category: str
    severity: str
    difficulty: str
    ui_context: Optional[str] = None
    code_snippet: Optional[str] = None
    security_implications: Optional[List[str]] = None
    performance_impact: Optional[str] = None
    browser_specific: Optional[List[str]] = None
    mobile_affected: bool = False
    accessibility_impact: Optional[str] = None
    testing_scenarios: Optional[List[str]] = None
    fix_priority: str = "medium"

class EnhancedExperimentFramework:
    def __init__(self, config_path: str = "configs/config.json"):
        """Initialize the enhanced experiment framework"""
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize LLM clients for different models
        self.llm_clients = {}
        self._initialize_llm_clients()
        
        # Create results directory
        self.results_dir = Path("runs/enhanced_experiments")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load advanced scenarios
        self.advanced_scenarios = self._load_advanced_scenarios()
        
        # Define enhanced prompt templates
        self.prompt_templates = self._create_enhanced_prompts()
    
    def _initialize_llm_clients(self):
        """Initialize LLM clients for different models"""
        # Primary model (from config)
        primary_config = LLMConfig(
            provider=self.config["llm"]["provider"],
            model=self.config["llm"]["model"],
            api_key_env=self.config["llm"]["api_key_env"],
            temperature=self.config["llm"]["temperature"]
        )
        self.llm_clients["primary"] = LLMClient(primary_config)
        
        # Try to initialize additional models if available
        try:
            # Claude (if API key available)
            if os.getenv("ANTHROPIC_API_KEY"):
                claude_config = LLMConfig(
                    provider="anthropic",
                    model="claude-3-5-sonnet-20241022",
                    api_key_env="ANTHROPIC_API_KEY",
                    temperature=0.1
                )
                self.llm_clients["claude"] = LLMClient(claude_config)
                print("✓ Claude client initialized")
        except Exception as e:
            print(f"⚠ Could not initialize Claude client: {e}")
        
        # GPT-4 (if different from primary)
        if self.config["llm"]["model"] != "gpt-4":
            try:
                gpt4_config = LLMConfig(
                    provider="openai",
                    model="gpt-4",
                    api_key_env="OPENAI_API_KEY",
                    temperature=0.1
                )
                self.llm_clients["gpt4"] = LLMClient(gpt4_config)
                print("✓ GPT-4 client initialized")
            except Exception as e:
                print(f"⚠ Could not initialize GPT-4 client: {e}")
    
    def _load_advanced_scenarios(self) -> List[AdvancedBugScenario]:
        """Load advanced bug scenarios from JSON file"""
        scenarios_path = Path("data/processed/advanced_scenarios.json")
        if not scenarios_path.exists():
            print("⚠ Advanced scenarios file not found. Run advanced_scenarios.py first.")
            return []
        
        with open(scenarios_path, 'r') as f:
            scenarios_data = json.load(f)
        
        scenarios = []
        for data in scenarios_data:
            scenario = AdvancedBugScenario(
                bug_id=data["bug_id"],
                title=data["title"],
                description=data["description"],
                expected_solution=data["expected_solution"],
                bug_category=data["bug_category"],
                severity=data["severity"],
                difficulty=data["difficulty"],
                ui_context=data.get("ui_context"),
                code_snippet=data.get("code_snippet"),
                security_implications=data.get("security_implications"),
                performance_impact=data.get("performance_impact"),
                browser_specific=data.get("browser_specific"),
                mobile_affected=data.get("mobile_affected", False),
                accessibility_impact=data.get("accessibility_impact"),
                testing_scenarios=data.get("testing_scenarios"),
                fix_priority=data.get("fix_priority", "medium")
            )
            scenarios.append(scenario)
        
        print(f"✓ Loaded {len(scenarios)} advanced scenarios")
        return scenarios
    
    def _create_enhanced_prompts(self) -> Dict[str, str]:
        """Create enhanced prompt templates for advanced scenarios"""
        return {
            "text_only": """You are a senior frontend engineer specializing in bug fixing and security. Given this bug report:

Bug Report: {bug_description}
Bug Category: {bug_category}
Severity: {severity}
Difficulty: {difficulty}

Task:
1. Identify the likely root cause in a React/Vue/Vanilla JS web application
2. Propose a minimal, safe fix that addresses the specific issue type
3. Explain how your solution addresses the user-visible issue
4. Note any security, performance, or accessibility improvements needed
5. Provide specific testing steps to verify the fix

Provide your response in this format:
ROOT CAUSE: [brief explanation]
SOLUTION: [code fix or detailed steps]
EXPLANATION: [how this fixes the issue]
SECURITY/PERFORMANCE: [specific improvements for this bug type]
TESTING: [steps to verify the fix works]
ACCESSIBILITY: [any accessibility improvements]""",

            "visual": """You are a senior frontend engineer. Given this bug report and visual context:

Bug Report: {bug_description}
Bug Category: {bug_category}
Severity: {severity}
Visual Context: {visual_context}

Task:
1. Analyze the visual elements and layout issues
2. Identify the root cause considering UI/UX principles and the specific bug type
3. Propose a fix that addresses both functionality and visual design
4. Ensure accessibility compliance and cross-browser compatibility
5. Provide visual testing steps

Provide your response in this format:
VISUAL ANALYSIS: [what you observe from the visual context]
ROOT CAUSE: [brief explanation]
SOLUTION: [CSS/HTML/JS fix]
EXPLANATION: [how this fixes the issue]
VISUAL TESTING: [steps to verify visual fix]
ACCESSIBILITY: [visual accessibility improvements]""",

            "multimodal": """You are a senior frontend engineer. Given this comprehensive bug report:

Bug Report: {bug_description}
Bug Category: {bug_category}
Severity: {severity}
Difficulty: {difficulty}
Visual Context: {visual_context}
Code Context: {code_context}
UI Events: {ui_events}

Task:
1. Analyze all available modalities (text, visual, code, interactions)
2. Identify the root cause considering the complete context and bug type
3. Propose a comprehensive fix that addresses all aspects
4. Ensure the solution is production-ready, secure, and accessible
5. Provide comprehensive testing and validation steps

Provide your response in this format:
CONTEXT ANALYSIS: [summary of all modalities]
ROOT CAUSE: [brief explanation]
SOLUTION: [comprehensive fix]
EXPLANATION: [how this fixes the issue]
SECURITY/PERFORMANCE: [specific improvements for this bug type]
TESTING: [comprehensive testing steps]
ACCESSIBILITY: [accessibility improvements]"""
        }
    
    def run_advanced_experiment(self, scenario: AdvancedBugScenario, modality: str, model_name: str = "primary") -> EnhancedExperimentResult:
        """Run an experiment with an advanced bug scenario"""
        if model_name not in self.llm_clients:
            raise ValueError(f"Model {model_name} not available. Available: {list(self.llm_clients.keys())}")
        
        start_time = time.time()
        
        # Select appropriate prompt template
        prompt_template = self.prompt_templates[modality]
        
        # Prepare context based on modality
        context_vars = {
            "bug_description": scenario.description,
            "bug_category": scenario.bug_category,
            "severity": scenario.severity,
            "difficulty": scenario.difficulty,
            "visual_context": scenario.ui_context or "No visual context available",
            "code_context": scenario.code_snippet or "No code context available",
            "ui_events": "No UI events available"
        }
        
        # Format prompt
        prompt = prompt_template.format(**context_vars)
        
        # Get LLM response
        llm_client = self.llm_clients[model_name]
        llm_response = llm_client.complete(prompt)
        
        processing_time = time.time() - start_time
        
        # Create enhanced experiment result
        result = EnhancedExperimentResult(
            experiment_id=f"{scenario.bug_id}_{modality}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            bug_description=scenario.description,
            modality=modality,
            prompt_template=prompt_template,
            llm_response=llm_response,
            evaluation_metrics=self._evaluate_advanced_response(llm_response, scenario),
            processing_time=processing_time,
            model_used=model_name,
            temperature=llm_client.cfg.temperature,
            bug_category=scenario.bug_category,
            bug_severity=scenario.severity,
            bug_difficulty=scenario.difficulty,
            fix_priority=scenario.fix_priority
        )
        
        return result
    
    def _evaluate_advanced_response(self, response: str, scenario: AdvancedBugScenario) -> Dict[str, Any]:
        """Evaluate the quality of the LLM response for advanced scenarios"""
        metrics = {
            "response_length": len(response),
            "contains_solution": "SOLUTION" in response.upper(),
            "contains_explanation": "EXPLANATION" in response.upper(),
            "contains_testing": "TESTING" in response.upper(),
            "contains_root_cause": "ROOT CAUSE" in response.upper(),
            "structured_response": self._check_enhanced_format(response),
            "solution_confidence": self._assess_advanced_solution_confidence(response, scenario),
            "security_awareness": self._assess_security_awareness(response, scenario),
            "performance_awareness": self._assess_performance_awareness(response, scenario),
            "browser_compatibility": self._assess_browser_compatibility(response, scenario)
        }
        return metrics
    
    def _check_enhanced_format(self, response: str) -> bool:
        """Check if response follows the enhanced structured format"""
        required_sections = ["SOLUTION", "EXPLANATION"]
        optional_sections = ["TESTING", "SECURITY", "PERFORMANCE", "ACCESSIBILITY"]
        
        required_present = all(section in response.upper() for section in required_sections)
        optional_present = any(section in response.upper() for section in optional_sections)
        
        return required_present and optional_present
    
    def _assess_advanced_solution_confidence(self, response: str, scenario: AdvancedBugScenario) -> float:
        """Assess confidence in the proposed solution for advanced scenarios"""
        confidence_score = 0.5  # Base score
        
        # Positive indicators
        if scenario.bug_category.lower() in response.lower():
            confidence_score += 0.2
        if scenario.expected_solution.lower() in response.lower():
            confidence_score += 0.2
        if "code" in response.lower() or "css" in response.lower():
            confidence_score += 0.1
        
        # Category-specific indicators
        if scenario.bug_category == "security" and any(term in response.lower() for term in ["sanitize", "validate", "csrf", "xss", "secure"]):
            confidence_score += 0.2
        elif scenario.bug_category == "performance" and any(term in response.lower() for term in ["memory", "leak", "optimize", "bundle", "render"]):
            confidence_score += 0.2
        elif scenario.bug_category == "cross_browser" and any(term in response.lower() for term in ["prefix", "fallback", "polyfill", "browser"]):
            confidence_score += 0.2
        
        # Negative indicators
        if "i don't know" in response.lower() or "cannot determine" in response.lower():
            confidence_score -= 0.3
        
        return max(0.0, min(1.0, confidence_score))
    
    def _assess_security_awareness(self, response: str, scenario: AdvancedBugScenario) -> float:
        """Assess security awareness in the response"""
        if scenario.bug_category != "security":
            return 0.5  # Neutral for non-security bugs
        
        security_terms = ["sanitize", "validate", "csrf", "xss", "secure", "authentication", "authorization", "input", "output"]
        security_mentions = sum(1 for term in security_terms if term in response.lower())
        
        return min(1.0, security_mentions / 3.0)  # Normalize to 0-1
    
    def _assess_performance_awareness(self, response: str, scenario: AdvancedBugScenario) -> float:
        """Assess performance awareness in the response"""
        if scenario.bug_category != "performance":
            return 0.5  # Neutral for non-performance bugs
        
        perf_terms = ["memory", "leak", "optimize", "bundle", "render", "performance", "efficient", "fast", "slow"]
        perf_mentions = sum(1 for term in perf_terms if term in response.lower())
        
        return min(1.0, perf_mentions / 3.0)  # Normalize to 0-1
    
    def _assess_browser_compatibility(self, response: str, scenario: AdvancedBugScenario) -> float:
        """Assess browser compatibility awareness in the response"""
        if scenario.bug_category != "cross_browser":
            return 0.5  # Neutral for non-browser bugs
        
        browser_terms = ["prefix", "fallback", "polyfill", "browser", "safari", "firefox", "chrome", "edge", "ie"]
        browser_mentions = sum(1 for term in browser_terms if term in response.lower())
        
        return min(1.0, browser_mentions / 3.0)  # Normalize to 0-1
    
    def run_model_comparison_experiment(self, scenario: AdvancedBugScenario, modality: str) -> List[EnhancedExperimentResult]:
        """Run the same experiment with multiple models for comparison"""
        results = []
        
        for model_name in self.llm_clients.keys():
            try:
                result = self.run_advanced_experiment(scenario, modality, model_name)
                results.append(result)
                print(f"✓ Completed {model_name} experiment for {scenario.bug_id}")
            except Exception as e:
                print(f"✗ Failed {model_name} experiment for {scenario.bug_id}: {e}")
        
        return results
    
    def run_comprehensive_test_suite(self, modalities: List[str] = None, models: List[str] = None) -> List[EnhancedExperimentResult]:
        """Run comprehensive tests across scenarios, modalities, and models"""
        if modalities is None:
            modalities = ["text_only", "multimodal"]
        if models is None:
            models = list(self.llm_clients.keys())
        
        all_results = []
        
        # Test each scenario with each modality and model
        for scenario in self.advanced_scenarios:
            for modality in modalities:
                for model in models:
                    try:
                        result = self.run_advanced_experiment(scenario, modality, model)
                        all_results.append(result)
                        print(f"✓ {scenario.bug_id} - {modality} - {model}")
                    except Exception as e:
                        print(f"✗ {scenario.bug_id} - {modality} - {model}: {e}")
        
        return all_results
    
    def save_results(self, results: List[EnhancedExperimentResult], filename: str = None):
        """Save experiment results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_experiments_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        # Convert results to dictionaries
        results_data = []
        for result in results:
            result_dict = asdict(result)
            results_data.append(result_dict)
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"✓ Saved {len(results)} results to {filepath}")
        return filepath
    
    def generate_experiment_report(self, results: List[EnhancedExperimentResult]) -> str:
        """Generate a comprehensive experiment report"""
        if not results:
            return "No results to analyze"
        
        # Group results by model and category
        model_results = {}
        category_results = {}
        
        for result in results:
            # Group by model
            if result.model_used not in model_results:
                model_results[result.model_used] = []
            model_results[result.model_used].append(result)
            
            # Group by bug category
            if result.bug_category not in category_results:
                category_results[result.bug_category] = []
            category_results[result.bug_category].append(result)
        
        # Calculate metrics
        report = []
        report.append("=" * 60)
        report.append("ENHANCED EXPERIMENT REPORT")
        report.append("=" * 60)
        report.append(f"Total Experiments: {len(results)}")
        report.append(f"Models Tested: {len(model_results)}")
        report.append(f"Bug Categories: {len(category_results)}")
        report.append("")
        
        # Model performance comparison
        report.append("MODEL PERFORMANCE COMPARISON:")
        report.append("-" * 30)
        for model, model_res in model_results.items():
            avg_confidence = sum(r.evaluation_metrics["solution_confidence"] for r in model_res) / len(model_res)
            avg_time = sum(r.processing_time for r in model_res) / len(model_res)
            report.append(f"{model.upper()}:")
            report.append(f"  - Avg Confidence: {avg_confidence:.3f}")
            report.append(f"  - Avg Time: {avg_time:.2f}s")
            report.append(f"  - Experiments: {len(model_res)}")
            report.append("")
        
        # Category performance
        report.append("BUG CATEGORY PERFORMANCE:")
        report.append("-" * 30)
        for category, cat_res in category_results.items():
            avg_confidence = sum(r.evaluation_metrics["solution_confidence"] for r in cat_res) / len(cat_res)
            report.append(f"{category.upper()}: {avg_confidence:.3f} avg confidence")
        
        return "\n".join(report)

def main():
    """Main function to demonstrate the enhanced framework"""
    framework = EnhancedExperimentFramework()
    
    print("Enhanced Experiment Framework Initialized")
    print(f"Available models: {list(framework.llm_clients.keys())}")
    print(f"Loaded {len(framework.advanced_scenarios)} advanced scenarios")
    
    # Example: Run a single experiment
    if framework.advanced_scenarios:
        scenario = framework.advanced_scenarios[0]  # First scenario
        print(f"\nRunning single experiment: {scenario.title}")
        result = framework.run_advanced_experiment(scenario, "text_only", "primary")
        print(f"Result: {result.evaluation_metrics['solution_confidence']:.3f} confidence")
    
    return framework

if __name__ == "__main__":
    main()
