#!/usr/bin/env python
"""
GUI Bug Repair LLM Experiment Framework
======================================

This framework provides a structured approach to testing LLM capabilities
for GUI bug repair across different scenarios and modalities.
"""

import os
import json
import time
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# Import our LLM client
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_client import LLMClient, LLMConfig

load_dotenv()

@dataclass
class ExperimentResult:
    """Represents the result of a single LLM experiment"""
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

@dataclass
class BugScenario:
    """Represents a bug scenario for testing"""
    bug_id: str
    title: str
    description: str
    expected_solution: str
    bug_category: str
    severity: str
    ui_context: Optional[str] = None
    code_snippet: Optional[str] = None
    screenshot_path: Optional[str] = None
    ocr_text: Optional[str] = None
    ui_events: Optional[str] = None

class ExperimentFramework:
    def __init__(self, config_path: str = "configs/config.json"):
        """Initialize the experiment framework"""
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize LLM client
        llm_config = LLMConfig(
            provider=self.config["llm"]["provider"],
            model=self.config["llm"]["model"],
            api_key_env=self.config["llm"]["api_key_env"],
            temperature=self.config["llm"]["temperature"]
        )
        self.llm_client = LLMClient(llm_config)
        
        # Create results directory
        self.results_dir = Path("runs/experiments")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Define prompt templates
        self.prompt_templates = {
            "text_only": """You are a senior frontend engineer specializing in bug fixing. Given this bug report:

Bug Report: {bug_description}

Task:
1. Identify the likely root cause in a React/Vue/Vanilla JS web application
2. Propose a minimal, safe fix
3. Explain how your solution addresses the user-visible issue
4. Note any accessibility improvements needed

Provide your response in this format:
ROOT CAUSE: [brief explanation]
SOLUTION: [code fix or detailed steps]
EXPLANATION: [how this fixes the issue]
ACCESSIBILITY: [any accessibility improvements]""",

            "visual": """You are a senior frontend engineer. Given this bug report and visual context:

Bug Report: {bug_description}
Visual Context: {visual_context}

Task:
1. Analyze the visual elements and layout issues
2. Identify the root cause considering UI/UX principles
3. Propose a fix that addresses both functionality and visual design
4. Ensure accessibility compliance

Provide your response in this format:
VISUAL ANALYSIS: [what you observe from the visual context]
ROOT CAUSE: [brief explanation]
SOLUTION: [CSS/HTML/JS fix]
EXPLANATION: [how this fixes the issue]
ACCESSIBILITY: [visual accessibility improvements]""",

            "multimodal": """You are a senior frontend engineer. Given this comprehensive bug report:

Bug Report: {bug_description}
Visual Context: {visual_context}
Code Context: {code_context}
UI Events: {ui_events}

Task:
1. Analyze all available modalities (text, visual, code, interactions)
2. Identify the root cause considering the complete context
3. Propose a comprehensive fix that addresses all aspects
4. Ensure the solution is production-ready and accessible

Provide your response in this format:
CONTEXT ANALYSIS: [summary of all modalities]
ROOT CAUSE: [brief explanation]
SOLUTION: [comprehensive fix]
EXPLANATION: [how this fixes the issue]
ACCESSIBILITY: [accessibility improvements]
TESTING: [suggested tests]"""
        }
    
    def create_sample_bug_scenarios(self) -> List[BugScenario]:
        """Create sample bug scenarios for testing"""
        scenarios = [
            BugScenario(
                bug_id="BUTTON_CLICK_01",
                title="Button not responding to clicks",
                description="The submit button on the login form doesn't respond when clicked. Users report the button appears disabled or unresponsive.",
                expected_solution="Check event handlers, CSS pointer-events, z-index layering",
                bug_category="interaction_event",
                severity="high",
                ui_context="Login form with submit button",
                code_snippet="""
function LoginForm() {
    const handleSubmit = (e) => {
        // Missing event.preventDefault() could be causing issues
        console.log('Form submitted');
    };
    
    return (
        <form onSubmit={handleSubmit}>
            <button type="submit">Login</button>
        </form>
    );
}"""
            ),
            
            BugScenario(
                bug_id="LAYOUT_BREAK_02", 
                title="Responsive layout breaks on mobile",
                description="The navigation menu doesn't collapse properly on mobile devices, causing horizontal overflow and making the site unusable.",
                expected_solution="Add proper media queries and mobile navigation handling",
                bug_category="visual_layout",
                severity="medium",
                ui_context="Navigation menu that should be collapsible",
                code_snippet="""
.nav-menu {
    display: flex;
    flex-direction: row;
    /* Missing media query for mobile */
}

.nav-item {
    padding: 10px;
    /* No mobile-specific styling */
}"""
            ),
            
            BugScenario(
                bug_id="ACCESSIBILITY_03",
                title="Missing ARIA labels on form inputs",
                description="Screen reader users cannot understand what each form field is for due to missing accessibility labels.",
                expected_solution="Add proper ARIA labels and form associations",
                bug_category="accessibility",
                severity="high",
                ui_context="Registration form with multiple inputs",
                code_snippet="""
<form>
    <input type="text" placeholder="Enter name" />
    <input type="email" placeholder="Email address" />
    <input type="password" placeholder="Password" />
    <!-- Missing labels and ARIA attributes -->
</form>"""
            ),
            
            BugScenario(
                bug_id="STATE_MANAGEMENT_04",
                title="Form state not persisting on page refresh",
                description="When users refresh the page during checkout, all form data is lost, causing frustration and potential cart abandonment.",
                expected_solution="Implement proper state management with localStorage or form persistence",
                bug_category="state_transition",
                severity="medium",
                ui_context="Checkout form that should persist data",
                ui_events="user fills form → page refresh → data lost"
            ),
            
            BugScenario(
                bug_id="COLOR_CONTRAST_05",
                title="Insufficient color contrast on error messages",
                description="Error messages displayed in light red text on white background make it difficult for users with visual impairments to read.",
                expected_solution="Improve color contrast ratios to meet WCAG guidelines",
                bug_category="visual_color_typography",
                severity="medium",
                ui_context="Error message display",
                code_snippet="""
.error-message {
    color: #ffcccc; /* Too light for good contrast */
    background-color: white;
}"""
            )
        ]
        return scenarios
    
    def run_experiment(self, scenario: BugScenario, modality: str) -> ExperimentResult:
        """Run a single experiment with the given scenario and modality"""
        start_time = time.time()
        
        # Select appropriate prompt template
        prompt_template = self.prompt_templates[modality]
        
        # Prepare context based on modality
        context_vars = {
            "bug_description": scenario.description,
            "visual_context": scenario.ocr_text or "No visual context available",
            "code_context": scenario.code_snippet or "No code context available",
            "ui_events": scenario.ui_events or "No UI events available"
        }
        
        # Format prompt
        prompt = prompt_template.format(**context_vars)
        
        # Get LLM response
        llm_response = self.llm_client.complete(prompt)
        
        processing_time = time.time() - start_time
        
        # Create experiment result
        result = ExperimentResult(
            experiment_id=f"{scenario.bug_id}_{modality}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            bug_description=scenario.description,
            modality=modality,
            prompt_template=prompt_template,
            llm_response=llm_response,
            evaluation_metrics=self._evaluate_response(llm_response, scenario),
            processing_time=processing_time,
            model_used=self.config["llm"]["model"],
            temperature=self.config["llm"]["temperature"]
        )
        
        return result
    
    def _evaluate_response(self, response: str, scenario: BugScenario) -> Dict[str, Any]:
        """Evaluate the quality of the LLM response"""
        metrics = {
            "response_length": len(response),
            "contains_solution": "SOLUTION" in response.upper(),
            "contains_explanation": "EXPLANATION" in response.upper(),
            "contains_accessibility": "ACCESSIBILITY" in response.upper(),
            "contains_root_cause": "ROOT CAUSE" in response.upper(),
            "structured_response": self._check_structured_format(response),
            "solution_confidence": self._assess_solution_confidence(response, scenario)
        }
        return metrics
    
    def _check_structured_format(self, response: str) -> bool:
        """Check if response follows the expected structured format"""
        sections = ["SOLUTION", "EXPLANATION", "ACCESSIBILITY"]
        return all(section in response.upper() for section in sections)
    
    def _assess_solution_confidence(self, response: str, scenario: BugScenario) -> float:
        """Assess confidence in the proposed solution"""
        confidence_score = 0.5  # Base score
        
        # Positive indicators
        if scenario.bug_category.lower() in response.lower():
            confidence_score += 0.2
        if "code" in response.lower() or "css" in response.lower():
            confidence_score += 0.2
        if scenario.expected_solution.lower() in response.lower():
            confidence_score += 0.1
        
        # Negative indicators
        if "i don't know" in response.lower() or "cannot determine" in response.lower():
            confidence_score -= 0.3
        
        return max(0.0, min(1.0, confidence_score))
    
    def run_experiment_suite(self, modalities: List[str] = None) -> List[ExperimentResult]:
        """Run experiments across all scenarios and modalities"""
        if modalities is None:
            modalities = ["text_only", "visual", "multimodal"]
        
        scenarios = self.create_sample_bug_scenarios()
        results = []
        
        print(f"Running experiment suite with {len(scenarios)} scenarios and {len(modalities)} modalities...")
        
        for i, scenario in enumerate(scenarios):
            print(f"\n--- Scenario {i+1}/{len(scenarios)}: {scenario.title} ---")
            
            for modality in modalities:
                print(f"  Testing {modality}...")
                try:
                    result = self.run_experiment(scenario, modality)
                    results.append(result)
                    print(f"    ✅ Completed (took {result.processing_time:.2f}s)")
                except Exception as e:
                    print(f"    ❌ Failed: {e}")
        
        return results
    
    def save_results(self, results: List[ExperimentResult], filename: str = None) -> str:
        """Save experiment results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experiment_results_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        # Convert results to JSON-serializable format
        results_data = [asdict(result) for result in results]
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Results saved to: {filepath}")
        return str(filepath)
    
    def generate_report(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Generate a comprehensive analysis report"""
        if not results:
            return {"error": "No results to analyze"}
        
        df = pd.DataFrame([asdict(result) for result in results])
        
        report = {
            "summary": {
                "total_experiments": len(results),
                "modalities_tested": df['modality'].unique().tolist(),
                "average_processing_time": df['processing_time'].mean(),
                "success_rate": len(results) / len(results)  # All completed
            },
            "modality_analysis": {},
            "performance_metrics": {
                "avg_response_length": df['response_length'].mean(),
                "structured_response_rate": df['structured_response'].mean(),
                "avg_confidence_score": df['evaluation_metrics'].apply(lambda x: x.get('solution_confidence', 0)).mean()
            }
        }
        
        # Analyze by modality
        for modality in df['modality'].unique():
            modality_df = df[df['modality'] == modality]
            report["modality_analysis"][modality] = {
                "count": len(modality_df),
                "avg_processing_time": modality_df['processing_time'].mean(),
                "avg_confidence": modality_df['evaluation_metrics'].apply(lambda x: x.get('solution_confidence', 0)).mean(),
                "structured_response_rate": modality_df['structured_response'].mean()
            }
        
        return report

def main():
    """Main function to run experiments"""
    print("🎯 GUI Bug Repair LLM Experiment Framework")
    print("=" * 50)
    
    # Initialize framework
    framework = ExperimentFramework()
    
    # Run experiments
    print("\n🚀 Starting experiments...")
    results = framework.run_experiment_suite()
    
    # Save results
    print("\n💾 Saving results...")
    results_file = framework.save_results(results)
    
    # Generate report
    print("\n📊 Generating report...")
    report = framework.generate_report(results)
    
    # Display summary
    print("\n📈 EXPERIMENT SUMMARY")
    print("-" * 30)
    print(f"Total experiments: {report['summary']['total_experiments']}")
    print(f"Modalities tested: {', '.join(report['summary']['modalities_tested'])}")
    print(f"Average processing time: {report['summary']['average_processing_time']:.2f}s")
    print(f"Structured response rate: {report['performance_metrics']['structured_response_rate']:.1%}")
    print(f"Average confidence score: {report['performance_metrics']['avg_confidence_score']:.2f}")
    
    # Show modality breakdown
    print("\n📊 MODALITY BREAKDOWN:")
    for modality, stats in report['modality_analysis'].items():
        print(f"  {modality}: {stats['count']} tests, "
              f"avg time: {stats['avg_processing_time']:.2f}s, "
              f"confidence: {stats['avg_confidence']:.2f}")
    
    print(f"\n✅ Results saved to: {results_file}")

if __name__ == "__main__":
    main()
