#!/usr/bin/env python
"""
Integration Script for SWE-bench GUI Bugs
=========================================

This script converts SWE-bench bugs to your experiment framework format,
replacing synthetic scenarios with real-world data.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

def convert_swe_bench_to_experiment_format():
    """Convert SWE-bench bugs to your experiment framework format"""
    
    print("🔄 Converting SWE-bench bugs to experiment format...")
    
    # Load the extracted GUI bugs
    bugs_dir = Path("data/processed/swe_bench_gui_bugs")
    bugs_files = list(bugs_dir.glob("swe_bench_gui_bugs_*.json"))
    
    if not bugs_files:
        print("❌ No SWE-bench bugs file found!")
        print("Run the extract_swe_bench_gui_bugs.py script first.")
        return None
    
    # Get the latest file
    latest_file = max(bugs_files, key=lambda x: x.stat().st_mtime)
    print(f"📁 Loading bugs from: {latest_file}")
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        swe_bugs = json.load(f)
    
    print(f"✅ Loaded {len(swe_bugs)} SWE-bench bugs")
    
    # Convert to your experiment format
    experiment_bugs = []
    
    for i, bug in enumerate(swe_bugs):
        # Map SWE-bench fields to your experiment format
        experiment_bug = {
            "bug_id": bug["swe_bench_id"],
            "title": bug["title"],
            "description": bug["description"],
            "expected_solution": generate_expected_solution(bug),
            "bug_category": bug["bug_type"],
            "severity": bug["severity"],
            "difficulty": estimate_difficulty(bug),
            "ui_context": bug["description"][:300] + "..." if len(bug["description"]) > 300 else bug["description"],
            "code_snippet": None,  # SWE-bench doesn't provide code snippets in metadata
            "security_implications": estimate_security_implications(bug),
            "performance_impact": estimate_performance_impact(bug),
            "browser_specific": estimate_browser_specific(bug),
            "mobile_affected": bug["bug_type"] in ["responsive", "mobile"],
            "accessibility_impact": estimate_accessibility_impact(bug),
            "testing_scenarios": generate_testing_scenarios(bug),
            "fix_priority": bug["severity"]
        }
        
        experiment_bugs.append(experiment_bug)
        
        if (i + 1) % 10 == 0:
            print(f"  Converted {i + 1}/{len(swe_bugs)} bugs...")
    
    # Save in your experiment format
    output_file = Path("data/processed/swe_bench_experiment_bugs.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(experiment_bugs, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Successfully converted {len(experiment_bugs)} SWE-bench bugs to experiment format")
    print(f"💾 Saved to: {output_file}")
    
    # Generate summary
    generate_conversion_summary(experiment_bugs, output_file)
    
    return experiment_bugs

def generate_expected_solution(bug: Dict[str, Any]) -> str:
    """Generate expected solution based on bug type and description"""
    bug_type = bug.get("bug_type", "")
    description = bug.get("description", "").lower()
    
    if bug_type == "ui_layout":
        return "Fix the layout issue by adjusting positioning, spacing, or overflow handling"
    elif bug_type == "interaction":
        return "Fix the interaction issue by correcting event handling, animations, or user input processing"
    elif bug_type == "styling":
        return "Fix the styling issue by correcting CSS properties, color application, or visual appearance"
    elif bug_type == "responsive":
        return "Fix the responsive design issue for different screen sizes and devices"
    elif bug_type == "performance":
        return "Optimize performance by improving rendering, animations, or resource usage"
    elif bug_type == "form":
        return "Fix the form validation or input handling issue"
    elif bug_type == "component":
        return "Fix the component functionality or add missing component features"
    else:
        return "Fix the described GUI issue according to the problem statement"

def estimate_difficulty(bug: Dict[str, Any]) -> str:
    """Estimate bug difficulty based on type and complexity"""
    bug_type = bug.get("bug_type", "")
    description = bug.get("description", "")
    
    # High difficulty indicators
    if any(word in description.lower() for word in ["complex", "complicated", "multiple", "integration", "performance"]):
        return "high"
    
    # Medium difficulty for most UI bugs
    if bug_type in ["ui_layout", "interaction", "responsive"]:
        return "medium"
    
    # Lower difficulty for styling issues
    if bug_type in ["styling", "component"]:
        return "low"
    
    return "medium"

def estimate_security_implications(bug: Dict[str, Any]) -> List[str]:
    """Estimate security implications of the bug"""
    description = bug.get("description", "").lower()
    
    implications = []
    
    if any(word in description for word in ["input", "form", "validation", "user data"]):
        implications.append("input validation")
    
    if any(word in description for word in ["authentication", "authorization", "access"]):
        implications.append("access control")
    
    if any(word in description for word in ["xss", "injection", "script"]):
        implications.append("code injection")
    
    if not implications:
        implications.append("minimal")
    
    return implications

def estimate_performance_impact(bug: Dict[str, Any]) -> str:
    """Estimate performance impact of the bug"""
    bug_type = bug.get("bug_type", "")
    description = bug.get("description", "").lower()
    
    if bug_type == "performance":
        return "high - directly affects performance"
    elif any(word in description for word in ["slow", "lag", "freeze", "unresponsive"]):
        return "medium - affects user experience"
    elif bug_type in ["animation", "transition"]:
        return "medium - affects rendering performance"
    else:
        return "low - minimal performance impact"

def estimate_browser_specific(bug: Dict[str, Any]) -> List[str]:
    """Estimate browser-specific issues"""
    description = bug.get("description", "").lower()
    
    browsers = []
    
    if "chrome" in description:
        browsers.append("chrome")
    if "firefox" in description:
        browsers.append("firefox")
    if "safari" in description:
        browsers.append("safari")
    if "edge" in description:
        browsers.append("edge")
    
    if not browsers:
        browsers.append("cross-browser")
    
    return browsers

def estimate_accessibility_impact(bug: Dict[str, Any]) -> str:
    """Estimate accessibility impact of the bug"""
    bug_type = bug.get("bug_type", "")
    description = bug.get("description", "").lower()
    
    if any(word in description for word in ["screen reader", "keyboard", "focus", "aria"]):
        return "high - affects accessibility features"
    elif bug_type in ["interaction", "form"]:
        return "medium - may affect user interaction"
    else:
        return "low - minimal accessibility impact"

def generate_testing_scenarios(bug: Dict[str, Any]) -> List[str]:
    """Generate testing scenarios for the bug"""
    bug_type = bug.get("bug_type", "")
    description = bug.get("description", "").lower()
    
    scenarios = []
    
    if bug_type == "ui_layout":
        scenarios.extend([
            "Test on different screen sizes",
            "Test with various content lengths",
            "Test overflow handling"
        ])
    elif bug_type == "interaction":
        scenarios.extend([
            "Test user interactions (click, hover, focus)",
            "Test with different input devices",
            "Test edge cases in user flow"
        ])
    elif bug_type == "responsive":
        scenarios.extend([
            "Test on mobile devices",
            "Test on tablets",
            "Test responsive breakpoints"
        ])
    elif bug_type == "form":
        scenarios.extend([
            "Test form validation",
            "Test with various input types",
            "Test form submission flow"
        ])
    else:
        scenarios.append("Test the described functionality")
        scenarios.append("Verify the fix resolves the issue")
    
    return scenarios

def generate_conversion_summary(bugs: List[Dict[str, Any]], output_file: Path):
    """Generate a summary of the conversion"""
    
    summary_file = output_file.parent / f"conversion_summary_{len(bugs)}_bugs.txt"
    
    summary = []
    summary.append("=" * 60)
    summary.append("SWE-BENCH TO EXPERIMENT FORMAT CONVERSION SUMMARY")
    summary.append("=" * 60)
    summary.append(f"Total Bugs Converted: {len(bugs)}")
    summary.append("")
    
    # Count by bug category
    categories = {}
    for bug in bugs:
        category = bug.get("bug_category", "unknown")
        if category not in categories:
            categories[category] = 0
        categories[category] += 1
    
    summary.append("Bug Categories:")
    for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        summary.append(f"  {category}: {count}")
    
    summary.append("")
    
    # Count by severity
    severities = {}
    for bug in bugs:
        severity = bug.get("severity", "unknown")
        if severity not in severities:
            severities[severity] = 0
        severities[severity] += 1
    
    summary.append("Severity Levels:")
    for severity, count in sorted(severities.items(), key=lambda x: x[1], reverse=True):
        summary.append(f"  {severity}: {count}")
    
    summary.append("")
    
    # Count by difficulty
    difficulties = {}
    for bug in bugs:
        difficulty = bug.get("difficulty", "unknown")
        if difficulty not in difficulties:
            difficulties[difficulty] = 0
        difficulties[difficulty] += 1
    
    summary.append("Difficulty Levels:")
    for difficulty, count in sorted(difficulties.items(), key=lambda x: x[1], reverse=True):
        summary.append(f"  {difficulty}: {count}")
    
    summary.append("")
    
    # Integration status
    summary.append("Integration Status:")
    summary.append("✅ SWE-bench bugs extracted")
    summary.append("✅ Converted to experiment format")
    summary.append("✅ Ready for LLM testing")
    summary.append("")
    
    # Next steps
    summary.append("Next Steps:")
    summary.append("1. Replace synthetic scenarios with these real bugs")
    summary.append("2. Update your experiment framework to use real data")
    summary.append("3. Run experiments on authentic bug scenarios")
    summary.append("4. Generate credible research results!")
    
    # Write summary
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary))
    
    print(f"📋 Conversion summary saved to: {summary_file}")

def main():
    """Main function to convert SWE-bench bugs"""
    print("🔄 SWE-Bench to Experiment Format Converter")
    print("=" * 50)
    
    # Convert the bugs
    experiment_bugs = convert_swe_bench_to_experiment_format()
    
    if experiment_bugs:
        print(f"\n🎉 Successfully converted {len(experiment_bugs)} bugs!")
        print("\n📋 What was accomplished:")
        print("✅ Extracted real GUI bugs from SWE-bench Multimodal")
        print("✅ Converted to your experiment framework format")
        print("✅ Generated comprehensive bug metadata")
        print("✅ Created integration summary")
        
        print("\n🚀 Next Steps:")
        print("1. Review the converted bugs in the JSON file")
        print("2. Update your experiment framework to use real data")
        print("3. Replace synthetic scenarios with these authentic bugs")
        print("4. Run experiments on real-world problems!")
        
        print(f"\n💡 Research now has {len(experiment_bugs)} real GUI bugs instead of synthetic ones!")
        
    else:
        print("\n❌ Conversion failed. Check the error messages above.")

if __name__ == "__main__":
    main()
