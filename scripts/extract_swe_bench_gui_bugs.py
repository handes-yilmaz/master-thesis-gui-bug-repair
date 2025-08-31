#!/usr/bin/env python
"""
Extract GUI-Related Bugs from SWE-Bench Multimodal
==================================================

This script efficiently extracts GUI-related bugs from SWE-bench Multimodal
with limited resources, focusing on web/frontend issues.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

def extract_gui_bugs_from_swe_bench(max_samples: int = 100, save_to_file: bool = True):
    """Extract GUI-related bugs from SWE-bench Multimodal"""
    
    print("🔍 Extracting GUI bugs from SWE-bench Multimodal...")
    
    try:
        from datasets import load_dataset
        
        # Load the multimodal dataset
        print("📥 Loading SWE-bench Multimodal dataset...")
        swebench_mm = load_dataset('SWE-bench/SWE-bench_Multimodal', split=f'test[:{max_samples}]')
        
        print(f"✅ Loaded {len(swebench_mm)} samples")
        
        # Define GUI-related keywords and patterns
        gui_keywords = [
            # UI Components
            'button', 'form', 'input', 'select', 'dropdown', 'modal', 'dialog',
            'table', 'grid', 'list', 'menu', 'navigation', 'sidebar', 'header',
            'footer', 'card', 'panel', 'tooltip', 'popup', 'overlay',
            
            # Web Technologies
            'react', 'vue', 'angular', 'javascript', 'css', 'html', 'dom',
            'frontend', 'ui', 'ux', 'interface', 'web', 'browser',
            
            # User Interactions
            'click', 'hover', 'focus', 'scroll', 'drag', 'drop', 'swipe',
            'tap', 'touch', 'keyboard', 'mouse', 'event', 'handler',
            
            # Visual Elements
            'layout', 'styling', 'design', 'responsive', 'mobile', 'desktop',
            'color', 'font', 'background', 'border', 'shadow', 'animation',
            'transition', 'visual', 'appearance', 'rendering'
        ]
        
        # Extract GUI-related bugs
        gui_bugs = []
        
        for i, sample in enumerate(swebench_mm):
            problem = sample.get('problem_statement', '').lower()
            repo = sample.get('repo', 'unknown')
            
            # Check if it's GUI-related
            keyword_matches = sum(1 for keyword in gui_keywords if keyword in problem)
            
            if keyword_matches >= 2:  # At least 2 GUI keywords
                # Classify the bug type
                bug_type = classify_bug_type(problem)
                
                bug_info = {
                    'swe_bench_id': sample.get('instance_id', f'bug_{i}'),
                    'repo': repo,
                    'title': extract_title(problem),
                    'description': problem[:500] + '...' if len(problem) > 500 else problem,
                    'bug_type': bug_type,
                    'severity': estimate_severity(problem),
                    'keywords_found': [kw for kw in gui_keywords if kw in problem],
                    'has_image': bool(sample.get('image_assets')),
                    'created_at': sample.get('created_at', ''),
                    'patch_available': bool(sample.get('patch')),
                    'test_patch_available': bool(sample.get('test_patch'))
                }
                
                gui_bugs.append(bug_info)
                
                if len(gui_bugs) % 10 == 0:
                    print(f"  Processed {len(gui_bugs)} GUI bugs...")
        
        print(f"\n🎯 Found {len(gui_bugs)} GUI-related bugs!")
        
        # Save to file if requested
        if save_to_file:
            output_dir = Path("data/processed/swe_bench_gui_bugs")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / f"swe_bench_gui_bugs_{len(gui_bugs)}.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(gui_bugs, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Saved to: {output_file}")
            
            # Also save a summary
            summary_file = output_dir / f"summary_{len(gui_bugs)}_gui_bugs.txt"
            save_summary(gui_bugs, summary_file)
            
            return output_file
        
        return gui_bugs
        
    except ImportError:
        print("❌ datasets library not available. Install with: pip install datasets")
        return None
    except Exception as e:
        print(f"❌ Error extracting bugs: {e}")
        return None

def classify_bug_type(problem_description: str) -> str:
    """Classify the bug type based on the problem description"""
    problem_lower = problem_description.lower()
    
    # UI Layout Issues
    if any(word in problem_lower for word in ['layout', 'positioning', 'alignment', 'spacing', 'overflow']):
        return 'ui_layout'
    
    # Interaction Issues
    if any(word in problem_lower for word in ['click', 'hover', 'focus', 'event', 'handler', 'interaction']):
        return 'interaction'
    
    # Styling Issues
    if any(word in problem_lower for word in ['css', 'styling', 'color', 'font', 'background', 'style']):
        return 'styling'
    
    # Responsive Issues
    if any(word in problem_lower for word in ['mobile', 'responsive', 'viewport', 'screen size', 'breakpoint']):
        return 'responsive'
    
    # Performance Issues
    if any(word in problem_lower for word in ['slow', 'lag', 'performance', 'rendering', 'animation']):
        return 'performance'
    
    # Form Issues
    if any(word in problem_lower for word in ['form', 'input', 'validation', 'submit', 'field']):
        return 'form'
    
    # Component Issues
    if any(word in problem_lower for word in ['component', 'widget', 'element', 'button', 'table']):
        return 'component'
    
    return 'general_ui'

def estimate_severity(problem_description: str) -> str:
    """Estimate bug severity based on keywords and context"""
    problem_lower = problem_description.lower()
    
    # Critical keywords
    critical_words = ['crash', 'error', 'exception', 'fail', 'broken', 'unusable', 'security']
    if any(word in problem_lower for word in critical_words):
        return 'critical'
    
    # High severity keywords
    high_words = ['bug', 'issue', 'problem', 'not working', 'broken', 'incorrect']
    if any(word in problem_lower for word in high_words):
        return 'high'
    
    # Medium severity keywords
    medium_words = ['improvement', 'enhancement', 'better', 'optimize', 'polish']
    if any(word in problem_lower for word in medium_words):
        return 'medium'
    
    return 'low'

def extract_title(problem_description: str) -> str:
    """Extract a title from the problem description"""
    # Look for the first line that might be a title
    lines = problem_description.split('\n')
    
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('<!--'):
            # Remove markdown formatting
            line = re.sub(r'[#*`]', '', line)
            if len(line) > 10 and len(line) < 100:
                return line
    
    # Fallback: first meaningful line
    for line in lines:
        line = line.strip()
        if line and not line.startswith('<!--') and len(line) > 10:
            return line[:80] + '...' if len(line) > 80 else line
    
    return "GUI Bug (No title available)"

def save_summary(bugs: List[Dict[str, Any]], output_file: Path):
    """Save a summary of the extracted bugs"""
    summary = []
    summary.append("=" * 60)
    summary.append("SWE-BENCH GUI BUGS SUMMARY")
    summary.append("=" * 60)
    summary.append(f"Total Bugs: {len(bugs)}")
    summary.append("")
    
    # Count by bug type
    bug_types = {}
    for bug in bugs:
        bug_type = bug.get('bug_type', 'unknown')
        if bug_type not in bug_types:
            bug_types[bug_type] = 0
        bug_types[bug_type] += 1
    
    summary.append("Bug Types:")
    for bug_type, count in sorted(bug_types.items(), key=lambda x: x[1], reverse=True):
        summary.append(f"  {bug_type}: {count}")
    
    summary.append("")
    
    # Count by severity
    severities = {}
    for bug in bugs:
        severity = bug.get('severity', 'unknown')
        if severity not in severities:
            severities[severity] = 0
        severities[severity] += 1
    
    summary.append("Severity Levels:")
    for severity, count in sorted(severities.items(), key=lambda x: x[1], reverse=True):
        summary.append(f"  {severity}: {count}")
    
    summary.append("")
    
    # Repositories
    repos = set(bug.get('repo', 'unknown') for bug in bugs)
    summary.append(f"Repositories: {len(repos)}")
    summary.append("Sample repositories:")
    for repo in list(repos)[:10]:
        summary.append(f"  - {repo}")
    
    summary.append("")
    
    # Sample bugs
    summary.append("Sample Bugs:")
    for i, bug in enumerate(bugs[:5]):
        title = bug.get('title', 'No title')[:60]
        bug_type = bug.get('bug_type', 'unknown')
        severity = bug.get('severity', 'unknown')
        summary.append(f"  {i+1}. [{bug_type}/{severity}] {title}")
    
    # Write summary
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary))
    
    print(f"📋 Summary saved to: {output_file}")

def create_integration_script(bugs: List[Dict[str, Any]]):
    """Create a script to integrate SWE-bench bugs with your experiment framework"""
    
    integration_script = """
# Integration script for SWE-bench GUI bugs
# This script converts SWE-bench bugs to your experiment framework format

import json
from pathlib import Path

def convert_swe_bench_to_experiment_format():
    \"\"\"Convert SWE-bench bugs to your experiment framework format\"\"\"
    
    # Load the extracted GUI bugs
    bugs_file = Path("data/processed/swe_bench_gui_bugs/swe_bench_gui_bugs_*.json")
    bugs_files = list(bugs_file.parent.glob(bugs_file.name))
    
    if not bugs_files:
        print("No SWE-bench bugs file found!")
        return
    
    latest_file = max(bugs_files, key=lambda x: x.stat().st_mtime)
    
    with open(latest_file, 'r') as f:
        swe_bugs = json.load(f)
    
    # Convert to your experiment format
    experiment_bugs = []
    
    for bug in swe_bugs:
        experiment_bug = {
            "bug_id": bug["swe_bench_id"],
            "title": bug["title"],
            "description": bug["description"],
            "expected_solution": "Fix the described GUI issue",
            "bug_category": bug["bug_type"],
            "severity": bug["severity"],
            "difficulty": "medium",  # Default difficulty
            "ui_context": bug["description"][:200],
            "code_snippet": None,  # SWE-bench doesn't provide code snippets in metadata
            "security_implications": None,
            "performance_impact": None,
            "browser_specific": None,
            "mobile_affected": "responsive" in bug["bug_type"],
            "accessibility_impact": None,
            "testing_scenarios": None,
            "fix_priority": bug["severity"]
        }
        
        experiment_bugs.append(experiment_bug)
    
    # Save in your experiment format
    output_file = Path("data/processed/swe_bench_experiment_bugs.json")
    with open(output_file, 'w') as f:
        json.dump(experiment_bugs, f, indent=2)
    
    print(f"✅ Converted {len(experiment_bugs)} SWE-bench bugs to experiment format")
    print(f"💾 Saved to: {output_file}")
    
    return experiment_bugs

if __name__ == "__main__":
    convert_swe_bench_to_experiment_format()
"""
    
    # Save the integration script
    output_dir = Path("scripts")
    output_dir.mkdir(exist_ok=True)
    
    script_file = output_dir / "integrate_swe_bench_bugs.py"
    with open(script_file, 'w') as f:
        f.write(integration_script)
    
    print(f"🔧 Integration script created: {script_file}")

def main():
    """Main function to extract GUI bugs from SWE-bench"""
    print("🚀 SWE-Bench GUI Bug Extractor")
    print("=" * 40)
    
    # Extract bugs (start with 50 to test)
    print("\n📊 Starting with 50 samples to test...")
    bugs = extract_gui_bugs_from_swe_bench(max_samples=50, save_to_file=True)
    
    if bugs:
        print(f"\n🎉 Successfully extracted {len(bugs)} GUI bugs!")
        
        # Create integration script
        print("\n🔧 Creating integration script...")
        create_integration_script(bugs)
        
        print("\n📋 Next Steps:")
        print("1. Review the extracted bugs in the JSON file")
        print("2. Run the integration script to convert to your format")
        print("3. Replace synthetic scenarios with real SWE-bench bugs")
        print("4. Run experiments on real data!")
        
    else:
        print("\n❌ Failed to extract bugs. Check the error messages above.")

if __name__ == "__main__":
    main()
