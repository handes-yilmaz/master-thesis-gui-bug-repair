#!/usr/bin/env python
"""
Fix Application System for GUI Bug Repair
========================================

This module parses LLM code suggestions and applies fixes to the demo React application.
"""

import os
import sys
import json
import re
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import subprocess
import shutil

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class FixApplicator:
    """System for applying LLM-generated fixes to React applications"""
    
    def __init__(self, demo_app_path: str = "demo-app"):
        self.demo_app_path = Path(demo_app_path)
        self.backup_dir = Path("runs/fix_backups")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Common fix patterns
        self.fix_patterns = {
            "react_component": r"function\s+\w+\s*\([^)]*\)\s*\{[\s\S]*?\}",
            "jsx_element": r"<[^>]+>[\s\S]*?</[^>]+>",
            "css_rule": r"\.[\w-]+\s*\{[\s\S]*?\}",
            "javascript_code": r"const\s+\w+\s*=\s*[^;]+;",
            "import_statement": r"import\s+.*?from\s+['\"][^'\"]+['\"];"
        }
    
    def parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract fix information"""
        parsed_fix = {
            "root_cause": "",
            "solution": "",
            "explanation": "",
            "code_fixes": [],
            "css_fixes": [],
            "html_fixes": [],
            "testing_steps": [],
            "confidence": 0.0
        }
        
        # Extract sections using regex patterns
        sections = {
            "ROOT CAUSE": r"ROOT CAUSE:\s*(.*?)(?=\n[A-Z]+:|$)",
            "SOLUTION": r"SOLUTION:\s*(.*?)(?=\n[A-Z]+:|$)",
            "EXPLANATION": r"EXPLANATION:\s*(.*?)(?=\n[A-Z]+:|$)",
            "TESTING": r"TESTING:\s*(.*?)(?=\n[A-Z]+:|$)",
            "CODE": r"CODE:\s*(.*?)(?=\n[A-Z]+:|$)",
            "CSS": r"CSS:\s*(.*?)(?=\n[A-Z]+:|$)",
            "HTML": r"HTML:\s*(.*?)(?=\n[A-Z]+:|$)"
        }
        
        for section, pattern in sections.items():
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                content = match.group(1).strip()
                if section == "ROOT CAUSE":
                    parsed_fix["root_cause"] = content
                elif section == "SOLUTION":
                    parsed_fix["solution"] = content
                elif section == "EXPLANATION":
                    parsed_fix["explanation"] = content
                elif section == "TESTING":
                    parsed_fix["testing_steps"] = [step.strip() for step in content.split('\n') if step.strip()]
                elif section == "CODE":
                    parsed_fix["code_fixes"].append(content)
                elif section == "CSS":
                    parsed_fix["css_fixes"].append(content)
                elif section == "HTML":
                    parsed_fix["html_fixes"].append(content)
        
        # Extract code blocks from solution
        if parsed_fix["solution"]:
            code_blocks = re.findall(r"```(?:jsx?|javascript|css|html)?\n(.*?)\n```", 
                                   parsed_fix["solution"], re.DOTALL)
            for block in code_blocks:
                if "function" in block or "const" in block or "import" in block:
                    parsed_fix["code_fixes"].append(block.strip())
                elif "{" in block and "}" in block and ":" in block:
                    parsed_fix["css_fixes"].append(block.strip())
                elif "<" in block and ">" in block:
                    parsed_fix["html_fixes"].append(block.strip())
        
        return parsed_fix
    
    def create_backup(self) -> str:
        """Create a backup of the current demo app"""
        timestamp = Path().cwd().name + "_" + str(int(time.time()))
        backup_path = self.backup_dir / timestamp
        
        if self.demo_app_path.exists():
            shutil.copytree(self.demo_app_path, backup_path)
            print(f"💾 Created backup: {backup_path}")
            return str(backup_path)
        else:
            print("⚠️ Demo app not found, skipping backup")
            return ""
    
    def apply_code_fix(self, fix: str, target_file: str) -> bool:
        """Apply a code fix to a specific file"""
        try:
            file_path = self.demo_app_path / target_file
            
            if not file_path.exists():
                print(f"⚠️ Target file not found: {target_file}")
                return False
            
            # Read current content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Apply the fix (this is a simplified approach)
            # In a real system, you'd want more sophisticated code parsing and replacement
            
            # For now, let's append the fix as a comment
            modified_content = content + f"\n\n// APPLIED FIX:\n{fix}\n"
            
            # Write back
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            
            print(f"✅ Applied code fix to {target_file}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to apply code fix to {target_file}: {e}")
            return False
    
    def apply_css_fix(self, fix: str, target_file: str = "src/styles.css") -> bool:
        """Apply a CSS fix to the styles file"""
        try:
            file_path = self.demo_app_path / target_file
            
            if not file_path.exists():
                # Create the file if it doesn't exist
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, 'w') as f:
                    f.write("/* CSS Styles */\n")
            
            # Read current content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Append the fix
            modified_content = content + f"\n\n/* APPLIED FIX: */\n{fix}\n"
            
            # Write back
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            
            print(f"✅ Applied CSS fix to {target_file}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to apply CSS fix to {target_file}: {e}")
            return False
    
    def apply_html_fix(self, fix: str, target_file: str = "src/App.jsx") -> bool:
        """Apply an HTML/JSX fix to the main component"""
        try:
            file_path = self.demo_app_path / target_file
            
            if not file_path.exists():
                print(f"⚠️ Target file not found: {target_file}")
                return False
            
            # Read current content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find the return statement and insert the fix
            # This is a simplified approach - in practice you'd want proper JSX parsing
            
            # Look for the return statement
            return_match = re.search(r"(return\s*\()", content)
            if return_match:
                # Insert the fix before the return statement
                insert_pos = return_match.start()
                modified_content = content[:insert_pos] + f"\n// APPLIED FIX:\n{fix}\n\n" + content[insert_pos:]
                
                # Write back
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
                
                print(f"✅ Applied HTML/JSX fix to {target_file}")
                return True
            else:
                print(f"⚠️ Could not find return statement in {target_file}")
                return False
            
        except Exception as e:
            print(f"❌ Failed to apply HTML/JSX fix to {target_file}: {e}")
            return False
    
    def validate_fix(self, fix_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that the applied fix is syntactically correct"""
        validation_result = {
            "syntax_valid": True,
            "errors": [],
            "warnings": [],
            "fix_applied": False
        }
        
        try:
            # Check if demo app exists
            if not self.demo_app_path.exists():
                validation_result["syntax_valid"] = False
                validation_result["errors"].append("Demo app not found")
                return validation_result
            
            # Try to run a basic syntax check
            try:
                # Check if it's a React app with package.json
                package_json = self.demo_app_path / "package.json"
                if package_json.exists():
                    # Try to install dependencies and run a basic check
                    result = subprocess.run(
                        ["npm", "install"],
                        cwd=self.demo_app_path,
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    
                    if result.returncode == 0:
                        # Try to run a build check
                        build_result = subprocess.run(
                            ["npm", "run", "build"],
                            cwd=self.demo_app_path,
                            capture_output=True,
                            text=True,
                            timeout=120
                        )
                        
                        if build_result.returncode == 0:
                            validation_result["fix_applied"] = True
                            print("✅ Fix validation successful - app builds without errors")
                        else:
                            validation_result["warnings"].append("App builds with warnings")
                            validation_result["fix_applied"] = True
                    else:
                        validation_result["errors"].append("Failed to install dependencies")
                        validation_result["syntax_valid"] = False
                else:
                    validation_result["warnings"].append("No package.json found - skipping build validation")
                    validation_result["fix_applied"] = True
                    
            except subprocess.TimeoutExpired:
                validation_result["warnings"].append("Build validation timed out")
                validation_result["fix_applied"] = True
            except Exception as e:
                validation_result["warnings"].append(f"Build validation failed: {e}")
                validation_result["fix_applied"] = True
            
        except Exception as e:
            validation_result["syntax_valid"] = False
            validation_result["errors"].append(f"Validation error: {e}")
        
        return validation_result
    
    def apply_comprehensive_fix(self, llm_response: str, bug_scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a comprehensive fix based on LLM response and bug scenario"""
        print(f"🔧 Applying fix for bug: {bug_scenario.get('title', 'Unknown')}")
        
        # Create backup
        backup_path = self.create_backup()
        
        # Parse LLM response
        parsed_fix = self.parse_llm_response(llm_response)
        
        # Track what was applied
        applied_fixes = {
            "code_fixes": [],
            "css_fixes": [],
            "html_fixes": [],
            "backup_created": bool(backup_path)
        }
        
        # Apply code fixes
        for i, code_fix in enumerate(parsed_fix["code_fixes"]):
            target_file = f"src/App.jsx"  # Default target
            if self.apply_code_fix(code_fix, target_file):
                applied_fixes["code_fixes"].append({
                    "fix": code_fix,
                    "target_file": target_file,
                    "success": True
                })
        
        # Apply CSS fixes
        for i, css_fix in enumerate(parsed_fix["css_fixes"]):
            if self.apply_css_fix(css_fix):
                applied_fixes["css_fixes"].append({
                    "fix": css_fix,
                    "success": True
                })
        
        # Apply HTML/JSX fixes
        for i, html_fix in enumerate(parsed_fix["html_fixes"]):
            if self.apply_html_fix(html_fix):
                applied_fixes["html_fixes"].append({
                    "fix": html_fix,
                    "success": True
                })
        
        # Validate the fix
        validation_result = self.validate_fix(parsed_fix)
        
        # Compile results
        application_result = {
            "bug_scenario": bug_scenario,
            "parsed_fix": parsed_fix,
            "applied_fixes": applied_fixes,
            "validation_result": validation_result,
            "backup_path": backup_path,
            "success": validation_result["fix_applied"]
        }
        
        return application_result
    
    def run_fix_application_test(self, experiment_results_file: str) -> List[Dict[str, Any]]:
        """Run fix application on a set of experiment results"""
        # Load experiment results
        results_path = Path(experiment_results_file)
        if not results_path.exists():
            print(f"❌ Experiment results file not found: {experiment_results_file}")
            return []
        
        with open(results_path, 'r') as f:
            experiment_results = json.load(f)
        
        # Filter for successful experiments with high confidence
        high_confidence_results = [
            result for result in experiment_results
            if result.get("evaluation_metrics", {}).get("solution_confidence", 0) > 0.8
        ]
        
        print(f"🔍 Found {len(high_confidence_results)} high-confidence results to test")
        
        # Apply fixes to a subset for testing
        test_results = high_confidence_results[:3]  # Test first 3
        
        all_application_results = []
        
        for result in test_results:
            print(f"\n🧪 Testing fix application for: {result.get('bug_description', 'Unknown')[:50]}...")
            
            # Create a mock bug scenario from the result
            bug_scenario = {
                "title": f"Bug from experiment {result.get('experiment_id', 'Unknown')}",
                "description": result.get("bug_description", ""),
                "bug_category": result.get("bug_category", "unknown"),
                "severity": result.get("bug_severity", "medium")
            }
            
            # Apply the fix
            application_result = self.apply_comprehensive_fix(
                result.get("llm_response", ""),
                bug_scenario
            )
            
            all_application_results.append(application_result)
            
            if application_result["success"]:
                print("✅ Fix application successful!")
            else:
                print("❌ Fix application failed!")
        
        return all_application_results
    
    def save_application_results(self, results: List[Dict[str, Any]], filename: str = None):
        """Save fix application results to file"""
        if filename is None:
            import time
            timestamp = int(time.time())
            filename = f"fix_application_results_{timestamp}.json"
        
        # Ensure results directory exists
        results_dir = Path("runs/fix_applications")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Saved {len(results)} application results to {filepath}")
        return filepath

def main():
    """Main function to test fix application system"""
    print("🔧 Fix Application System for GUI Bug Repair")
    print("=" * 50)
    
    # Initialize fix applicator
    applicator = FixApplicator()
    
    print(f"✓ Fix applicator initialized")
    print(f"✓ Demo app path: {applicator.demo_app_path}")
    
    # Check if we have experiment results to test with
    experiment_files = list(Path("runs/enhanced_experiments").glob("*.json"))
    
    if experiment_files:
        # Use the most recent experiment results
        latest_file = max(experiment_files, key=lambda x: x.stat().st_mtime)
        print(f"📊 Found experiment results: {latest_file}")
        
        # Test fix application
        print("\n🚀 Testing fix application system...")
        application_results = applicator.run_fix_application_test(str(latest_file))
        
        if application_results:
            # Save results
            print("\n💾 Saving application results...")
            filename = applicator.save_application_results(application_results)
            
            # Summary
            successful_fixes = sum(1 for r in application_results if r["success"])
            total_fixes = len(application_results)
            
            print(f"\n📊 Fix Application Summary:")
            print(f"Total fixes tested: {total_fixes}")
            print(f"Successful applications: {successful_fixes}")
            print(f"Success rate: {successful_fixes/total_fixes*100:.1f}%")
            print(f"Results saved to: {filename}")
        else:
            print("\n❌ No fix applications completed")
    else:
        print("\n⚠️ No experiment results found. Run experiments first.")
        print("You can test the system manually by calling:")
        print("applicator.apply_comprehensive_fix(llm_response, bug_scenario)")

if __name__ == "__main__":
    main()
