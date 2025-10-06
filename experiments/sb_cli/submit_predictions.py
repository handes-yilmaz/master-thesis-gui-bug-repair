#!/usr/bin/env python3
"""
SWE-bench Prediction Submission Script
Converts experiment results to sb-cli format and submits for validation
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any
import argparse

def convert_results_to_predictions(results_file: str, output_file: str) -> bool:
    """
    Convert experiment results to sb-cli predictions format.
    """
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        predictions = {}
        
        for result in results:
            instance_id = result['instance_id']
            
            # Extract the model patch from the result
            model_patch = result.get('model_patch', '')
            if not model_patch:
                print(f"Warning: No model_patch found for {instance_id}")
                continue
            
            predictions[instance_id] = {
                "model_patch": model_patch,
                "model_name_or_path": result['model']
            }
        
        # Save predictions in sb-cli format
        with open(output_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        print(f"✅ Converted {len(predictions)} predictions to {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error converting results: {e}")
        return False

def submit_predictions(predictions_file: str, subset: str = "swe-bench-m", split: str = "dev", 
                      run_id: str = None, output_dir: str = "sb-cli-reports") -> bool:
    """
    Submit predictions to SWE-bench M API using sb-cli.
    """
    try:
        cmd = [
            "sb-cli", "submit",
            subset, split,
            "--predictions_path", predictions_file,
            "--output_dir", output_dir,
            "--gen_report", "1",
            "--verify_submission", "1",
            "--wait_for_evaluation", "1"
        ]
        
        if run_id:
            cmd.extend(["--run_id", run_id])
        
        print(f"🚀 Submitting predictions to {subset} {split}...")
        print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        print("✅ Submission successful!")
        print("📊 Evaluation completed!")
        print(f"📁 Reports saved to: {output_dir}/")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Submission failed: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Error during submission: {e}")
        return False

def get_submission_report(output_dir: str, run_id: str = None) -> Dict[str, Any]:
    """
    Get the submission report and extract Pass@1 results.
    """
    try:
        # Find the most recent report file
        report_dir = Path(output_dir)
        if not report_dir.exists():
            print(f"❌ Report directory not found: {output_dir}")
            return {}
        
        # Look for JSON report files
        report_files = list(report_dir.glob("*.json"))
        if not report_files:
            print(f"❌ No report files found in {output_dir}")
            return {}
        
        # Use the most recent report file
        latest_report = max(report_files, key=lambda p: p.stat().st_mtime)
        
        with open(latest_report, 'r') as f:
            report = json.load(f)
        
        print(f"📊 Report loaded from: {latest_report}")
        return report
        
    except Exception as e:
        print(f"❌ Error reading report: {e}")
        return {}

def analyze_results(report: Dict[str, Any]) -> None:
    """
    Analyze and display the evaluation results.
    """
    if not report:
        print("❌ No report data to analyze")
        return
    
    print("\n" + "="*60)
    print("📊 SWE-BENCH EVALUATION RESULTS")
    print("="*60)
    
    # Basic statistics
    total_instances = report.get('total_instances', 0)
    submitted_instances = report.get('submitted_instances', 0)
    completed_instances = report.get('completed_instances', 0)
    failed_instances = report.get('failed_instances', 0)
    
    print(f"📈 Total Instances: {total_instances}")
    print(f"📤 Submitted: {submitted_instances}")
    print(f"✅ Completed: {completed_instances}")
    print(f"❌ Failed: {failed_instances}")
    
    # Pass@1 calculation
    if completed_instances > 0:
        # Look for pass@1 results in the report
        pass_at_1 = report.get('pass_at_1', 0)
        pass_at_1_rate = (pass_at_1 / completed_instances) * 100 if completed_instances > 0 else 0
        
        print(f"\n🎯 PASS@1 RESULTS:")
        print(f"   Pass@1: {pass_at_1}/{completed_instances} ({pass_at_1_rate:.1f}%)")
    
    # Additional metrics if available
    if 'metrics' in report:
        metrics = report['metrics']
        print(f"\n📊 ADDITIONAL METRICS:")
        for metric, value in metrics.items():
            print(f"   {metric}: {value}")
    
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Submit experiment results to SWE-bench M API")
    parser.add_argument("results_file", help="Path to experiment results JSON file")
    parser.add_argument("--subset", default="swe-bench-m", choices=["swe-bench-m", "swe-bench_lite", "swe-bench_verified"],
                       help="SWE-bench subset to submit to")
    parser.add_argument("--split", default="dev", help="Split to submit to (dev/test)")
    parser.add_argument("--run-id", help="Custom run ID for the submission")
    parser.add_argument("--output-dir", default="sb-cli-reports", help="Directory to save reports")
    parser.add_argument("--convert-only", action="store_true", help="Only convert results, don't submit")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze existing reports")
    
    args = parser.parse_args()
    
    if args.analyze_only:
        # Analyze existing reports
        report = get_submission_report(args.output_dir)
        analyze_results(report)
        return
    
    # Convert results to predictions format
    results_file = Path(args.results_file)
    predictions_file = results_file.parent / f"{results_file.stem}_predictions.json"
    
    if not convert_results_to_predictions(str(results_file), str(predictions_file)):
        sys.exit(1)
    
    if args.convert_only:
        print(f"✅ Predictions converted to: {predictions_file}")
        return
    
    # Submit predictions
    if not submit_predictions(str(predictions_file), args.subset, args.split, args.run_id, args.output_dir):
        sys.exit(1)
    
    # Analyze results
    report = get_submission_report(args.output_dir)
    analyze_results(report)

if __name__ == "__main__":
    main()
