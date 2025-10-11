#!/usr/bin/env python3
"""
Create SWE-bench CLI submission JSON from results
"""
import json
import sys
from pathlib import Path
from typing import Dict


def collect_patches(results_dir: Path, model_name: str) -> Dict[str, Dict]:
    """
    Collect all patches from experiment results
    
    Args:
        results_dir: Base results directory
        model_name: Model name for submission
        
    Returns:
        Dict mapping instance_id to submission format
    """
    patches = {}
    empty_count = 0
    
    # Find all changes.diff files
    for diff_file in results_dir.rglob("changes.diff"):
        # Extract instance_id from path
        # Path structure: results/test/repo/instance_id/changes.diff
        parts = diff_file.parts
        if len(parts) < 2:
            continue
        
        instance_id = parts[-2]  # Second to last part
        
        # Read diff
        with open(diff_file, 'r') as f:
            model_patch = f.read()
        
        # Skip empty patches
        if not model_patch.strip():
            print(f"⚠️  Skipped (empty): {instance_id}")
            empty_count += 1
            continue
        
        # Add to submission
        patches[instance_id] = {
            "model_patch": model_patch,
            "model_name_or_path": model_name
        }
        print(f"✅ Added: {instance_id}")
    
    return patches, empty_count


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python create_submission.py <results_dir> [model_name]")
        print("")
        print("Examples:")
        print("  python create_submission.py results GUIRepair_optimized_GPT4o")
        print("  python create_submission.py results_textonly GUIRepair_textonly_GPT4o")
        sys.exit(1)
    
    results_dir = Path(sys.argv[1])
    model_name = sys.argv[2] if len(sys.argv) > 2 else "GUIRepair_optimized"
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    print("="*70)
    print("SWE-bench CLI Submission Generator")
    print("="*70)
    print(f"Results dir: {results_dir}")
    print(f"Model name: {model_name}")
    print("")
    
    # Collect patches
    print("📦 Collecting patches...")
    patches, empty_count = collect_patches(results_dir, model_name)
    
    # Create output filename
    output_file = results_dir / f"predictions_{model_name}.json"
    
    # Save submission JSON
    with open(output_file, 'w') as f:
        json.dump(patches, f, indent=4)
    
    # Summary
    print("")
    print("="*70)
    print("✅ Submission file created!")
    print("="*70)
    print(f"Output: {output_file}")
    print(f"Total patches: {len(patches)}")
    print(f"Empty patches skipped: {empty_count}")
    print("")
    
    # Breakdown by repository
    repo_counts = {}
    for instance_id in patches.keys():
        repo = instance_id.split('__')[0]
        repo_counts[repo] = repo_counts.get(repo, 0) + 1
    
    print("📊 Patches by repository:")
    for repo in sorted(repo_counts.keys()):
        print(f"   • {repo:20} {repo_counts[repo]:3} patches")
    
    print("")
    print("🚀 Ready for submission:")
    print(f"   sb-cli submit swe-bench-m test \\")
    print(f"       --predictions_path {output_file} \\")
    print(f"       --run_id {model_name}")
    print("")


if __name__ == "__main__":
    main()



