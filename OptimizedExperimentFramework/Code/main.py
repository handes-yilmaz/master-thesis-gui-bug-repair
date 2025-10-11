"""
GUIRepair Optimized 3-Phase Framework - Main Entry Point

Usage:
    python main.py --instance_id bpmn-io__bpmn-js-1080 --api_key YOUR_KEY
    python main.py --repo_prefix bpmn-io --api_key YOUR_KEY
    python main.py --all --api_key YOUR_KEY
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, List
from datasets import Dataset, load_dataset

from config import GUIRepairConfig
from core import GUIRepairWorkflow, WorkflowRunner
from utils import read_json, save_json


# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='GUIRepair Optimized 3-Phase Framework',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Instance selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--instance_id',
        type=str,
        help='Single instance ID (e.g., bpmn-io__bpmn-js-1080)'
    )
    group.add_argument(
        '--repo_prefix',
        type=str,
        help='Process all instances from repo (e.g., bpmn-io)'
    )
    group.add_argument(
        '--all',
        action='store_true',
        help='Process all instances in dataset'
    )
    
    # Required arguments
    parser.add_argument(
        '--api_key',
        type=str,
        required=True,
        help='OpenAI or Claude API key'
    )
    
    # Model configuration
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o-2024-08-06',
        help='LLM model name'
    )
    
    # Phase temperatures
    parser.add_argument('--file_loc_temp', type=float, default=1.0)
    parser.add_argument('--file_loc_samples', type=int, default=2)
    parser.add_argument('--elem_loc_temp', type=float, default=0.7)
    parser.add_argument('--elem_loc_samples', type=int, default=2)
    parser.add_argument('--patch_temp', type=float, default=1.0)
    parser.add_argument('--patch_samples', type=int, default=1)
    
    # File limits
    parser.add_argument('--max_files', type=int, default=4)
    parser.add_argument('--max_lines', type=int, default=500)
    parser.add_argument('--context', type=int, default=10)
    
    # Dataset
    parser.add_argument(
        '--dataset',
        type=str,
        default='princeton-nlp/SWE-bench_Multimodal'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['test', 'dev']
    )
    parser.add_argument(
        '--repo_path',
        type=str,
        default='../Data/Reproduce_Scenario'
    )
    
    # Output
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results'
    )
    
    # Experiment mode
    parser.add_argument(
        '--text_only',
        action='store_true',
        help='Run in text-only mode (no images)'
    )
    
    # Other
    parser.add_argument('--wait_time', type=int, default=0)
    
    return parser.parse_args()


def load_dataset_cached(dataset_name: str, split: str) -> Dict:
    """
    Load dataset with caching
    
    Args:
        dataset_name: Dataset name
        split: Dataset split (test/dev)
        
    Returns:
        Dataset as dict
    """
    cache_dir = Path('dataset') / split
    cache_file = cache_dir / f"{dataset_name.split('/')[-1]}.json"
    
    # Try cache first
    if cache_file.exists():
        logger.info(f"📦 Loading cached dataset: {cache_file}")
        return read_json(str(cache_file))
    
    # Download and cache
    logger.info(f"📥 Downloading dataset: {dataset_name} ({split})")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = load_dataset(dataset_name, split=split)
    
    # Convert to dict
    dataset_dict = {}
    for instance in dataset:
        instance_id = instance['instance_id']
        dataset_dict[instance_id] = dict(instance)
    
    # Save cache
    save_json(str(cache_file), dataset_dict)
    logger.info(f"💾 Cached dataset to: {cache_file}")
    
    return dataset_dict


def main():
    """Main entry point"""
    args = parse_args()
    
    # Create config
    config = GUIRepairConfig.from_args(args)
    
    logger.info("="*70)
    logger.info("GUIRepair Optimized 3-Phase Framework")
    logger.info("="*70)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Mode: {'Text-only' if config.text_only else 'Multimodal'}")
    logger.info(f"Dataset: {config.dataset} ({config.dataset_split})")
    logger.info("="*70 + "\n")
    
    # Load dataset
    dataset = load_dataset_cached(config.dataset, config.dataset_split)
    logger.info(f"📚 Dataset loaded: {len(dataset)} instances\n")
    
    # Create workflow runner
    runner = WorkflowRunner(config)
    
    # Determine which instances to process
    if args.instance_id:
        # Single instance
        instance_ids = [args.instance_id]
        logger.info(f"🎯 Processing single instance: {args.instance_id}\n")
        
    elif args.repo_prefix:
        # All instances from a repository
        instance_ids = [
            iid for iid in dataset.keys()
            if iid.startswith(args.repo_prefix + '__')
        ]
        logger.info(f"🎯 Processing repository: {args.repo_prefix}")
        logger.info(f"   Found {len(instance_ids)} instances\n")
        
    else:  # args.all
        # All instances
        instance_ids = list(dataset.keys())
        logger.info(f"🎯 Processing ALL instances: {len(instance_ids)}\n")
    
    # Validate instances exist
    instance_ids = [iid for iid in instance_ids if iid in dataset]
    
    if not instance_ids:
        logger.error("❌ No valid instances to process!")
        return 1
    
    # Run workflow(s)
    try:
        if len(instance_ids) == 1:
            # Single instance - detailed output
            result = runner.run_instance(instance_ids[0], dataset)
            
            # Print summary
            print("\n" + "="*70)
            print("RESULTS")
            print("="*70)
            print(f"Instance: {result['instance_id']}")
            print(f"Success: {result['success']}")
            print(f"Duration: {result['duration_seconds']:.1f}s")
            print(f"Tokens: {result['total_tokens']}")
            print(f"Modified files: {len(result.get('modified_files', []))}")
            print("="*70)
            
        else:
            # Multiple instances - batch processing
            results = runner.run_batch(instance_ids, dataset)
            
            # Print summary
            successful = sum(1 for r in results.values() if r.get('success'))
            total_tokens = sum(r.get('total_tokens', 0) for r in results.values())
            
            print("\n" + "="*70)
            print("BATCH RESULTS")
            print("="*70)
            print(f"Processed: {len(results)} instances")
            print(f"Successful: {successful} ({successful/len(results)*100:.1f}%)")
            print(f"Total tokens: {total_tokens}")
            print("="*70)
            
            # Save batch results
            batch_results_file = Path(config.output_dir) / f"batch_results_{config.dataset_split}.json"
            save_json(str(batch_results_file), results)
            logger.info(f"\n💾 Batch results saved to: {batch_results_file}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Interrupted by user")
        return 130
        
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())



