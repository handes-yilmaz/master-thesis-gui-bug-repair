"""
Main GUIRepair Workflow - Orchestrates the 3 Phases
"""
import logging
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

import sys
from pathlib import Path as PathLib
sys.path.insert(0, str(PathLib(__file__).parent.parent))

from config import GUIRepairConfig
from utils import (
    read_json, save_json, save_file,
    LLMClient, JavaScriptParser, RepoManager
)
from utils.file_io import load_images_for_multimodal
from core.localization import FileLocalizer, ElementLocalizer, LocalizationResult
from core.patch_generation import PatchGenerator, PatchResult


logger = logging.getLogger(__name__)


class GUIRepairWorkflow:
    """
    Main workflow orchestrator for GUIRepair 3-phase approach
    
    Coordinates:
    - Phase 1: File Localization
    - Phase 2: Element Localization  
    - Phase 3: Patch Generation
    """
    
    def __init__(self, config: GUIRepairConfig):
        """
        Initialize GUIRepair workflow
        
        Args:
            config: Framework configuration
        """
        self.config = config
        
        # Determine LLM provider from model name
        if 'gpt' in config.model_name.lower() or 'o4' in config.model_name.lower():
            provider = 'openai'
        elif 'claude' in config.model_name.lower():
            provider = 'claude'
        else:
            provider = 'openai'  # Default
        
        # Initialize LLM client
        self.llm = LLMClient(
            provider=provider,
            api_key=config.api_key,
            model=config.model_name,
            wait_time=config.wait_time_after_api
        )
        
        # Initialize parser
        self.parser = JavaScriptParser()
    
    def run(
        self,
        instance_id: str,
        bug_report: str,
        repo_path: str,
        output_dir: str
    ) -> Dict:
        """
        Run complete 3-phase workflow for a single instance
        
        Args:
            instance_id: Instance identifier (e.g., "bpmn-io__bpmn-js-1080")
            bug_report: Bug report text
            repo_path: Path to repository
            output_dir: Where to save results
            
        Returns:
            Results dict with patches, metadata, and statistics
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"🚀 GUIRepair Workflow: {instance_id}")
        logger.info(f"{'='*70}\n")
        
        start_time = datetime.now()
        
        # Setup paths
        repo_path = Path(repo_path)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize managers
        repo_manager = RepoManager(str(repo_path))
        
        # Load images if multimodal mode
        images = None
        if self.config.enable_images:
            image_dir = self._get_image_dir(instance_id)
            if image_dir and image_dir.exists():
                provider = 'openai' if 'gpt' in self.config.model_name.lower() else 'claude'
                images = load_images_for_multimodal(str(image_dir), provider)
                logger.info(f"📷 Loaded {len(images)} images")
        
        # Clean repository
        logger.info("🧹 Cleaning repository...")
        repo_manager.clean_repo()
        
        # Phase 1: File Localization
        file_localizer = FileLocalizer(
            self.config,
            self.llm,
            repo_manager,
            output_path
        )
        
        bug_files, file_metadata = file_localizer.localize(
            bug_report,
            images
        )
        
        if not bug_files:
            logger.error("❌ No bug files identified!")
            return self._create_failure_result(instance_id, "No bug files found")
        
        # Phase 2: Element Localization
        element_localizer = ElementLocalizer(
            self.config,
            self.llm,
            self.parser,
            repo_manager,
            output_path
        )
        
        bug_elements, element_metadata = element_localizer.localize(
            bug_report,
            bug_files,
            images
        )
        
        # Create localization result
        loc_result = LocalizationResult(
            bug_files,
            bug_elements,
            file_metadata,
            element_metadata
        )
        
        logger.info(f"\n{loc_result.summary()}")
        
        # Get code context for patch generation
        code_context = loc_result.get_code_context()
        
        # Phase 3: Patch Generation
        patch_generator = PatchGenerator(
            self.config,
            self.llm,
            repo_path,
            output_path
        )
        
        unified_diff, patch_metadata = patch_generator.generate(
            bug_report,
            code_context,
            images
        )
        
        # Create patch result
        patch_result = PatchResult(unified_diff, patch_metadata)
        
        logger.info(f"\n{patch_result.summary()}")
        
        # Calculate total time
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Compile results
        results = {
            'instance_id': instance_id,
            'success': patch_result.success,
            'unified_diff': unified_diff,
            'bug_files': bug_files,
            'modified_files': patch_result.get_modified_files(),
            'duration_seconds': duration,
            'total_tokens': (
                file_metadata['token_usage']['total_tokens'] +
                element_metadata['token_usage']['total_tokens'] +
                patch_metadata['token_usage']['total_tokens']
            ),
            'phase_breakdown': {
                'file_localization': file_metadata,
                'element_localization': element_metadata,
                'patch_generation': patch_metadata
            }
        }
        
        # Save final results
        save_json(
            str(output_path / "workflow_results.json"),
            results
        )
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✅ Workflow completed in {duration:.1f} seconds")
        logger.info(f"   Total tokens: {results['total_tokens']}")
        logger.info(f"   Patch generated: {patch_result.success}")
        logger.info(f"{'='*70}\n")
        
        return results
    
    def _get_image_dir(self, instance_id: str) -> Optional[Path]:
        """
        Get image directory for instance
        
        Args:
            instance_id: Instance identifier
            
        Returns:
            Path to image directory or None
        """
        # Expected path: repo_path/split/repo/instance_id/IMAGE
        repo_prefix = instance_id.split('__')[0]
        
        image_dir = Path(self.config.repo_path) / self.config.dataset_split / repo_prefix / instance_id / "IMAGE"
        
        return image_dir if image_dir.exists() else None
    
    def _create_failure_result(self, instance_id: str, reason: str) -> Dict:
        """Create result dict for failed runs"""
        return {
            'instance_id': instance_id,
            'success': False,
            'unified_diff': '',
            'error': reason,
            'total_tokens': 0
        }
    
    def validate_patch(self, patch_file: str, repo_path: str) -> bool:
        """
        Validate that a patch can be applied (for testing)
        
        Args:
            patch_file: Path to patch file
            repo_path: Path to repository
            
        Returns:
            True if patch applies cleanly
        """
        import subprocess
        
        try:
            # Try to apply patch
            result = subprocess.run(
                ['git', 'apply', '--check', patch_file],
                cwd=repo_path,
                capture_output=True
            )
            
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Error validating patch: {e}")
            return False


class WorkflowRunner:
    """
    Helper class to run workflows on multiple instances
    """
    
    def __init__(self, config: GUIRepairConfig):
        self.config = config
        self.workflow = GUIRepairWorkflow(config)
    
    def run_instance(
        self,
        instance_id: str,
        dataset: Dict
    ) -> Dict:
        """
        Run workflow for a single instance from dataset
        
        Args:
            instance_id: Instance identifier
            dataset: Dataset dict with instance info
            
        Returns:
            Results dict
        """
        if instance_id not in dataset:
            logger.error(f"Instance {instance_id} not found in dataset")
            return {'instance_id': instance_id, 'success': False, 'error': 'Not in dataset'}
        
        instance_data = dataset[instance_id]
        
        # Get paths
        repo_name = instance_data['repo'].split('/')[-1]
        repo_prefix = instance_id.split('__')[0]
        
        repo_path = Path(self.config.repo_path) / self.config.dataset_split / repo_prefix / instance_id / "REPO" / repo_name
        output_dir = Path(self.config.output_dir) / self.config.dataset_split / repo_prefix / instance_id
        
        # Verify repo exists
        if not repo_path.exists():
            logger.error(f"Repository not found: {repo_path}")
            return {'instance_id': instance_id, 'success': False, 'error': 'Repo not found'}
        
        # Run workflow
        return self.workflow.run(
            instance_id=instance_id,
            bug_report=instance_data['problem_statement'],
            repo_path=str(repo_path),
            output_dir=str(output_dir)
        )
    
    def run_batch(
        self,
        instance_ids: List[str],
        dataset: Dict
    ) -> Dict[str, Dict]:
        """
        Run workflow for multiple instances
        
        Args:
            instance_ids: List of instance IDs
            dataset: Dataset dict
            
        Returns:
            Dict mapping instance IDs to results
        """
        results = {}
        
        for i, instance_id in enumerate(instance_ids, 1):
            logger.info(f"\n{'='*70}")
            logger.info(f"Processing {i}/{len(instance_ids)}: {instance_id}")
            logger.info(f"{'='*70}")
            
            try:
                result = self.run_instance(instance_id, dataset)
                results[instance_id] = result
            except Exception as e:
                logger.error(f"Error processing {instance_id}: {e}")
                results[instance_id] = {
                    'instance_id': instance_id,
                    'success': False,
                    'error': str(e)
                }
        
        return results

