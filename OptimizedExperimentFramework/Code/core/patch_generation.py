"""
Phase 3: Patch Generation
Generates code patches to fix the identified bugs
"""
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import GUIRepairConfig
from utils import (
    read_file, save_file, save_json,
    LLMClient, DiffUtils
)
from prompts import PatchGenerationPrompts


logger = logging.getLogger(__name__)


class PatchGenerator:
    """
    Phase 3: Patch Generation
    Generates SEARCH/REPLACE patches to fix bugs
    """
    
    def __init__(
        self,
        config: GUIRepairConfig,
        llm_client: LLMClient,
        repo_path: Path,
        output_dir: Path
    ):
        """
        Initialize patch generator
        
        Args:
            config: Framework configuration
            llm_client: LLM client for API calls
            repo_path: Path to repository
            output_dir: Directory to save outputs
        """
        self.config = config
        self.llm = llm_client
        self.repo_path = repo_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(
        self,
        bug_report: str,
        code_context: Dict[str, str],
        images: List[Dict] = None
    ) -> Tuple[str, Dict]:
        """
        Generate patches for the bug
        
        Args:
            bug_report: The bug report text
            code_context: Dict mapping file paths to code snippets
            images: Optional images for multimodal
            
        Returns:
            (unified_diff, metadata)
        """
        logger.info("🔧 Phase 3: Patch Generation started")
        
        # Step 1: Generate prompts
        logger.info("  💬 Generating patch generation prompts...")
        system_prompt = PatchGenerationPrompts.create_system_prompt()
        user_prompt = PatchGenerationPrompts.create_user_prompt(
            bug_report,
            code_context,
            images
        )
        
        # Step 2: Call LLM to generate patches
        logger.info("  🤖 Calling LLM for patch generation...")
        
        results, token_usage = self.llm.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.config.patch_generation_temp,
            response_format=None,  # Unstructured - raw patch text
            num_samples=self.config.patch_generation_samples
        )
        
        # Save raw LLM outputs
        save_json(
            str(self.output_dir / "4-1_patch_llm_outputs.json"),
            results
        )
        
        # Step 3: Parse and apply patches
        logger.info("  🔍 Parsing SEARCH/REPLACE blocks...")
        all_patches = self._parse_all_patches(results)
        
        # Save parsed patches
        save_json(
            str(self.output_dir / "4-2_parsed_patches.json"),
            all_patches
        )
        
        # Step 4: Apply best patch
        logger.info("  ⚙️  Applying patches...")
        unified_diff = self._apply_patches(all_patches)
        
        # Save diff
        diff_path = self.output_dir / "changes.diff"
        save_file(str(diff_path), unified_diff)
        
        if unified_diff.strip():
            logger.info(f"  ✅ Patch generated: {len(unified_diff)} bytes")
        else:
            logger.warning("  ⚠️  No valid patches generated")
        
        metadata = {
            'token_usage': token_usage.model_dump(),
            'num_samples': self.config.patch_generation_samples,
            'patch_size': len(unified_diff),
            'files_modified': len(all_patches) if all_patches else 0
        }
        
        return unified_diff, metadata
    
    def _parse_all_patches(self, llm_results: Dict) -> Dict[str, List[Dict]]:
        """
        Parse patches from all LLM samples
        
        Args:
            llm_results: Dict of sample results
            
        Returns:
            Dict mapping file paths to list of patches
        """
        all_patches = {}
        
        for sample_key, llm_output in llm_results.items():
            if not isinstance(llm_output, str):
                continue
            
            # Validate format
            valid, errors = DiffUtils.validate_patch_format(llm_output)
            if not valid:
                logger.warning(f"  ⚠️  Sample {sample_key} has invalid format: {errors}")
                continue
            
            # Extract patches
            patches_by_file = DiffUtils.extract_patches_from_llm_text(llm_output)
            
            # Merge with all_patches (first valid patch wins)
            for file_path, patches in patches_by_file.items():
                if file_path not in all_patches:
                    all_patches[file_path] = patches
        
        return all_patches
    
    def _apply_patches(self, patches_by_file: Dict[str, List[Dict]]) -> str:
        """
        Apply patches to files and generate unified diff
        
        Args:
            patches_by_file: Dict mapping file paths to patches
            
        Returns:
            Unified diff string
        """
        if not patches_by_file:
            return ""
        
        all_diffs = []
        
        for file_path, patches in patches_by_file.items():
            # Read original file
            full_path = self.repo_path / file_path
            
            if not full_path.exists():
                logger.warning(f"  ⚠️  File not found for patching: {file_path}")
                continue
            
            original_content = read_file(str(full_path))
            
            # Apply all patches for this file
            success, modified_content, errors = DiffUtils.apply_patches_to_file(
                file_path,
                original_content,
                patches
            )
            
            if not success:
                logger.warning(f"  ⚠️  Some patches failed for {file_path}:")
                for error in errors:
                    logger.warning(f"     {error}")
            
            # Generate diff if content changed
            if original_content != modified_content:
                diff = DiffUtils.generate_unified_diff(
                    file_path,
                    original_content,
                    modified_content
                )
                all_diffs.append(diff)
                logger.info(f"  ✅ Patched: {file_path}")
        
        return '\n'.join(all_diffs)
    
    def validate_and_refine(
        self,
        original_patch: str,
        error_message: str
    ) -> Optional[str]:
        """
        Attempt to fix a failed patch by asking LLM to refine it
        
        Args:
            original_patch: The patch that failed
            error_message: Why it failed
            
        Returns:
            Refined patch or None
        """
        logger.info("  🔄 Attempting patch refinement...")
        
        # Create validation prompt
        validation_prompt = PatchGenerationPrompts.create_patch_validation_prompt(
            original_patch,
            error_message
        )
        
        # Call LLM
        results, _ = self.llm.chat(
            system_prompt="You are an expert at fixing code patches. Generate a corrected SEARCH/REPLACE block.",
            user_prompt=validation_prompt,
            temperature=0.0,
            num_samples=1
        )
        
        # Extract refined patch
        for sample_data in results.values():
            if isinstance(sample_data, str):
                return sample_data
        
        return None


class PatchResult:
    """
    Result container for patch generation
    Stores patches, diffs, and metadata
    """
    
    def __init__(
        self,
        unified_diff: str,
        metadata: Dict,
        patches_by_file: Dict[str, List[Dict]] = None
    ):
        self.unified_diff = unified_diff
        self.metadata = metadata
        self.patches_by_file = patches_by_file or {}
        self.success = bool(unified_diff.strip())
    
    def get_modified_files(self) -> List[str]:
        """Get list of files that were modified"""
        return list(self.patches_by_file.keys())
    
    def summary(self) -> str:
        """Get human-readable summary"""
        status = "✅ SUCCESS" if self.success else "❌ NO PATCHES"
        return f"""Patch Generation Results: {status}
  Modified files: {len(self.get_modified_files())}
  Diff size: {len(self.unified_diff)} bytes
  Tokens: {self.metadata['token_usage']['total_tokens']}
"""

