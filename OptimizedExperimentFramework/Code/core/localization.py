"""
Phase 1 & 2: File and Element Localization
Identifies bug-related files and specific code elements
"""
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import GUIRepairConfig
from utils import (
    read_file, save_json,
    LLMClient, JavaScriptParser, RepoManager
)
from prompts import FileLocalizationPrompts, ElementLocalizationPrompts


logger = logging.getLogger(__name__)


class FileLocalizer:
    """
    Phase 1: File Localization
    Identifies which files in the repository are related to the bug
    """
    
    def __init__(
        self,
        config: GUIRepairConfig,
        llm_client: LLMClient,
        repo_manager: RepoManager,
        output_dir: Path
    ):
        """
        Initialize file localizer
        
        Args:
            config: Framework configuration
            llm_client: LLM client for API calls
            repo_manager: Repository manager
            output_dir: Directory to save outputs
        """
        self.config = config
        self.llm = llm_client
        self.repo = repo_manager
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def localize(
        self,
        bug_report: str,
        images: List[Dict] = None
    ) -> Tuple[List[str], Dict]:
        """
        Identify bug-related files
        
        Args:
            bug_report: The bug report text
            images: Optional images for multimodal
            
        Returns:
            (bug_file_paths, metadata)
        """
        logger.info("🔍 Phase 1: File Localization started")
        
        # Step 1: Get repository structure
        logger.info("  📁 Analyzing repository structure...")
        repo_structure = self.repo.get_repo_structure()
        formatted_structure = self.repo.format_structure_for_llm(repo_structure)
        
        # Save structure for debugging
        save_json(
            str(self.output_dir / "1-1_repo_structure.json"),
            repo_structure
        )
        
        # Step 2: Generate prompts
        logger.info("  💬 Generating file localization prompts...")
        system_prompt = FileLocalizationPrompts.create_system_prompt(
            provider=self.config.model_name
        )
        user_prompt = FileLocalizationPrompts.create_user_prompt(
            bug_report,
            formatted_structure,
            images
        )
        
        # Step 3: Call LLM to identify files
        logger.info("  🤖 Calling LLM for file identification...")
        response_format = FileLocalizationPrompts.get_response_format()
        
        results, token_usage = self.llm.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.config.file_localization_temp,
            response_format=response_format,
            num_samples=self.config.file_localization_samples
        )
        
        # Step 4: Extract and validate file paths
        bug_files = self._extract_file_paths(results, repo_structure)
        
        logger.info(f"  ✅ Identified {len(bug_files)} bug-related files")
        
        # Save results
        save_json(
            str(self.output_dir / "1-2_bug_files.json"),
            results
        )
        
        metadata = {
            'token_usage': token_usage.model_dump(),
            'num_files': len(bug_files),
            'samples': self.config.file_localization_samples
        }
        
        return bug_files, metadata
    
    def _extract_file_paths(
        self,
        llm_results: Dict,
        repo_structure: Dict
    ) -> List[str]:
        """
        Extract and validate file paths from LLM response
        
        Args:
            llm_results: LLM response dict
            repo_structure: Repository structure for validation
            
        Returns:
            List of valid bug file paths
        """
        # Get all valid files from repo
        all_files = self.repo.get_file_list(repo_structure)
        
        # Collect files from all samples
        bug_files_set = set()
        
        for sample_key, sample_data in llm_results.items():
            if isinstance(sample_data, dict) and 'bug_files' in sample_data:
                for file_path in sample_data['bug_files']:
                    # Clean the path
                    cleaned_path = file_path.strip().lstrip('/')
                    
                    # Validate against repo structure
                    if self._is_valid_file(cleaned_path, all_files):
                        bug_files_set.add(cleaned_path)
                    else:
                        # Try fuzzy matching
                        matched = self._fuzzy_match_file(cleaned_path, all_files)
                        if matched:
                            bug_files_set.add(matched)
                            logger.debug(f"  Fuzzy matched: {cleaned_path} → {matched}")
        
        return sorted(bug_files_set)
    
    def _is_valid_file(self, file_path: str, all_files: List[str]) -> bool:
        """Check if file exists in repo"""
        return file_path in all_files
    
    def _fuzzy_match_file(self, file_path: str, all_files: List[str]) -> Optional[str]:
        """Try to match file path fuzzily (handles path variations)"""
        # Try exact match first
        if file_path in all_files:
            return file_path
        
        # Try removing leading path components
        parts = file_path.split('/')
        for i in range(len(parts)):
            partial = '/'.join(parts[i:])
            for repo_file in all_files:
                if repo_file.endswith(partial):
                    return repo_file
        
        # Try matching just the filename
        filename = parts[-1]
        matches = [f for f in all_files if f.endswith(filename)]
        if len(matches) == 1:
            return matches[0]
        
        return None


class ElementLocalizer:
    """
    Phase 2: Element Localization
    Identifies specific classes and functions within bug files
    """
    
    def __init__(
        self,
        config: GUIRepairConfig,
        llm_client: LLMClient,
        parser: JavaScriptParser,
        repo_manager: RepoManager,
        output_dir: Path
    ):
        """
        Initialize element localizer
        
        Args:
            config: Framework configuration
            llm_client: LLM client for API calls
            parser: JavaScript parser
            repo_manager: Repository manager
            output_dir: Directory to save outputs
        """
        self.config = config
        self.llm = llm_client
        self.parser = parser
        self.repo = repo_manager
        self.output_dir = output_dir
    
    def localize(
        self,
        bug_report: str,
        bug_files: List[str],
        images: List[Dict] = None
    ) -> Tuple[Dict[str, Dict], Dict]:
        """
        Identify bug elements (classes/functions) in files
        
        Args:
            bug_report: The bug report text
            bug_files: List of bug-related file paths
            images: Optional images for multimodal
            
        Returns:
            (bug_elements_by_file, metadata)
        """
        logger.info("🔍 Phase 2: Element Localization started")
        
        # Step 1: Parse and compress files
        logger.info(f"  📄 Parsing {len(bug_files)} files...")
        compressed_files = self._compress_bug_files(bug_files)
        
        # Save compressed files
        save_json(
            str(self.output_dir / "2-1_compressed_bug_files.json"),
            compressed_files
        )
        
        # Step 2: Refine to key files if too many
        if len(compressed_files) > self.config.max_candidate_files:
            logger.info(f"  🔧 Refining to {self.config.max_candidate_files} key files...")
            compressed_files = self._refine_to_key_files(
                bug_report,
                compressed_files
            )
            save_json(
                str(self.output_dir / "2-2_key_bug_files.json"),
                compressed_files
            )
        
        # Step 3: Generate prompts
        logger.info("  💬 Generating element localization prompts...")
        system_prompt = ElementLocalizationPrompts.create_system_prompt(
            provider=self.config.model_name
        )
        
        # Add repo-specific hints
        repo_name = str(self.repo.repo_path.name)
        hint = ElementLocalizationPrompts.add_repo_specific_hints(repo_name)
        if hint:
            system_prompt += hint
        
        user_prompt = ElementLocalizationPrompts.create_user_prompt(
            bug_report,
            compressed_files,
            images
        )
        
        # Step 4: Call LLM to identify elements
        logger.info("  🤖 Calling LLM for element identification...")
        response_format = ElementLocalizationPrompts.get_response_format()
        
        results, token_usage = self.llm.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.config.element_localization_temp,
            response_format=response_format,
            num_samples=self.config.element_localization_samples
        )
        
        # Save results
        save_json(
            str(self.output_dir / "3-1_bug_elements.json"),
            results
        )
        
        # Step 5: Extract elements and get code context
        bug_elements = self._extract_elements(results, compressed_files)
        
        logger.info(f"  ✅ Identified elements in {len(bug_elements)} files")
        
        metadata = {
            'token_usage': token_usage.model_dump(),
            'num_files': len(bug_elements),
            'samples': self.config.element_localization_samples
        }
        
        return bug_elements, metadata
    
    def _compress_bug_files(self, bug_files: List[str]) -> Dict[str, str]:
        """
        Parse and compress bug files for LLM consumption
        
        Args:
            bug_files: List of file paths
            
        Returns:
            Dict of file path to compressed content
        """
        compressed = {}
        
        for file_path in bug_files:
            full_path = self.repo.repo_path / file_path
            
            if not full_path.exists():
                logger.warning(f"  ⚠️  File not found: {file_path}")
                continue
            
            # Parse file
            classes, functions, file_lines = self.parser.parse_file(str(full_path))
            
            # Compress
            compressed_content = self.parser.compress_code(
                file_lines,
                max_lines=self.config.max_lines_per_file
            )
            
            # Add structure info
            structure_summary = self.parser.get_file_structure_summary(
                classes, functions
            )
            
            compressed[file_path] = f"{structure_summary}\n\n{compressed_content}"
        
        return compressed
    
    def _refine_to_key_files(
        self,
        bug_report: str,
        compressed_files: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Refine file list to top candidates when too many files
        
        Args:
            bug_report: Bug report text
            compressed_files: All compressed files
            
        Returns:
            Refined dict of key files
        """
        # Generate refinement prompt
        user_prompt = FileLocalizationPrompts.create_refinement_prompt(
            bug_report,
            compressed_files,
            max_files=self.config.max_candidate_files
        )
        
        # Call LLM
        response_format = FileLocalizationPrompts.get_response_format()
        results, _ = self.llm.chat(
            system_prompt=FileLocalizationPrompts.create_system_prompt(),
            user_prompt=user_prompt,
            temperature=self.config.file_localization_temp,
            response_format=response_format,
            num_samples=1
        )
        
        # Extract refined files
        refined_files = {}
        for sample_data in results.values():
            if isinstance(sample_data, dict) and 'bug_files' in sample_data:
                for file_path in sample_data['bug_files']:
                    if file_path in compressed_files:
                        refined_files[file_path] = compressed_files[file_path]
        
        return refined_files
    
    def _extract_elements(
        self,
        llm_results: Dict,
        compressed_files: Dict[str, str]
    ) -> Dict[str, Dict]:
        """
        Extract class/function elements and get full code context
        
        Args:
            llm_results: LLM response with bug_classes and bug_functions
            compressed_files: Compressed file contents
            
        Returns:
            Dict mapping file path to elements with code context
        """
        # Collect all mentioned classes and functions
        all_classes = set()
        all_functions = set()
        
        for sample_data in llm_results.values():
            if isinstance(sample_data, dict):
                all_classes.update(sample_data.get('bug_classes', []))
                all_functions.update(sample_data.get('bug_functions', []))
        
        # Get code context for each element
        elements_by_file = {}
        
        for file_path in compressed_files.keys():
            full_path = self.repo.repo_path / file_path
            
            # Parse file to get actual elements
            classes, functions, file_lines = self.parser.parse_file(str(full_path))
            
            # Find matching elements
            matched_elements = {
                'classes': [],
                'functions': [],
                'code_snippets': []
            }
            
            # Match classes
            for cls in classes:
                if cls['name'] in all_classes or any(
                    cls_name in cls['name'] for cls_name in all_classes
                ):
                    code_snippet = self.parser.extract_element_code(
                        file_lines, cls, context_lines=5
                    )
                    matched_elements['classes'].append({
                        'name': cls['name'],
                        'lines': f"{cls['start_line']}-{cls['end_line']}",
                        'code': code_snippet
                    })
            
            # Match functions
            for func in functions:
                if func['name'] in all_functions or any(
                    func_name in func['name'] for func_name in all_functions
                ):
                    code_snippet = self.parser.extract_element_code(
                        file_lines, func, context_lines=5
                    )
                    matched_elements['functions'].append({
                        'name': func['name'],
                        'lines': f"{func['start_line']}-{func['end_line']}",
                        'code': code_snippet
                    })
            
            # If no specific elements matched, include whole file (compressed)
            if not matched_elements['classes'] and not matched_elements['functions']:
                compressed_code = self.parser.compress_code(
                    file_lines,
                    max_lines=self.config.max_lines_per_file
                )
                matched_elements['code_snippets'].append(compressed_code)
            
            elements_by_file[file_path] = matched_elements
        
        # Save elements
        save_json(
            str(self.output_dir / "3-2_bug_elements_with_context.json"),
            elements_by_file
        )
        
        return elements_by_file


class LocalizationResult:
    """
    Result container for localization phases
    Stores file paths, elements, and metadata
    """
    
    def __init__(
        self,
        bug_files: List[str],
        bug_elements: Dict[str, Dict],
        file_metadata: Dict,
        element_metadata: Dict
    ):
        self.bug_files = bug_files
        self.bug_elements = bug_elements
        self.file_metadata = file_metadata
        self.element_metadata = element_metadata
        self.total_tokens = (
            file_metadata['token_usage']['total_tokens'] +
            element_metadata['token_usage']['total_tokens']
        )
    
    def get_code_context(self) -> Dict[str, str]:
        """
        Get formatted code context for patch generation
        
        Returns:
            Dict mapping file paths to formatted code snippets
        """
        code_context = {}
        
        for file_path, elements in self.bug_elements.items():
            snippets = []
            
            # Add class code
            for cls in elements.get('classes', []):
                snippets.append(f"// Class: {cls['name']} (lines {cls['lines']})")
                snippets.append(cls['code'])
                snippets.append("")
            
            # Add function code
            for func in elements.get('functions', []):
                snippets.append(f"// Function: {func['name']} (lines {func['lines']})")
                snippets.append(func['code'])
                snippets.append("")
            
            # Add generic code snippets
            for snippet in elements.get('code_snippets', []):
                snippets.append(snippet)
            
            code_context[file_path] = '\n'.join(snippets)
        
        return code_context
    
    def summary(self) -> str:
        """Get human-readable summary"""
        return f"""Localization Results:
  Files: {len(self.bug_files)}
  Elements: {sum(len(e.get('classes', [])) + len(e.get('functions', [])) for e in self.bug_elements.values())}
  Tokens: {self.total_tokens}
"""

