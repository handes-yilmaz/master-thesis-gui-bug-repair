"""
Diff and patch utilities
Handles SEARCH/REPLACE format parsing and diff generation
"""
import re
from typing import List, Dict, Tuple, Optional


class DiffUtils:
    """
    Handle patch operations:
    - Parse SEARCH/REPLACE blocks from LLM output
    - Apply patches to code
    - Generate unified diffs
    """
    
    @staticmethod
    def parse_search_replace_blocks(llm_output: str) -> List[Dict[str, any]]:
        """
        Parse SEARCH/REPLACE blocks from LLM output
        
        Expected format:
        ```
        ### filename.js
        <<<<<<< SEARCH
        old code
        =======
        new code
        >>>>>>> REPLACE
        ```
        
        Args:
            llm_output: Raw LLM output containing patches
            
        Returns:
            List of dicts with 'file', 'search', 'replace' keys
        """
        patches = []
        
        # Find all file blocks
        file_pattern = r'###\s+([^\n]+)\n'
        search_replace_pattern = r'<+\s*SEARCH\s*\n(.*?)\n\s*=+\s*\n(.*?)\n\s*>+\s*REPLACE'
        
        # Split into file sections
        sections = re.split(file_pattern, llm_output)
        
        # Process pairs of (filename, content)
        for i in range(1, len(sections), 2):
            if i + 1 >= len(sections):
                break
                
            filename = sections[i].strip()
            content = sections[i + 1]
            
            # Find all SEARCH/REPLACE blocks in this file
            matches = re.finditer(
                search_replace_pattern,
                content,
                re.DOTALL | re.MULTILINE
            )
            
            for match in matches:
                search_block = match.group(1).strip()
                replace_block = match.group(2).strip()
                
                patches.append({
                    'file': filename,
                    'search': search_block,
                    'replace': replace_block
                })
        
        return patches
    
    @staticmethod
    def apply_search_replace(
        file_content: str,
        search: str,
        replace: str
    ) -> Tuple[bool, str]:
        """
        Apply a single SEARCH/REPLACE operation
        
        Args:
            file_content: Original file content
            search: Text to search for
            replace: Text to replace with
            
        Returns:
            (success, new_content)
        """
        # Normalize whitespace for matching
        def normalize_whitespace(text: str) -> str:
            """Normalize whitespace while preserving structure"""
            lines = text.split('\n')
            normalized = []
            for line in lines:
                # Preserve indentation structure
                stripped = line.lstrip()
                if stripped:
                    indent_level = len(line) - len(stripped)
                    normalized.append(' ' * indent_level + stripped)
                else:
                    normalized.append('')
            return '\n'.join(normalized)
        
        # Try exact match first
        if search in file_content:
            new_content = file_content.replace(search, replace, 1)
            return True, new_content
        
        # Try with normalized whitespace
        normalized_search = normalize_whitespace(search)
        normalized_content = normalize_whitespace(file_content)
        
        if normalized_search in normalized_content:
            # Find the original text matching the normalized search
            # This is complex, so for now just use the normalized version
            new_content = normalized_content.replace(normalized_search, replace, 1)
            return True, new_content
        
        return False, file_content
    
    @staticmethod
    def generate_unified_diff(
        file_path: str,
        original: str,
        modified: str
    ) -> str:
        """
        Generate unified diff format
        
        Args:
            file_path: Path to file
            original: Original content
            modified: Modified content
            
        Returns:
            Unified diff string
        """
        from difflib import unified_diff
        
        original_lines = original.splitlines(keepends=True)
        modified_lines = modified.splitlines(keepends=True)
        
        diff = unified_diff(
            original_lines,
            modified_lines,
            fromfile=f'a/{file_path}',
            tofile=f'b/{file_path}',
            lineterm=''
        )
        
        return ''.join(diff)
    
    @staticmethod
    def extract_patches_from_llm_text(text: str) -> Dict[str, List[Dict]]:
        """
        Extract all patches from LLM text output
        Handles multiple formats and edge cases
        
        Args:
            text: Raw LLM output
            
        Returns:
            Dict mapping filename to list of patch dicts
        """
        # Try SEARCH/REPLACE format first
        search_replace_patches = DiffUtils.parse_search_replace_blocks(text)
        
        # Organize by file
        patches_by_file = {}
        for patch in search_replace_patches:
            file = patch['file']
            if file not in patches_by_file:
                patches_by_file[file] = []
            patches_by_file[file].append({
                'search': patch['search'],
                'replace': patch['replace']
            })
        
        return patches_by_file
    
    @staticmethod
    def apply_patches_to_file(
        file_path: str,
        original_content: str,
        patches: List[Dict]
    ) -> Tuple[bool, str, List[str]]:
        """
        Apply multiple patches to a file
        
        Args:
            file_path: Path to file
            original_content: Original file content
            patches: List of patch dicts with 'search' and 'replace'
            
        Returns:
            (all_successful, final_content, error_messages)
        """
        content = original_content
        errors = []
        all_successful = True
        
        for i, patch in enumerate(patches, 1):
            success, content = DiffUtils.apply_search_replace(
                content,
                patch['search'],
                patch['replace']
            )
            
            if not success:
                errors.append(
                    f"Patch {i}/{len(patches)} failed: "
                    f"Could not find search text in {file_path}"
                )
                all_successful = False
        
        return all_successful, content, errors
    
    @staticmethod
    def validate_patch_format(llm_output: str) -> Tuple[bool, List[str]]:
        """
        Validate that LLM output contains properly formatted patches
        
        Args:
            llm_output: Raw LLM output
            
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # Check for file markers
        if '###' not in llm_output:
            errors.append("No file markers (###) found")
        
        # Check for SEARCH/REPLACE blocks
        if '<<<<<<< SEARCH' not in llm_output or '>>>>>>> REPLACE' not in llm_output:
            errors.append("No SEARCH/REPLACE blocks found")
        
        # Check for balance
        search_count = llm_output.count('<<<<<<< SEARCH')
        replace_count = llm_output.count('>>>>>>> REPLACE')
        separator_count = llm_output.count('=======')
        
        if search_count != replace_count or search_count != separator_count:
            errors.append(
                f"Unbalanced blocks: {search_count} SEARCH, "
                f"{separator_count} separators, {replace_count} REPLACE"
            )
        
        return len(errors) == 0, errors



