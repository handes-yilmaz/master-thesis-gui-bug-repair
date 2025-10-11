"""
Repository management utilities
Handles git operations, repo structure analysis, and file operations
"""
import subprocess
import os
from pathlib import Path
from typing import Dict, List, Optional


class RepoManager:
    """
    Manage repository operations:
    - Clean/reset repo state
    - Get repository structure  
    - Apply patches
    - Run build commands
    """
    
    def __init__(self, repo_path: str):
        """
        Initialize repository manager
        
        Args:
            repo_path: Path to repository root
        """
        self.repo_path = Path(repo_path)
        if not self.repo_path.exists():
            raise ValueError(f"Repository path does not exist: {repo_path}")
    
    def clean_repo(self, remove_untracked: bool = True) -> bool:
        """
        Clean repository to pristine state
        
        Args:
            remove_untracked: Whether to remove untracked files
            
        Returns:
            True if successful
        """
        try:
            # Reset any changes
            subprocess.run(
                ["git", "reset", "--hard", "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                check=True
            )
            
            # Remove untracked files if requested
            if remove_untracked:
                subprocess.run(
                    ["git", "clean", "-fd"],
                    cwd=self.repo_path,
                    capture_output=True,
                    check=True
                )
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Error cleaning repo: {e.stderr.decode()}")
            return False
    
    def get_repo_structure(
        self,
        excluded_dirs: Optional[List[str]] = None,
        file_extensions: Optional[List[str]] = None,
        max_depth: int = 10
    ) -> Dict:
        """
        Get repository directory structure
        
        Args:
            excluded_dirs: Directories to exclude (e.g., node_modules, dist)
            file_extensions: Only include these extensions (e.g., ['.js', '.jsx'])
            max_depth: Maximum directory depth to traverse
            
        Returns:
            Nested dictionary representing repo structure
        """
        if excluded_dirs is None:
            excluded_dirs = {
                'node_modules', 'dist', 'build', 'out', 
                '.git', '__pycache__', 'test', 'tests',
                '__tests__', 'coverage'
            }
        
        if file_extensions is None:
            file_extensions = ['.js', '.jsx', '.ts', '.tsx']
        
        def _walk_dir(path: Path, depth: int) -> Dict:
            """Recursively walk directory"""
            if depth > max_depth:
                return {}
            
            structure = {}
            
            try:
                for item in sorted(path.iterdir()):
                    # Skip excluded directories
                    if item.is_dir() and item.name in excluded_dirs:
                        continue
                    
                    # Handle directories
                    if item.is_dir():
                        substructure = _walk_dir(item, depth + 1)
                        if substructure:  # Only add if not empty
                            structure[item.name] = substructure
                    
                    # Handle files
                    elif item.is_file():
                        if any(item.name.endswith(ext) for ext in file_extensions):
                            structure[item.name] = "file"
            
            except PermissionError:
                pass  # Skip directories we can't read
            
            return structure
        
        return _walk_dir(self.repo_path, 0)
    
    def format_structure_for_llm(self, structure: Dict, indent: int = 0) -> str:
        """
        Format repository structure for LLM consumption
        
        Args:
            structure: Nested dict from get_repo_structure()
            indent: Current indentation level
            
        Returns:
            Pretty-printed structure string
        """
        lines = []
        
        for name, content in structure.items():
            if isinstance(content, dict):
                # Directory
                lines.append(' ' * indent + f'{name}/')
                lines.append(self.format_structure_for_llm(content, indent + 2))
            else:
                # File
                lines.append(' ' * indent + name)
        
        return '\n'.join(lines)
    
    def get_file_list(
        self,
        structure: Optional[Dict] = None,
        prefix: str = ""
    ) -> List[str]:
        """
        Get flat list of all files from structure
        
        Args:
            structure: Repo structure dict (if None, generates it)
            prefix: Current path prefix
            
        Returns:
            List of relative file paths
        """
        if structure is None:
            structure = self.get_repo_structure()
        
        files = []
        
        for name, content in structure.items():
            if isinstance(content, dict):
                # Directory - recurse
                files.extend(
                    self.get_file_list(content, prefix + name + '/')
                )
            else:
                # File
                files.append(prefix + name)
        
        return files
    
    def apply_patch(self, patch_content: str, file_path: str) -> bool:
        """
        Apply patch to a specific file
        
        Args:
            patch_content: Patch diff content
            file_path: Relative path to file in repo
            
        Returns:
            True if patch applied successfully
        """
        full_path = self.repo_path / file_path
        
        if not full_path.exists():
            print(f"Error: File not found: {file_path}")
            return False
        
        try:
            # Create temp patch file
            patch_file = self.repo_path / '.temp_patch.diff'
            with open(patch_file, 'w') as f:
                f.write(patch_content)
            
            # Apply patch using git
            result = subprocess.run(
                ["git", "apply", str(patch_file)],
                cwd=self.repo_path,
                capture_output=True
            )
            
            # Clean up temp file
            patch_file.unlink()
            
            if result.returncode == 0:
                return True
            else:
                print(f"Patch apply failed: {result.stderr.decode()}")
                return False
                
        except Exception as e:
            print(f"Error applying patch: {e}")
            return False
    
    def get_git_diff(self) -> str:
        """
        Get current git diff (changes since last commit)
        
        Returns:
            Diff output as string
        """
        try:
            result = subprocess.run(
                ["git", "diff"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Error getting diff: {e}")
            return ""



