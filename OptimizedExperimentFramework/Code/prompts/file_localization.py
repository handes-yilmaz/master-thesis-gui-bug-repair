"""
Phase 1: File Localization Prompts
Identify bug-related files from repository structure
"""
from typing import List, Dict, Any
from pydantic import BaseModel


class BugFilesResponse(BaseModel):
    """Expected response format for file localization"""
    bug_scenario: str
    bug_files: List[str]
    explanation: str


class FileLocalizationPrompts:
    """
    Generate prompts for Phase 1: File Localization
    Identifies which files in the repository are related to the bug
    """
    
    @staticmethod
    def create_system_prompt(provider: str = "openai") -> str:
        """
        Create system prompt for file localization
        
        Args:
            provider: 'openai' or 'claude'
            
        Returns:
            System prompt string
        """
        base_prompt = """The user will provide the Bug Report (may attach the bug images) and Repository Structure. Please describe the bug scenario images, then return all bug related files and explain why these files are bug related.

EXAMPLE INPUT:

* Bug Report
'''
problem_statement
'''

* Repository Structure
'''
repo_structure
'''

* Bug Scenario Images
'''
image
'''

EXAMPLE OUTPUT:
{   
    "bug_scenario": "Description of the bug scenario.",
    "bug_files": ["src/bug_file1.js", "src/path/bug_file2.js", "src/path/bug_file3.js"],
    "explanation": "Explanation of why these files are bug related."
}"""
        
        if provider == "claude":
            return base_prompt + "\n\nNote that you need to output in JSON format with keys: \"bug_scenario\" (str), \"bug_files\" (list), and \"explanation\" (str)."
        
        return base_prompt
    
    @staticmethod
    def create_user_prompt(
        bug_report: str,
        repo_structure: str,
        images: List[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Create user prompt for file localization
        
        Args:
            bug_report: The bug report text
            repo_structure: Formatted repository structure
            images: Optional list of image objects for multimodal
            
        Returns:
            Formatted user prompt (list for multimodal support)
        """
        prompt_text = f"""I will give you the bug related information (i.e., Bug Report) for your references, you need to find all suspicious bug related files in the code Repo.
    1. Read the bug report and view the bug scenario images (if images are available) to describe and analyze the bug scenario images; 
    2. Look at the Repository Structure to find bug related files that would need to be edited to fix the problem; 
    3. Return all bug related files and explain why these files are bug related.
    
* Bug Report
'''
{bug_report}
'''

* Repository Structure
'''
{repo_structure}
'''
"""
        
        # Build user message
        user_content = [{"type": "text", "text": prompt_text}]
        
        # Add images if provided
        if images:
            user_content.extend(images)
        
        return user_content
    
    @staticmethod
    def create_refinement_prompt(
        bug_report: str,
        candidate_files: Dict[str, str],
        max_files: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Create prompt to refine file list (if too many candidates)
        
        Args:
            bug_report: The bug report text
            candidate_files: Dict of file paths to compressed content
            max_files: Maximum number of files to select
            
        Returns:
            Formatted user prompt
        """
        files_text = "\n\n".join([
            f"### {path}\n{content}" 
            for path, content in candidate_files.items()
        ])
        
        limit_text = f" (at most {max_files} key bug files)" if len(candidate_files) > max_files else ""
        
        prompt_text = f"""I will give you the bug related information (i.e., Bug Report) for your references, you need to find key bug files{limit_text} by looking at all Compressed Bug Files.
    1. Read the bug report and analyze the bug scenario; 
    2. Look at all compressed bug files to find key bug files that would need to be edited to fix the problem; 
    3. Return the key bug files{limit_text} and explain why these files are bug files.
    
* Bug Report
'''
{bug_report}
'''

* Compressed Bug Files
'''
{files_text}
'''
"""
        
        return [{"type": "text", "text": prompt_text}]
    
    @staticmethod
    def get_response_format():
        """Get Pydantic response format for OpenAI structured output"""
        return BugFilesResponse



