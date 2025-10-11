"""
Phase 2: Element Localization Prompts
Identify specific classes and functions within bug files
"""
from typing import List, Dict, Any
from pydantic import BaseModel


class BugElementsResponse(BaseModel):
    """Expected response format for element localization"""
    bug_scenario: str
    bug_classes: List[str]
    bug_functions: List[str]
    explanation: str


class ElementLocalizationPrompts:
    """
    Generate prompts for Phase 2: Element Localization
    Identifies specific classes and functions that need to be fixed
    """
    
    @staticmethod
    def create_system_prompt(provider: str = "openai") -> str:
        """
        Create system prompt for element localization
        
        Args:
            provider: 'openai' or 'claude'
            
        Returns:
            System prompt string
        """
        base_prompt = """The user will provide the Bug Report (may attach the bug images) and Compressed Key Bug Files. Please describe the bug scenario images, then return key bug classes and functions that need to be edited to fix the problem.

Explanation of Compressed Bug Files: We provide a compressed format (skeleton) of each file containing the list of class, function, or variable declarations. In the skeleton format, we provide only the headers of the classes and functions. For classes, we further include any class fields and methods (signatures only). Additionally, we keep comments at the class and module level to provide further information.

EXAMPLE INPUT:

* Bug Report
'''
problem_statement
'''

* Compressed Key Bug Files
'''
{
    "src/bug_file1.js": "compressed_content",
    "src/bug_file2.js": "compressed_content"
}
'''

* Bug Scenario Images
'''
image
'''

EXAMPLE OUTPUT:
{   
    "bug_scenario": "Description of the bug scenario.",
    "bug_classes": ["class_name_1", "class_name_2"],
    "bug_functions": ["function_name_1", "function_name_2", "function_name_3"],
    "explanation": "Explanation of why these classes/functions are bug locations."
}"""
        
        if provider == "claude":
            return base_prompt + "\n\nNote that you need to output in JSON format with keys: \"bug_scenario\" (str), \"bug_classes\" (list), \"bug_functions\" (list), and \"explanation\" (str)."
        
        return base_prompt
    
    @staticmethod
    def create_user_prompt(
        bug_report: str,
        compressed_files: Dict[str, str],
        images: List[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Create user prompt for element localization
        
        Args:
            bug_report: The bug report text
            compressed_files: Dict of file paths to compressed content
            images: Optional list of image objects for multimodal
            
        Returns:
            Formatted user prompt
        """
        # Format compressed files
        files_dict = {
            path: content 
            for path, content in compressed_files.items()
        }
        
        import json
        files_text = json.dumps(files_dict, indent=2)
        
        prompt_text = f"""I will give you the bug related information (i.e., Bug Report) for your references, you need to find key bug classes/functions by looking at all Compressed Key Bug Files.
    1. Read the bug report and view the bug scenario images (if images are available) to describe the bug scenario images; 
    2. Look at all compressed key bug files to find key bug classes and functions that would need to be edited to fix the problem; 
    3. Return all key bug classes and functions and explain why these are the bug locations.
    
* Bug Report
'''
{bug_report}
'''

* Compressed Key Bug Files
'''
{files_text}
'''
"""
        
        # Build user message
        user_content = [{"type": "text", "text": prompt_text}]
        
        # Add images if provided
        if images:
            user_content.extend(images)
        
        return user_content
    
    @staticmethod
    def add_repo_specific_hints(repo_name: str) -> str:
        """
        Add repository-specific hints for better localization
        
        Args:
            repo_name: Name of the repository
            
        Returns:
            Additional hint text to append to prompt
        """
        hints = {
            "p5.js": "\n*Note: For p5.js, function format is 'p5.RendererGL.prototype.newBuffers = function(gId, obj) {...}', the function name is 'p5.RendererGL.prototype.newBuffers'. Don't output incomplete function names.",
            "prism": "\n*Note: For PrismJS, many functions are defined as 'Prism.languages.xxx'. Include the full qualifier.",
        }
        
        for key, hint in hints.items():
            if key.lower() in repo_name.lower():
                return hint
        
        return ""
    
    @staticmethod
    def get_response_format():
        """Get Pydantic response format for OpenAI structured output"""
        return BugElementsResponse



