"""
Phase 3: Patch Generation Prompts
Generate code patches to fix identified bugs
"""
from typing import List, Dict, Any


class PatchGenerationPrompts:
    """
    Generate prompts for Phase 3: Patch Generation
    Creates SEARCH/REPLACE patches to fix the bug
    """
    
    @staticmethod
    def create_system_prompt() -> str:
        """
        Create system prompt for patch generation
        
        Returns:
            System prompt string
        """
        return """The user will provide the Bug Report (may attach the bug images) and Bug Code Snippets. Please analyze the bug scenario images to infer possible bug root cause, then locate the bug locations and generate patches for "* Bug Code Snippets".

Explanation of Bug Code Snippets: We'll provide the key bug code snippets from the bug file, for the rest of the code sections we use ... to omit.
Note that we use dict format to record Bug Code Snippets, the dict's key is the bug file path, and the dict's value is the bug code snippets.

EXAMPLE INPUT:

* Bug Report
'''
problem_statement
'''

* Bug Code Snippets
'''
{
    "bug_file_path_1": "bug_code_snippets",
    "bug_file_path_2": "bug_code_snippets",
    "bug_file_path_3": "bug_code_snippets"
}
'''

* Bug Scenario Images
'''
image
'''


Please first localize the bug based on the issue statement, and then generate *SEARCH/REPLACE* edits (i.e., patches) to fix the issue.

Every *SEARCH/REPLACE* edit must use this format:
1. The bug file path (Please give the specific bug file path in "* Bug Code Snippets", e.g., src/components/Image/ImageSearch.js. Not the reproduce code file.)
2. The start of search block: <<<<<<< SEARCH
3. A contiguous chunk of lines to search for in the existing source code
4. The dividing line: =======
5. The lines to replace into the source code
6. The end of the replace block: >>>>>>> REPLACE

EXAMPLE OUTPUT:
(Here is a *SEARCH/REPLACE* edit example)

```javascript
### bug_file_path_2
<<<<<<< SEARCH
from flask import Flask
from transformer import generate
from transformer import train
=======
import math
from flask import Flask
from transformer import generate
from transformer import train
>>>>>>> REPLACE
```

IMPORTANT RULES:
1. The *SEARCH/REPLACE* edit REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
2. You must provide sufficient *SEARCH* edit context (No less than 3 lines of code) to ensure that the code location can be successfully searched!
3. You can't use "/* ~~~~~~~~~~~~~~~~~~~~ */" or "..." to alter and ignore the original code content, you must keep the original code format and content in the *SEARCH/REPLACE* edit!
4. Wrap the *SEARCH/REPLACE* edit in blocks ```javascript...```.
5. Don't try to fix the reproduce code in Bug Report! Only fix code in "* Bug Code Snippets"!
"""
    
    @staticmethod
    def create_user_prompt(
        bug_report: str,
        bug_code_snippets: Dict[str, str],
        images: List[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Create user prompt for patch generation
        
        Args:
            bug_report: The bug report text
            bug_code_snippets: Dict of file paths to code snippets
            images: Optional list of image objects for multimodal
            
        Returns:
            Formatted user prompt
        """
        # Format code snippets
        import json
        snippets_text = json.dumps(bug_code_snippets, indent=2)
        
        prompt_text = f"""I will give you the bug related information (i.e., Bug Report) for your references, you need to generate patches (*SEARCH/REPLACE* edits) by looking at all Bug Code Snippets.
    1. Read the bug report and view the bug scenario images (if images are available) to describe the bug scenario images and reason about the bug root causes; 
    2. Look at all bug snippets files in "* Bug Code Snippets" to analyze and locate bug locations that would need to be edited to fix the problem; 
    3. Generate patches for bug files in "* Bug Code Snippets" to fix the current bug. (!!! Note that don't try to fix the reproduce code in Bug Report!!!)
    
 
* Bug Report
'''
{bug_report}
'''

* Bug Code Snippets
'''
{snippets_text}
'''
"""
        
        # Build user message - IMAGES FIRST for better attention
        user_content = []
        
        # Add images first if provided (better for multimodal models)
        if images:
            user_content.extend(images)
        
        # Then add text
        user_content.append({"type": "text", "text": prompt_text})
        
        return user_content
    
    @staticmethod
    def create_patch_validation_prompt(
        original_patch: str,
        error_message: str
    ) -> str:
        """
        Create prompt to fix a failed patch
        
        Args:
            original_patch: The original patch that failed
            error_message: Error message from patch application
            
        Returns:
            Prompt text for patch refinement
        """
        return f"""The following patch failed to apply:

{original_patch}

Error: {error_message}

Please generate a corrected patch that:
1. Uses exact text from the original file for SEARCH block
2. Has proper indentation
3. Includes enough context (at least 3 lines)

Return only the corrected *SEARCH/REPLACE* block.
"""
    
    @staticmethod
    def create_multi_sample_prompt(
        bug_report: str,
        bug_code_snippets: Dict[str, str],
        previous_attempts: List[str] = None
    ) -> str:
        """
        Create prompt for generating alternative patches
        
        Args:
            bug_report: The bug report text
            bug_code_snippets: Dict of file paths to code snippets
            previous_attempts: List of previous patch attempts
            
        Returns:
            Prompt text emphasizing alternative solutions
        """
        import json
        snippets_text = json.dumps(bug_code_snippets, indent=2)
        
        base_prompt = f"""* Bug Report
'''
{bug_report}
'''

* Bug Code Snippets
'''
{snippets_text}
'''

Generate an ALTERNATIVE patch to fix this bug."""
        
        if previous_attempts:
            attempts_text = "\n\n".join([
                f"Attempt {i+1}:\n{attempt}" 
                for i, attempt in enumerate(previous_attempts)
            ])
            base_prompt += f"""

Previous attempts (try a different approach):
{attempts_text}"""
        
        return base_prompt
    
    @staticmethod
    def get_language_hints(file_extension: str) -> str:
        """
        Get language-specific hints for patch generation
        
        Args:
            file_extension: File extension (e.g., '.js', '.jsx', '.ts')
            
        Returns:
            Language-specific hint text
        """
        hints = {
            ".js": "JavaScript",
            ".jsx": "JSX (React)",
            ".ts": "TypeScript",
            ".tsx": "TSX (React with TypeScript)",
            ".py": "Python",
            ".java": "Java",
        }
        
        lang = hints.get(file_extension, "code")
        return f"\nWrap the *SEARCH/REPLACE* edit in blocks ```{lang.lower()}...```."



