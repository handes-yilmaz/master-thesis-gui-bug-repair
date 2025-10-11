"""
JavaScript/TypeScript code parsing utilities
Uses Node.js for robust AST parsing
"""
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class JavaScriptParser:
    """
    Parse JavaScript/TypeScript files to extract:
    - Class definitions
    - Function definitions
    - Code structure
    """
    
    def __init__(self, parser_script_path: Optional[str] = None):
        """
        Initialize JavaScript parser
        
        Args:
            parser_script_path: Path to Node.js parser script
                              If None, looks for parse_js_old.js in parent directory
        """
        if parser_script_path:
            self.parser_script = parser_script_path
        else:
            # Default to parse_js_old.js from Code directory
            self.parser_script = str(
                Path(__file__).parent.parent.parent / "Code" / "parse_js_old.js"
            )
    
    def parse_file(
        self,
        file_path: str
    ) -> Tuple[List[Dict], List[Dict], List[str]]:
        """
        Parse JavaScript file and extract classes and functions
        
        Args:
            file_path: Path to .js/.jsx/.ts/.tsx file
            
        Returns:
            (classes, functions, file_lines)
            - classes: List of class definitions with line numbers
            - functions: List of function definitions with line numbers
            - file_lines: All lines of the file
        """
        try:
            # Run Node.js parser
            result = subprocess.run(
                ["node", self.parser_script, file_path],
                capture_output=True,
                text=True,
                check=True,
                timeout=30  # Prevent hanging
            )
            
            # Parse output
            parsed_data = json.loads(result.stdout)
            
            # Read file lines
            with open(file_path, 'r', encoding='utf-8') as f:
                file_lines = f.read().splitlines()
            
            return (
                parsed_data.get("classes", []),
                parsed_data.get("functions", []),
                file_lines
            )
            
        except subprocess.CalledProcessError as e:
            # Parser failed - return empty structures
            print(f"⚠️  Parser failed for {file_path}: {e.stderr}")
            return self._fallback_parse(file_path)
            
        except subprocess.TimeoutExpired:
            print(f"⚠️  Parser timeout for {file_path}")
            return self._fallback_parse(file_path)
            
        except Exception as e:
            print(f"⚠️  Unexpected error parsing {file_path}: {e}")
            return self._fallback_parse(file_path)
    
    def _fallback_parse(self, file_path: str) -> Tuple[List, List, List[str]]:
        """Fallback when parser fails - just return file lines"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_lines = f.read().splitlines()
            return [], [], file_lines
        except Exception:
            return [], [], []
    
    def compress_code(
        self,
        file_lines: List[str],
        element_location: Optional[Dict] = None,
        max_lines: int = 500,
        context_lines: int = 10
    ) -> str:
        """
        Compress code by keeping only relevant parts
        
        Args:
            file_lines: All lines of the file
            element_location: Dict with 'start_line' and 'end_line' for focus area
            max_lines: Maximum lines to include
            context_lines: Lines of context around focus area
            
        Returns:
            Compressed code as string
        """
        total_lines = len(file_lines)
        
        # If file is small enough, return as-is
        if total_lines <= max_lines:
            return '\n'.join(file_lines)
        
        # If no specific location, return first max_lines
        if not element_location:
            compressed = file_lines[:max_lines]
            compressed.append(f'\n... ({total_lines - max_lines} more lines) ...')
            return '\n'.join(compressed)
        
        # Focus on specific element with context
        start = max(0, element_location['start_line'] - context_lines)
        end = min(total_lines, element_location['end_line'] + context_lines)
        
        compressed = []
        
        # Add header if we're not starting from beginning
        if start > 0:
            compressed.append(f'... ({start} lines before) ...\n')
        
        # Add focused lines with line numbers
        for line_num in range(start, end):
            compressed.append(f'{line_num + 1:4d}| {file_lines[line_num]}')
        
        # Add footer if there's more after
        if end < total_lines:
            compressed.append(f'\n... ({total_lines - end} lines after) ...')
        
        return '\n'.join(compressed)
    
    def extract_element_code(
        self,
        file_lines: List[str],
        element: Dict,
        context_lines: int = 5
    ) -> str:
        """
        Extract code for a specific element (class or function) with context
        
        Args:
            file_lines: All lines of the file
            element: Dict with 'name', 'start_line', 'end_line'
            context_lines: Lines of context before/after
            
        Returns:
            Code snippet as string
        """
        start = max(0, element['start_line'] - context_lines)
        end = min(len(file_lines), element['end_line'] + context_lines)
        
        snippet = []
        for line_num in range(start, end):
            snippet.append(f'{line_num + 1:4d}| {file_lines[line_num]}')
        
        return '\n'.join(snippet)
    
    def get_file_structure_summary(
        self,
        classes: List[Dict],
        functions: List[Dict]
    ) -> str:
        """
        Create a summary of file structure
        
        Returns:
            Human-readable structure summary
        """
        summary = []
        
        if classes:
            summary.append("Classes:")
            for cls in classes:
                summary.append(f"  - {cls['name']} (lines {cls['start_line']}-{cls['end_line']})")
        
        if functions:
            summary.append("Functions:")
            for func in functions:
                summary.append(f"  - {func['name']} (lines {func['start_line']}-{func['end_line']})")
        
        if not classes and not functions:
            summary.append("(No classes or functions detected)")
        
        return '\n'.join(summary)



