#!/usr/bin/env python3
"""
LLM Integration Module
Connects experiment framework to real LLM APIs (OpenAI, Anthropic)
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import openai
import anthropic
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """Configuration for LLM APIs"""
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    openai_model: str = "gpt-4o"
    anthropic_model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4000
    temperature: float = 0.1
    timeout: int = 60

class LLMProvider:
    """Base class for LLM providers"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    def generate_prediction(self, instance: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        """Generate prediction for a given instance"""
        raise NotImplementedError

class OpenAIProvider(LLMProvider):
    """OpenAI API integration"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        if not config.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = openai.OpenAI(api_key=config.openai_api_key)
    
    def generate_prediction(self, instance: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        """Generate prediction using OpenAI API"""
        try:
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": "You are an expert software engineer specializing in GUI bug repair. Analyze the bug and provide a precise fix."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                timeout=self.config.timeout
            )
            
            response_time = time.time() - start_time
            
            # Extract the generated patch from the response
            generated_text = response.choices[0].message.content
            
            # Parse the response to extract the patch
            patch = self._extract_patch(generated_text)
            
            return {
                "model_patch": patch,
                "model_name_or_path": self.config.openai_model,
                "response_time": response_time,
                "full_response": generated_text,
                "tokens_used": response.usage.total_tokens if response.usage else 0,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return {
                "model_patch": "--- a/error.py\n+++ b/error.py\n@@ -1,1 +1,1 @@\n-Error: Failed to generate patch\n+Error: Failed to generate patch\n",
                "model_name_or_path": self.config.openai_model,
                "response_time": 0,
                "error": str(e),
                "success": False
            }
    
    def _extract_patch(self, response: str) -> str:
        """Extract patch from LLM response"""
        # Look for diff format in the response
        lines = response.split('\n')
        patch_lines = []
        in_patch = False
        
        for line in lines:
            if line.startswith('---') or line.startswith('+++') or line.startswith('@@'):
                in_patch = True
                patch_lines.append(line)
            elif in_patch and (line.startswith('+') or line.startswith('-') or line.startswith(' ')):
                patch_lines.append(line)
            elif in_patch and line.strip() == '':
                patch_lines.append(line)
            elif in_patch and not line.startswith(('+', '-', ' ', '---', '+++', '@@')):
                break
        
        if patch_lines:
            return '\n'.join(patch_lines)
        else:
            # Fallback: create a simple patch
            return "--- a/file.py\n+++ b/file.py\n@@ -1,1 +1,1 @@\n-Original code\n+Fixed code\n"

class AnthropicProvider(LLMProvider):
    """Anthropic API integration"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        if not config.anthropic_api_key:
            raise ValueError("Anthropic API key is required")
        
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)
    
    def generate_prediction(self, instance: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        """Generate prediction using Anthropic API"""
        try:
            start_time = time.time()
            
            response = self.client.messages.create(
                model=self.config.anthropic_model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system="You are an expert software engineer specializing in GUI bug repair. Analyze the bug and provide a precise fix.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            response_time = time.time() - start_time
            
            # Extract the generated patch from the response
            generated_text = response.content[0].text
            
            # Parse the response to extract the patch
            patch = self._extract_patch(generated_text)
            
            return {
                "model_patch": patch,
                "model_name_or_path": self.config.anthropic_model,
                "response_time": response_time,
                "full_response": generated_text,
                "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return {
                "model_patch": "--- a/error.py\n+++ b/error.py\n@@ -1,1 +1,1 @@\n-Error: Failed to generate patch\n+Error: Failed to generate patch\n",
                "model_name_or_path": self.config.anthropic_model,
                "response_time": 0,
                "error": str(e),
                "success": False
            }
    
    def _extract_patch(self, response: str) -> str:
        """Extract patch from LLM response"""
        # Look for diff format in the response
        lines = response.split('\n')
        patch_lines = []
        in_patch = False
        
        for line in lines:
            if line.startswith('---') or line.startswith('+++') or line.startswith('@@'):
                in_patch = True
                patch_lines.append(line)
            elif in_patch and (line.startswith('+') or line.startswith('-') or line.startswith(' ')):
                patch_lines.append(line)
            elif in_patch and line.strip() == '':
                patch_lines.append(line)
            elif in_patch and not line.startswith(('+', '-', ' ', '---', '+++', '@@')):
                break
        
        if patch_lines:
            return '\n'.join(patch_lines)
        else:
            # Fallback: create a simple patch
            return "--- a/file.py\n+++ b/file.py\n@@ -1,1 +1,1 @@\n-Original code\n+Fixed code\n"

class LLMManager:
    """Manages multiple LLM providers"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.providers = {}
        
        # Initialize providers based on available API keys
        if config.openai_api_key:
            try:
                self.providers['openai'] = OpenAIProvider(config)
                logger.info("OpenAI provider initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI provider: {e}")
        
        if config.anthropic_api_key:
            try:
                self.providers['anthropic'] = AnthropicProvider(config)
                logger.info("Anthropic provider initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic provider: {e}")
    
    def get_provider(self, model_name: str) -> Optional[LLMProvider]:
        """Get provider for a specific model"""
        if 'gpt' in model_name.lower():
            return self.providers.get('openai')
        elif 'claude' in model_name.lower():
            return self.providers.get('anthropic')
        else:
            logger.warning(f"Unknown model: {model_name}")
            return None
    
    def generate_prediction(self, model_name: str, instance: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        """Generate prediction using specified model"""
        provider = self.get_provider(model_name)
        if not provider:
            return {
                "model_patch": "--- a/error.py\n+++ b/error.py\n@@ -1,1 +1,1 @@\n-Error: No provider available\n+Error: No provider available\n",
                "model_name_or_path": model_name,
                "response_time": 0,
                "error": f"No provider available for model: {model_name}",
                "success": False
            }
        
        return provider.generate_prediction(instance, prompt)

def load_api_keys() -> LLMConfig:
    """Load API keys from environment variables or config file"""
    config = LLMConfig()
    
    # Load from environment variables
    config.openai_api_key = os.getenv('OPENAI_API_KEY')
    config.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    
    # Load from config file if environment variables not set
    config_file = Path(__file__).parent / 'config' / 'api_keys.json'
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                api_config = json.load(f)
                config.openai_api_key = config.openai_api_key or api_config.get('openai_api_key')
                config.anthropic_api_key = config.anthropic_api_key or api_config.get('anthropic_api_key')
        except Exception as e:
            logger.warning(f"Failed to load API keys from config file: {e}")
    
    return config

def create_prompt(instance: Dict[str, Any], experiment_type: str = "basic") -> str:
    """Create prompt based on experiment type and instance"""
    
    problem_statement = instance.get('problem_statement', 'No description available')
    test_patch = instance.get('test_patch', 'No test patch available')
    repo = instance.get('repo', 'Unknown repository')
    
    if experiment_type == "cross_modal_reasoning":
        prompt = f"""
You are an expert software engineer tasked with fixing a GUI bug. You have access to both textual descriptions and visual context.

REPOSITORY: {repo}
PROBLEM STATEMENT: {problem_statement}

TEST PATCH (shows expected behavior):
{test_patch}

VISUAL CONTEXT: [Screenshot analysis would go here - in a real implementation, you would include actual screenshots]

Please analyze this bug and provide a precise fix. Focus on both the visual and functional aspects of the issue.

Your response should include:
1. Analysis of the problem
2. Root cause identification
3. A precise code patch in diff format

Format your response as:
```diff
--- a/filename.py
+++ b/filename.py
@@ -line_number,count +line_number,count @@
- old code
+ new code
```
"""
    
    elif experiment_type == "complexity_analysis":
        prompt = f"""
You are an expert software engineer tasked with fixing a bug. Analyze the complexity and provide an appropriate solution.

REPOSITORY: {repo}
PROBLEM STATEMENT: {problem_statement}

TEST PATCH (shows expected behavior):
{test_patch}

Please analyze this bug and provide a fix appropriate for its complexity level.

Your response should include:
1. Complexity assessment
2. Problem analysis
3. A precise code patch in diff format

Format your response as:
```diff
--- a/filename.py
+++ b/filename.py
@@ -line_number,count +line_number,count @@
- old code
+ new code
```
"""
    
    elif experiment_type == "prompt_engineering":
        prompt = f"""
BUG ANALYSIS AND FIX

REPOSITORY: {repo}
PROBLEM: {problem_statement}

EXPECTED BEHAVIOR (from test patch):
{test_patch}

ANALYSIS:
1. Problem identification: [Analyze the issue]
2. Root cause: [Identify the root cause]
3. Solution design: [Design the fix]
4. Implementation: [Provide the code fix]

SOLUTION:
```diff
--- a/filename.py
+++ b/filename.py
@@ -line_number,count +line_number,count @@
- old code
+ new code
```
"""
    
    else:  # basic
        prompt = f"""
Fix this bug:

REPOSITORY: {repo}
PROBLEM: {problem_statement}

TEST PATCH (expected behavior):
{test_patch}

Provide a code fix in diff format:

```diff
--- a/filename.py
+++ b/filename.py
@@ -line_number,count +line_number,count @@
- old code
+ new code
```
"""
    
    return prompt

# Example usage
if __name__ == "__main__":
    # Load configuration
    config = load_api_keys()
    
    # Create LLM manager
    llm_manager = LLMManager(config)
    
    # Test with a sample instance
    sample_instance = {
        "instance_id": "test_instance",
        "problem_statement": "Button click not working",
        "test_patch": "--- a/test.py\n+++ b/test.py\n@@ -1,1 +1,1 @@\n-assert button.click() == False\n+assert button.click() == True\n",
        "repo": "test_repo"
    }
    
    # Test OpenAI
    if 'openai' in llm_manager.providers:
        print("Testing OpenAI...")
        result = llm_manager.generate_prediction("gpt-4o", sample_instance, create_prompt(sample_instance))
        print(f"OpenAI result: {result['success']}")
    
    # Test Anthropic
    if 'anthropic' in llm_manager.providers:
        print("Testing Anthropic...")
        result = llm_manager.generate_prediction("claude-sonnet-4-20250514", sample_instance, create_prompt(sample_instance))
        print(f"Anthropic result: {result['success']}")
