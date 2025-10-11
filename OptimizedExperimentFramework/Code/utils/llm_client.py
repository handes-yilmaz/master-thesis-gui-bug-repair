"""
Clean LLM Client wrapper for OpenAI and Claude
Handles API calls, response parsing, and token tracking
"""
import time
import json
import re
from typing import Dict, Any, Optional, Tuple
from pydantic import BaseModel
from openai import OpenAI
import anthropic

# Import JSON5 for robust parsing
try:
    import json5
except ImportError:
    json5 = json  # Fallback to standard json


class TokenUsage(BaseModel):
    """Track token usage"""
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMClient:
    """
    Unified LLM client for OpenAI and Claude
    Provides clean interface for structured and unstructured outputs
    """
    
    def __init__(
        self,
        provider: str,
        api_key: str,
        model: str,
        wait_time: int = 0
    ):
        """
        Initialize LLM client
        
        Args:
            provider: 'openai' or 'claude'
            api_key: API key for provider
            model: Model name (e.g., 'gpt-4o-2024-08-06')
            wait_time: Seconds to wait after each API call
        """
        self.provider = provider.lower()
        self.model = model
        self.wait_time = wait_time
        
        if self.provider == "openai":
            self.client = OpenAI(api_key=api_key)
        elif self.provider == "claude":
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def chat(
        self,
        system_prompt: str,
        user_prompt: Any,  # Can be string or list (for multimodal)
        temperature: float = 0.0,
        response_format: Optional[BaseModel] = None,
        num_samples: int = 1
    ) -> Tuple[Dict[str, Any], TokenUsage]:
        """
        Main chat interface - routes to appropriate provider
        
        Args:
            system_prompt: System instruction
            user_prompt: User message (str) or multimodal content (list)
            temperature: Sampling temperature
            response_format: Pydantic model for structured output
            num_samples: Number of completions to generate
            
        Returns:
            (results_dict, token_usage)
        """
        if self.provider == "openai":
            return self._openai_chat(
                system_prompt, user_prompt, temperature,
                response_format, num_samples
            )
        else:
            return self._claude_chat(
                system_prompt, user_prompt, temperature,
                response_format, num_samples
            )
    
    def _openai_chat(
        self,
        system_prompt: str,
        user_prompt: Any,
        temperature: float,
        response_format: Optional[BaseModel],
        num_samples: int
    ) -> Tuple[Dict[str, Any], TokenUsage]:
        """OpenAI API call"""
        
        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Handle multimodal input
        if isinstance(user_prompt, list):
            messages.append({"role": "user", "content": user_prompt})
        else:
            messages.append({"role": "user", "content": user_prompt})
        
        # Make API call
        if response_format:
            # Structured output
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                temperature=temperature,
                n=num_samples,
                response_format=response_format
            )
            
            # Parse results
            results = {}
            for idx, choice in enumerate(completion.choices, 1):
                results[idx] = choice.message.parsed.model_dump()
        else:
            # Unstructured output
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                n=num_samples
            )
            
            # Parse results
            results = {}
            for idx, choice in enumerate(completion.choices, 1):
                results[idx] = choice.message.content
        
        # Track tokens
        usage = TokenUsage(
            model=self.model,
            prompt_tokens=completion.usage.prompt_tokens,
            completion_tokens=completion.usage.completion_tokens,
            total_tokens=completion.usage.total_tokens
        )
        
        # Rate limiting
        time.sleep(self.wait_time)
        
        return results, usage
    
    def _claude_chat(
        self,
        system_prompt: str,
        user_prompt: Any,
        temperature: float,
        response_format: Optional[BaseModel],
        num_samples: int
    ) -> Tuple[Dict[str, Any], TokenUsage]:
        """Claude API call"""
        
        results = {}
        total_input_tokens = 0
        total_output_tokens = 0
        
        for sample_idx in range(num_samples):
            # Make API call
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8192,
                temperature=temperature,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            )
            
            # Extract content
            content = response.content[0].text
            
            # Parse if structured output expected
            if response_format:
                parsed = self._extract_json_from_text(content)
                results[sample_idx + 1] = parsed
            else:
                results[sample_idx + 1] = content
            
            # Track tokens
            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens
            
            # Rate limiting between samples
            if sample_idx < num_samples - 1:
                time.sleep(self.wait_time)
        
        # Create usage object
        usage = TokenUsage(
            model=self.model,
            prompt_tokens=total_input_tokens,
            completion_tokens=total_output_tokens,
            total_tokens=total_input_tokens + total_output_tokens
        )
        
        return results, usage
    
    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract and parse JSON from LLM response
        Handles both clean JSON and JSON embedded in markdown
        """
        # Try to find JSON in text
        json_match = re.search(r'({[\s\S]*})', text)
        if not json_match:
            return {"error": "No JSON found in response"}
        
        json_str = json_match.group(1)
        
        try:
            # Try standard JSON first
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try JSON5 for more lenient parsing
            try:
                return json5.loads(json_str)
            except Exception as e:
                return {"error": f"JSON parse error: {e}"}



