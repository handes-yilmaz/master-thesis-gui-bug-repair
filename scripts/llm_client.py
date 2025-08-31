#!/usr/bin/env python
import os
import json
from dataclasses import dataclass
from typing import Optional

# OpenAI support
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Anthropic support
try:
    import anthropic
except Exception:
    anthropic = None

@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.2

class LLMClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        
        if cfg.provider == "openai":
            if OpenAI is None:
                raise RuntimeError("openai package not installed. Please `pip install openai`.")
            api_key = os.environ.get(cfg.api_key_env, "")
            if not api_key:
                raise RuntimeError(f"Missing API key in env var {cfg.api_key_env}.")
            self.client = OpenAI(api_key=api_key)
            
        elif cfg.provider == "anthropic":
            if anthropic is None:
                raise RuntimeError("anthropic package not installed. Please `pip install anthropic`.")
            api_key = os.environ.get(cfg.api_key_env, "")
            if not api_key:
                raise RuntimeError(f"Missing API key in env var {cfg.api_key_env}.")
            self.client = anthropic.Anthropic(api_key=api_key)
            
        else:
            raise NotImplementedError(f"Provider {cfg.provider} not supported.")

    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if self.cfg.provider == "openai":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            resp = self.client.chat.completions.create(
                model=self.cfg.model,
                temperature=self.cfg.temperature,
                messages=messages
            )
            return resp.choices[0].message.content or ""
            
        elif self.cfg.provider == "anthropic":
            messages = []
            if system_prompt:
                messages.append({"role": "user", "content": f"{system_prompt}\n\n{prompt}"})
            else:
                messages.append({"role": "user", "content": prompt})
            
            resp = self.client.messages.create(
                model=self.cfg.model,
                temperature=self.cfg.temperature,
                max_tokens=4000,
                messages=messages
            )
            return resp.content[0].text or ""
            
        else:
            raise NotImplementedError
