import os
from typing import Any

import google.generativeai as genai

from .base_llm import BaseLLM


class GeminiLLM(BaseLLM):
    def __init__(self, api_key_env_var: str = "GOOGLE_API_KEY", model_name: str = "gemini-1.5-flash") -> None:
        api_key = os.getenv(api_key_env_var)
        if not api_key:
            raise ValueError(f"Environment variable {api_key_env_var} not set.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        temperature = kwargs.get("temp", 0.0)
        max_output_tokens = kwargs.get("max_tokens", 512)

        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
            )
            return (getattr(response, "text", None) or "").strip()
        except Exception as e:
            return f"Error: {e}"


