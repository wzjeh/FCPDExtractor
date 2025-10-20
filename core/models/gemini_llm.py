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
        
        # 放宽安全过滤器，避免学术论文被误拦截
        safety_settings = {
            genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
        }
        
        self.model = genai.GenerativeModel(model_name, safety_settings=safety_settings)

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
            
            # 检查安全过滤器阻止
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    finish_reason = candidate.finish_reason
                    # SAFETY = 2, RECITATION = 3
                    if finish_reason in [2, 3]:
                        print(f"⚠️ Gemini内容被过滤 (finish_reason={finish_reason})，尝试简化prompt重试...")
                        # 返回空字符串，让上层逻辑处理
                        return ""
            
            return (getattr(response, "text", None) or "").strip()
        except Exception as e:
            print(f"⚠️ Gemini API错误: {e}")
            return ""


