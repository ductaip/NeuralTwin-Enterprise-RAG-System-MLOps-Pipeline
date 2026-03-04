import os
from typing import Any, Dict, List, Optional

from loguru import logger
from openai import OpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage

from llm_engineering.domain.exceptions import LLMGenerationError
from llm_engineering.settings import settings


class VLLMClient:
    """
    Client for interacting with a vLLM server via the OpenAI-compatible API.
    Designed to be compatible with LangChain's invoke interface.
    """

    def __init__(self) -> None:
        self.api_key = settings.VLLM_API_KEY
        self.base_url = settings.VLLM_BASE_URL
        self.model = settings.VLLM_MODEL_ID
        
        # Use Synchronous Client for compatibility with existing sync code
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate text using vLLM.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.exception(f"Error generating text with vLLM: {e}")
            raise LLMGenerationError(f"vLLM generation failed: {e}") from e

    def invoke(self, messages: Any) -> AIMessage:
        """
        LangChain-compatible invoke method.
        Expects a list of LangChain messages or dicts.
        Returns an AIMessage object (duck-typed).
        """
        formatted_messages = []
        
        # Handle list of BaseMessage (LangChain)
        if isinstance(messages, list):
            for m in messages:
                if hasattr(m, "content") and hasattr(m, "type"):
                    role = "user" if m.type == "human" else "assistant"
                    if m.type == "system": role = "system"
                    formatted_messages.append({"role": role, "content": m.content})
                elif isinstance(m, dict):
                    formatted_messages.append(m)
        else:
            # Fallback for single message or string (if applicable)
             if hasattr(messages, "content"):
                 formatted_messages.append({"role": "user", "content": messages.content})
             elif isinstance(messages, str):
                 formatted_messages.append({"role": "user", "content": messages})

        content = self.generate(formatted_messages)
        return AIMessage(content=content)
