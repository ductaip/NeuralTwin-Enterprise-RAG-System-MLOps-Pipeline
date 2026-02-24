from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from llm_engineering.settings import settings

def get_llm(temperature: float = 0.0):
    if settings.USE_VLLM:
        from llm_engineering.infrastructure.llm.vllm import VLLMClient
        return VLLMClient()

    if settings.USE_OLLAMA:
        return ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL_ID,
            temperature=temperature,
            keep_alive="5m"
        )
    
    return ChatOpenAI(
        model=settings.OPENAI_MODEL_ID,
        api_key=settings.OPENAI_API_KEY,
        temperature=temperature
    )
