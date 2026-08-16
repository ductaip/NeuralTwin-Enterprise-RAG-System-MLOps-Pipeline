from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from codeatlas.settings import settings

def get_llm(temperature: float = 0.0):
    if settings.MODAL_VLLM_BASE_URL:
        from codeatlas.infrastructure.llm.modal_vllm_provider import ModalVLLMProvider
        return ModalVLLMProvider()

    if settings.USE_GROQ:
        from codeatlas.infrastructure.llm.groq_provider import GroqProvider
        return GroqProvider()

    if settings.USE_VLLM:
        from codeatlas.infrastructure.llm.vllm import VLLMClient
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
