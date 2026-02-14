from llm_engineering.domain.base.patterns import SingletonMeta
from loguru import logger

class HydeGenerator(metaclass=SingletonMeta):
    def __init__(self, mock: bool = True):
        self._mock = mock

    def generate(self, query: str) -> str:
        """
        Generates a hypothetical answer for the given query.
        """
        if self._mock:
            return self._mock_generate(query)
        
        # Real implementation would call an LLM here
        return self._mock_generate(query)

    def _mock_generate(self, query: str) -> str:
        logger.info(f"Generating hypothetical answer for query: {query}")
        
        # Simple heuristic to make the hypothetical answer look relevant
        if "jwt" in query.lower():
            return f"JWT (JSON Web Token) is a compact, URL-safe means of representing claims to be transferred between two parties. The claims in a JWT are encoded as a JSON object that is used as the payload of a JSON Web Signature (JWS) structure or as the plaintext of a JSON Web Encryption (JWE) structure. {query}"
        elif "oauth" in query.lower():
            return f"OAuth 2.0 is the industry-standard protocol for authorization. OAuth 2.0 focuses on client developer simplicity while providing specific authorization flows for web applications, desktop applications, mobile phones, and living room devices. {query}"
        elif "kafka" in query.lower():
            return f"Apache Kafka is an open-source distributed event streaming platform used by thousands of companies for high-performance data pipelines, streaming analytics, data integration, and mission-critical applications. {query}"
            
        return f"Hypothetical answer regarding {query}. This document discusses the key concepts and implementation details relevant to the user's request."
