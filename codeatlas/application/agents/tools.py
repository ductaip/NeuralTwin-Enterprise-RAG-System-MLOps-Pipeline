from enum import StrEnum
from typing import List, Optional

from loguru import logger
from pydantic import BaseModel


class AgentAction(StrEnum):
    SEARCH_KNOWLEDGE_BASE = "search_knowledge_base"
    WEB_SEARCH = "web_search"
    CALCULATE = "calculate"
    SEARCH_GRAPH = "search_graph"
    SYNTHESIZE_ANSWER = "synthesize_answer"


class ToolResult(BaseModel):
    action: AgentAction
    result: str
    metadata: Optional[dict] = None


class AgentTools:
    """
    Registry of tools available to the Research Agent.
    """

    @staticmethod
    def search_knowledge_base(query: str) -> str:
        """
        Search the vector database (Qdrant) for relevant context.
        """
        logger.info(f"Tool executed: SEARCH_KNOWLEDGE_BASE with query='{query}'")

        raise NotImplementedError("search_knowledge_base has no real implementation yet.")

    @staticmethod
    def web_search(query: str) -> str:
        """
        Perform a web search.
        """
        logger.info(f"Tool executed: WEB_SEARCH with query='{query}'")

        raise NotImplementedError("web_search has no real implementation yet.")

    @staticmethod
    def calculate(expression: str) -> str:
        """
        Safe evaluation of mathematical expressions.
        """
        logger.info(f"Tool executed: CALCULATE with expression='{expression}'")
        try:
            # Very basic safety check
            allowed = set("0123456789+-*/(). ")
            if not all(c in allowed for c in expression):
                return "Error: Invalid characters in expression"
            
            return str(eval(expression, {"__builtins__": None}, {}))
        except Exception as e:
            return f"Error calculating: {str(e)}"

    @staticmethod
    def search_graph(query: str) -> str:
        """
        Search the Neo4j knowledge graph for entity relations.
        """
        logger.info(f"Tool executed: SEARCH_GRAPH with query='{query}'")

        raise NotImplementedError("search_graph has no real implementation yet.")
