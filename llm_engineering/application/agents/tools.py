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
        Simulate searching the vector database (Qdrant).
        """
        logger.info(f"Tool executed: SEARCH_KNOWLEDGE_BASE with query='{query}'")
        
        # Mock logic based on query keywords
        if "jwt" in query.lower() or "auth" in query.lower():
            return (
                "Found 3 relevant documents:\n"
                "1. [Source: Github/nest-auth] 'Implemented JWT strategy using passport-jwt...'\n"
                "2. [Source: Medium] 'Best practices for storing JWT in cookies vs localStorage...'\n"
                "3. [Source: Github/auth-service] 'Refactored login flow to return access and refresh tokens...'"
            )
        elif "deploy" in query.lower():
            return (
                "Found 2 relevant documents:\n"
                "1. [Source: Github/infra] 'Docker compose file for production deployment...'\n"
                "2. [Source: Medium] 'Scaling NestJS on AWS ECS...'"
            )
        
        return "No relevant documents found in the knowledge base."

    @staticmethod
    def web_search(query: str) -> str:
        """
        Simulate a web search (e.g., Google/Bing).
        """
        logger.info(f"Tool executed: WEB_SEARCH with query='{query}'")
        
        # Mock logic
        if "oauth2" in query.lower():
            return (
                "Search Results for 'OAuth2 vs JWT':\n"
                "- OAuth2 is a protocol for authorization, JWT is a token format.\n"
                "- Use OAuth2 for third-party access (e.g., 'Log in with Google').\n"
                "- Use JWT for stateless authentication between microservices."
            )
        
        return "Web search returned general information about Python and MLOps."

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
        Mock Graph Search using Neo4j Adapter.
        """
        logger.info(f"Tool executed: SEARCH_GRAPH with query='{query}'")
        from llm_engineering.infrastructure.graph.neo4j_adapter import Neo4jAdapter
        
        # Ensure connection (mock/real)
        try:
             # In a real app we'd query Neo4j here
             pass
        except:
             pass

        if "jwt" in query.lower():
            return "Graph Results: (JWT)-[:USED_FOR]->(Authentication), (JWT)-[:TYPE]->(Token)"
        elif "fastapi" in query.lower():
             return "Graph Results: (FastAPI)-[:IS_A]->(Web Framework), (FastAPI)-[:SUPPORTS]->(Async)"
        
        return f"No specific graph entities found for: {query}"
