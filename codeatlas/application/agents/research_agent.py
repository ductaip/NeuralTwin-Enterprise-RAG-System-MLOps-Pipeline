from dataclasses import dataclass, field
from typing import List, Optional

from loguru import logger

from codeatlas.application.agents.tools import AgentAction, AgentTools


@dataclass
class ThoughtStep:
    thought: str
    action: AgentAction
    action_input: str
    observation: Optional[str] = None


@dataclass
class AgentResult:
    query: str
    answer: str
    thought_chain: List[ThoughtStep] = field(default_factory=list)


class ResearchAgent:
    """
    A ReAct (Reasoning + Acting) Agent.

    Reasoning loop not yet implemented — see CodeAtlas roadmap Phase 3
    (LangGraph mode QA), which replaces this with a real tool-using loop.
    """

    def __init__(self):
        self.max_iterations = 5

    def solve(self, query: str) -> AgentResult:
        """
        Main entry point for the agent to solve a query.
        """
        logger.info(f"Agent received query: {query}")

        raise NotImplementedError(
            "ResearchAgent.solve() has no reasoning loop yet. "
            "Implemented in CodeAtlas roadmap Phase 3 (LangGraph mode QA)."
        )
