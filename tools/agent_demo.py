from rich.console import Console
from rich.markdown import Markdown
from rich.markdown import Markdown
from rich.panel import Panel
from unittest.mock import MagicMock
import sys

# Mock ZenML Client to avoid connection retries during demo
mock_zenml = MagicMock()
mock_zenml.client.Client.return_value.get_secret.side_effect = Exception("Mocked connection failure")
sys.modules["zenml.client"] = mock_zenml

from llm_engineering.application.agents.research_agent import ResearchAgent

console = Console()

def run_demo():
    console.print(Panel.fit("[bold blue]NeuralTwin Agentic RAG Demo[/bold blue]"))
    
    query = "Analyze the relation between JWT and FastAPI in our system and compare authentication with industry standards."
    console.print(Panel.fit(f"[bold blue]User Query:[/bold blue] {query}"))
    
    console.print("[yellow]Initializing Agent...[/yellow]")
    agent = ResearchAgent(use_mock_llm=True)
    
    console.print("[yellow]Agent Reasoning Chain:[/yellow]")
    result = agent.solve(query)
    
    console.print("\n[bold green]Final Answer:[/bold green]")
    console.print(Markdown(result.answer))

if __name__ == "__main__":
    run_demo()
