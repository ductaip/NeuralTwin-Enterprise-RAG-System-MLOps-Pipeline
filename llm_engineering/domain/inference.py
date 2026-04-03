from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Existing abstract contracts
# ---------------------------------------------------------------------------


class DeploymentStrategy(ABC):
    @abstractmethod
    def deploy(self, model, endpoint_name: str, endpoint_config_name: str) -> None:
        pass


class Inference(ABC):
    """An abstract class for performing inference."""

    def __init__(self):
        self.model = None

    @abstractmethod
    def set_payload(self, inputs, parameters=None):
        pass

    @abstractmethod
    def inference(self):
        pass


# ---------------------------------------------------------------------------
# Reasoning Breakdown – Domain Data Contracts
# ---------------------------------------------------------------------------


class ReasoningStepType(str, Enum):
    """The four canonical phases of the NeuralTwin reasoning process."""

    UNDERSTAND = "understand"
    PLAN = "plan"
    EXECUTE = "execute"
    VERIFY = "verify"


class ReasoningStep(BaseModel):
    """A single atomic step produced by the LLM reasoning decomposition."""

    step_number: int = Field(..., ge=1, le=4, description="Ordinal position (1-4).")
    step_type: ReasoningStepType = Field(..., description="Phase of the reasoning cycle.")
    title: str = Field(..., min_length=1, max_length=120, description="Short heading for this step.")
    description: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Detailed explanation of what happens in this step.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model's self-assessed confidence for this step (0-1).",
    )

    class Config:
        frozen = True


class ReasoningStepsResponse(BaseModel):
    """Aggregate returned by ``NeuralTwinAIFacade.extract_reasoning_steps_from_query``."""

    session_id: str = Field(..., description="Caller-supplied session identifier.")
    original_query: str = Field(..., description="The raw user query that was decomposed.")
    steps: list[ReasoningStep] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Exactly four reasoning steps (Understand → Plan → Execute → Verify).",
    )
    model_id: str = Field(..., description="Identifier of the LLM backend that generated the steps.")
    processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Wall-clock time in milliseconds spent on LLM generation.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the response was created.",
    )
