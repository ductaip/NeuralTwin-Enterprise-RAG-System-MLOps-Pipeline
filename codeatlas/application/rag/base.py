from abc import ABC, abstractmethod
from typing import Any

from langchain.prompts import PromptTemplate
from pydantic import BaseModel

from codeatlas.domain.queries import Query


class PromptTemplateFactory(ABC, BaseModel):
    @abstractmethod
    def create_template(self) -> PromptTemplate:
        pass


class RAGStep(ABC):
    @abstractmethod
    def generate(self, query: Query, *args, **kwargs) -> Any:
        pass
