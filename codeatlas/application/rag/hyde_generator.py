import re

import opik
from loguru import logger

from codeatlas.application.utils.llm_factory import get_llm
from codeatlas.domain.base.patterns import SingletonMeta

from .prompt_templates import CodeHydeTemplate

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|```$", re.MULTILINE)


class HydeGenerator(metaclass=SingletonMeta):
    """Generates a hypothetical *code* snippet for a query, per spec §2.5: embedding a
    plausible implementation retrieves closer to real implementations than embedding a
    prose answer would.
    """

    def __init__(self) -> None:
        self._template = CodeHydeTemplate()

    @opik.track(name="HydeGenerator.generate")
    def generate(self, query: str) -> str:
        prompt = self._template.create_template().format(question=query)
        model = get_llm(temperature=0.2)

        response = model.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)

        cleaned = _FENCE_RE.sub("", content).strip()
        logger.info(f"HyDE generated {len(cleaned)} chars for query: {query[:80]}")
        return cleaned or query
