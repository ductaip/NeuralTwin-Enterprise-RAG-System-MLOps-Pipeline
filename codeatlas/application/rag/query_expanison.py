import opik
from loguru import logger

from codeatlas.application.utils.llm_factory import get_llm
from codeatlas.domain.queries import Query

from .base import RAGStep
from .prompt_templates import QueryExpansionTemplate


class QueryExpansion(RAGStep):
    @opik.track(name="QueryExpansion.generate")
    def generate(self, query: Query, expand_to_n: int) -> list[Query]:
        assert expand_to_n > 0, f"'expand_to_n' should be greater than 0. Got {expand_to_n}."

        query_expansion_template = QueryExpansionTemplate()
        prompt = query_expansion_template.create_template(expand_to_n - 1).format(question=query.content)
        model = get_llm(temperature=0)

        response = model.invoke(prompt)
        result = response.content if hasattr(response, "content") else str(response)

        queries_content = result.strip().split(query_expansion_template.separator)

        queries = [query]
        queries += [
            query.replace_content(stripped_content)
            for content in queries_content
            if (stripped_content := content.strip())
        ]

        return queries


if __name__ == "__main__":
    query = Query.from_str("Write an article about the best types of advanced RAG methods.")
    query_expander = QueryExpansion()
    expanded_queries = query_expander.generate(query, expand_to_n=3)
    for expanded_query in expanded_queries:
        logger.info(expanded_query.content)
