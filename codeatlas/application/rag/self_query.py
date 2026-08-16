import opik
from loguru import logger

from codeatlas.application import utils
from codeatlas.application.utils.llm_factory import get_llm
from codeatlas.domain.documents import UserDocument
from codeatlas.domain.queries import Query

from .base import RAGStep
from .prompt_templates import SelfQueryTemplate


class SelfQuery(RAGStep):
    @opik.track(name="SelfQuery.generate")
    def generate(self, query: Query) -> Query:
        prompt = SelfQueryTemplate().create_template().format(question=query.content)
        model = get_llm(temperature=0)

        response = model.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        user_full_name = content.strip("\n ")

        if user_full_name == "none":
            return query

        first_name, last_name = utils.split_user_full_name(user_full_name)
        user = UserDocument.get_or_create(first_name=first_name, last_name=last_name)

        query.author_id = user.id
        query.author_full_name = user.full_name

        return query


if __name__ == "__main__":
    query = Query.from_str("I am Paul Iusztin. Write an article about the best types of advanced RAG methods.")
    self_query = SelfQuery()
    query = self_query.generate(query)
    logger.info(f"Extracted author_id: {query.author_id}")
    logger.info(f"Extracted author_full_name: {query.author_full_name}")
