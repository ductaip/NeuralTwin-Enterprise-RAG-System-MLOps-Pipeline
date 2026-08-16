from langchain.prompts import PromptTemplate

from .base import PromptTemplateFactory


class QueryExpansionTemplate(PromptTemplateFactory):
    prompt: str = """You are an AI language model assistant. Your task is to generate {expand_to_n}
    different versions of the given user question to retrieve relevant documents from a vector
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search.
    Provide these alternative questions seperated by '{separator}'.
    Original question: {question}"""

    @property
    def separator(self) -> str:
        return "#next-question#"

    def create_template(self, expand_to_n: int) -> PromptTemplate:
        return PromptTemplate(
            template=self.prompt,
            input_variables=["question"],
            partial_variables={
                "separator": self.separator,
                "expand_to_n": expand_to_n,
            },
        )


class SelfQueryTemplate(PromptTemplateFactory):
    prompt: str = """You are an AI language model assistant. Your task is to extract information from a user question.
    The required information that needs to be extracted is the user name or user id. 
    Your response should consists of only the extracted user name (e.g., John Doe) or id (e.g. 1345256), nothing else.
    If the user question does not contain any user name or id, you should return the following token: none.
    
    For example:
    QUESTION 1:
    My name is Paul Iusztin and I want a post about...
    RESPONSE 1:
    Paul Iusztin
    
    QUESTION 2:
    I want to write a post about...
    RESPONSE 2:
    none
    
    QUESTION 3:
    My user id is 1345256 and I want to write a post about...
    RESPONSE 3:
    1345256
    
    User question: {question}"""

    def create_template(self) -> PromptTemplate:
        return PromptTemplate(template=self.prompt, input_variables=["question"])


class ContextualEnrichmentTemplate(PromptTemplateFactory):
    """One-sentence description of a chunk for contextual retrieval (spec §2.5)."""

    prompt: str = """You are documenting a codebase. Given a function/method and facts about
    where it lives in the codebase, write ONE short sentence (max 30 words) describing what
    module it belongs to, its role, and who calls it. Output only the sentence, no preamble.

    Symbol: {qualified_name}
    Graph facts: {graph_context}
    Code:
    {code}

    One-sentence context:"""

    def create_template(self) -> PromptTemplate:
        return PromptTemplate(
            template=self.prompt, input_variables=["qualified_name", "graph_context", "code"]
        )


class CodeHydeTemplate(PromptTemplateFactory):
    """HyDE for code: generate a hypothetical Python snippet that would answer the
    question, then embed *that* — code embeds closer to code than prose does."""

    prompt: str = """You are a senior Python engineer. Given a question about a codebase,
    write a short, plausible Python code snippet (function or class, with a docstring)
    that would answer it — as if it were the actual implementation. Do not explain, do
    not use markdown fences, output only the code.

    Question: {question}

    Hypothetical code:"""

    def create_template(self) -> PromptTemplate:
        return PromptTemplate(template=self.prompt, input_variables=["question"])
