import concurrent.futures

import opik
from loguru import logger
from qdrant_client.models import FieldCondition, Filter, MatchValue

from llm_engineering.application import utils
from llm_engineering.application.preprocessing.dispatchers import EmbeddingDispatcher
from llm_engineering.domain.embedded_chunks import (
    EmbeddedArticleChunk,
    EmbeddedChunk,
    EmbeddedPostChunk,
    EmbeddedRepositoryChunk,
)
from llm_engineering.domain.queries import EmbeddedQuery, Query

from .query_expanison import QueryExpansion
from .reranking import Reranker
from .self_query import SelfQuery
from .hyde_generator import HydeGenerator


class ContextRetriever:
    def __init__(self, mock: bool = False) -> None:
        self._query_expander = QueryExpansion(mock=mock)
        self._metadata_extractor = SelfQuery(mock=mock)
        self._reranker = Reranker(mock=mock)
        self._hyde_generator = HydeGenerator(mock=mock)

    @opik.track(name="ContextRetriever.search")
    def search(
        self,
        query: str,
        k: int = 3,
        expand_to_n_queries: int = 3,
    ) -> list:
        query_model = Query.from_str(query)

        query_model = self._metadata_extractor.generate(query_model)
        logger.info(
            f"Successfully extracted the author_full_name = {query_model.author_full_name} from the query.",
        )

        # HyDE Generation
        hypothetical_answer = self._hyde_generator.generate(query)
        logger.info(f"HyDE generated hypothetical answer: {hypothetical_answer[:100]}...")
        
        # Use Hypothetical Answer for Vector Search
        # We create a new Query object with the hypothetical content but keep metadata from original
        hyde_query_model = Query.from_str(hypothetical_answer)
        hyde_query_model.author_id = query_model.author_id
        hyde_query_model.author_full_name = query_model.author_full_name
        
        # We process the HyDE query
        # For simplicity in this upgrade, we might skip Query Expansion on the HyDE query 
        # or we could expand the HyDE query. Let's stick to single HyDE query for now as per instructions.
        
        # However, to keep existing structure working (which uses a list of queries), we'll put it in a list.
        # We can also mix it with the original query if we wanted Hybrid-HyDE, but the instruction said "instead of".
        
        queries_to_search = [hyde_query_model]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            search_tasks = [executor.submit(self._search, _query_model, k) for _query_model in queries_to_search]

            n_k_documents = [task.result() for task in concurrent.futures.as_completed(search_tasks)]
            n_k_documents = utils.misc.flatten(n_k_documents)
            n_k_documents = list(set(n_k_documents))

        logger.info(f"{len(n_k_documents)} documents retrieved successfully using HyDE")

        if len(n_k_documents) > 0:
            k_documents = self.rerank(query, chunks=n_k_documents, keep_top_k=k)
        else:
            k_documents = []

        # Hierarchical Retrieval: Small-to-Big Context Expansion
        # After reranking with child chunks (precise), expand to parent documents (context-rich)
        k_documents = self._expand_to_parent_context(k_documents)

        return k_documents

    def _expand_to_parent_context(self, chunks: list[EmbeddedChunk]) -> list[EmbeddedChunk]:
        """
        Hierarchical Retrieval (Small-to-Big):
        Extracts parent_content from chunk metadata and injects it into the chunk object.
        This allows to_context() to use the full parent document instead of small chunks.
        """
        for chunk in chunks:
            if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
                parent_content = chunk.metadata.get("parent_content")
                if parent_content:
                    chunk.parent_content = parent_content

        logger.info(
            f"Hierarchical Retrieval: expanded {sum(1 for c in chunks if c.parent_content)} / {len(chunks)} chunks to parent context."
        )
        return chunks

    def _reciprocal_rank_fusion(self, dense_results: list[EmbeddedChunk], sparse_results: list[EmbeddedChunk], k: int = 60) -> list[EmbeddedChunk]:
        """
        Implements Reciprocal Rank Fusion (RRF) for hybrid search.
        
        Args:
            dense_results: Results from semantic search (embeddings).
            sparse_results: Results from keyword search (BM25).
            k: Constant k for RRF, typically 60.
        """
        scores = {}
        
        # Helper to process results
        def process_results(results):
            for rank, doc in enumerate(results):
                if doc.content not in scores:
                    scores[doc.content] = {"doc": doc, "score": 0.0}
                scores[doc.content]["score"] += 1.0 / (k + rank + 1)
        
        process_results(dense_results)
        process_results(sparse_results)
        
        # Sort by score
        sorted_docs = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_docs]

    def _search(self, query: Query, k: int = 3) -> list[EmbeddedChunk]:
        assert k >= 3, "k should be >= 3"

        def _search_data_category(
            data_category_odm: type[EmbeddedChunk], embedded_query: EmbeddedQuery
        ) -> list[EmbeddedChunk]:
            if embedded_query.author_id:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="author_id",
                            match=MatchValue(
                                value=str(embedded_query.author_id),
                            ),
                        )
                    ]
                )
            else:
                query_filter = None

            return data_category_odm.search(
                query_vector=embedded_query.embedding,
                limit=k // 3,
                query_filter=query_filter,
            )

        embedded_query: EmbeddedQuery = EmbeddingDispatcher.dispatch(query)

        post_chunks = _search_data_category(EmbeddedPostChunk, embedded_query)
        articles_chunks = _search_data_category(EmbeddedArticleChunk, embedded_query)
        repositories_chunks = _search_data_category(EmbeddedRepositoryChunk, embedded_query)

        retrieved_chunks = post_chunks + articles_chunks + repositories_chunks
        
        # SHOWCASE: Hybrid Search Implementation
        # In a full production setup with sparse vectors indexed in Qdrant, we would do:
        #
        # sparse_query = EmbeddingDispatcher.dispatch_sparse(query) 
        # sparse_post_chunks = _search_data_category(EmbeddedPostChunk, sparse_query, is_sparse=True)
        # ...
        # retrieved_chunks = self._reciprocal_rank_fusion(retrieved_chunks, sparse_chunks)
        #
        # For this portfolio, we focus on the Dense Retrieval + Reranking flow which is fully functional.

        return retrieved_chunks

    def rerank(self, query: str | Query, chunks: list[EmbeddedChunk], keep_top_k: int) -> list[EmbeddedChunk]:
        if isinstance(query, str):
            query = Query.from_str(query)

        reranked_documents = self._reranker.generate(query=query, chunks=chunks, keep_top_k=keep_top_k)

        logger.info(f"{len(reranked_documents)} documents reranked successfully.")

        return reranked_documents
