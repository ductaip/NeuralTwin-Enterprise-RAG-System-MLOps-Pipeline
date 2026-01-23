import numpy as np
from loguru import logger

class RAGMetrics:
    """
    A comprehensive suite of metrics for evaluating RAG pipeline performance.
    """

    @staticmethod
    def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
        """
        Calculates Precision@K.
        
        Args:
            retrieved_ids: List of document IDs retrieved by the system.
            relevant_ids: List of ground-truth relevant document IDs.
            k: Cut-off rank.
        
        Returns:
            Precision score at rank k.
        """
        if not relevant_ids:
            return 0.0
        
        k = min(k, len(retrieved_ids))
        retrieved_at_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        
        intersection = retrieved_at_k.intersection(relevant_set)
        return len(intersection) / k

    @staticmethod
    def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
        """
        Calculates Recall@K.
        """
        if not relevant_ids:
            return 0.0
            
        k = min(k, len(retrieved_ids))
        retrieved_at_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        
        intersection = retrieved_at_k.intersection(relevant_set)
        return len(intersection) / len(relevant_set)

    @staticmethod
    def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
        """
        Calculates Mean Reciprocal Rank (MRR).
        Consider checking the first relevant document's rank.
        """
        relevant_set = set(relevant_ids)
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def evaluate_retrieval(
        retrieved_docs_map: dict[str, list[str]], 
        ground_truth_map: dict[str, list[str]], 
        k_values: list[int] = [1, 3, 5, 10]
    ) -> dict:
        """
        Runs evaluation over a dataset.
        
        Args:
            retrieved_docs_map: Dict mapping query_id -> list of retrieved doc IDs
            ground_truth_map: Dict mapping query_id -> list of relevant doc IDs
            k_values: List of K values to compute Precision/Recall for.
        """
        results = {f"P@{k}": [] for k in k_values}
        results.update({f"R@{k}": [] for k in k_values})
        results["MRR"] = []
        
        for query_id, retrieved in retrieved_docs_map.items():
            relevant = ground_truth_map.get(query_id, [])
            if not relevant:
                continue
                
            for k in k_values:
                results[f"P@{k}"].append(RAGMetrics.precision_at_k(retrieved, relevant, k))
                results[f"R@{k}"].append(RAGMetrics.recall_at_k(retrieved, relevant, k))
            
            results["MRR"].append(RAGMetrics.mean_reciprocal_rank(retrieved, relevant))
            
        # Aggregate
        aggregated = {metric: np.mean(scores) for metric, scores in results.items()}
        logger.info(f"Evaluation Results: {aggregated}")
        return aggregated

if __name__ == "__main__":
    # Example Usage Showcase
    mock_retrieved = {"q1": ["doc1", "doc2", "doc3"], "q2": ["doc4", "doc5"]}
    mock_truth = {"q1": ["doc1", "doc3"], "q2": ["doc6"]}
    
    print("Running Mock Evaluation...")
    metrics = RAGMetrics.evaluate_retrieval(mock_retrieved, mock_truth)
    print(metrics)
