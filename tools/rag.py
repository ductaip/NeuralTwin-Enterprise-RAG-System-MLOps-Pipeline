import argparse

from langchain.globals import set_verbose
from loguru import logger

from codeatlas.application.rag.retriever import ContextRetriever
from codeatlas.infrastructure.opik_utils import configure_opik

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="Repo indexed via codeatlas.ingest")
    parser.add_argument("--query", required=True)
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    configure_opik()
    set_verbose(True)

    retriever = ContextRetriever(repo_id=args.repo_id)
    documents = retriever.search(args.query, k=args.k)

    logger.info("Retrieved documents:")
    for rank, document in enumerate(documents):
        logger.info(f"{rank + 1}: {document.qualified_name}  ({document.file_path}:{document.start_line}-{document.end_line})")
