
# ==============================================================================
#  LLM Twin - Portfolio Edition | Makefile
# ==============================================================================

# Ensure poetry is in PATH
export PATH := $(HOME)/.local/bin:$(PATH)

.PHONY: help setup start stop clean test lint format data-pipeline rag-server

help:  ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Install dependencies and setup environment
	@bash setup.sh

start: ## Start local infrastructure (Mongo + Qdrant)
	@poetry run poe local-infrastructure-up

stop: ## Stop local infrastructure
	@poetry run poe local-infrastructure-down

restart: stop start ## Restart infrastructure

clean: ## Remove temporary files and caches
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete

# --- APP EXECUTION ---

rag-server: ## Start the RAG Inference API
	@echo "🚀 Starting RAG Inference Service..."
	@poetry run poe run-inference-ml-service

rag-test: ## Send a test query to the local RAG server
	@poetry run poe call-inference-ml-service

train-showcase: ## Run the training pipeline (Simulated/Mocked)
	@echo "🧪 Running Training Showcase..."
	@export RUN_TRAINING=true && poetry run poe run-training-pipeline

# --- DATA PIPELINES ---

etl: ## Run the Data ETL Pipeline
	@echo "📡 Running ETL Pipeline..."
	@poetry run poe run-digital-data-etl

feature-engineering: ## Run Feature Engineering Pipeline
	@echo "⚙️ Running Feature Engineering..."
	@poetry run poe run-feature-engineering-pipeline

# --- QA & FORMATTING ---

lint: ## Run code linting
	@poetry run poe lint-check

format: ## Run code formatting
	@poetry run poe format-fix

# --- ARCHITECTURE DEMO ---

start-streaming: ## Start Kafka and Streaming Consumers
	@echo "🌊 Starting Kafka Event Stream..."
	@docker compose up -d zookeeper kafka
	@echo "⏳ Waiting for Kafka to be ready..."
	@sleep 10
	@echo "🚀 Starting Consumers..."
	@poetry run python -m llm_engineering.infrastructure.streaming.consumers.document_processor_consumer &
	@poetry run python -m llm_engineering.infrastructure.streaming.consumers.embedding_consumer &

run-agent-demo: ## Run the Agentic RAG Demo
	@echo "🤖 Running Research Agent Demo..."
	@poetry run python tools/agent_demo.py

kafka-topics: ## List active Kafka topics
	@docker exec llm_engineering_kafka kafka-topics --list --bootstrap-server localhost:9092

graph-ingest: ## Ingest mock data into Neo4j
	@echo "🕸️ Ingesting Graph Data..."
	@poetry run python tools/run_graph_ingestion.py
