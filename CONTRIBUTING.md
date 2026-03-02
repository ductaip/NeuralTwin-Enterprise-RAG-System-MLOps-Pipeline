# 🤜🤛 Contributing to NeuralTwin

First off, thanks for taking the time to contribute! 🎉

We want to make creating your own LLM Twin as easy and transparent as possible.

## 🛠️ Development Setup

This project uses **Poetry** for dependency management and **Docker Compose** for infrastructure.

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Poetry (`pip install poetry`)

### Quick Start

1. **Clone the repo**
   ```bash
   git clone https://github.com/ductaip/neuraltwin.git
   cd neuraltwin
   ```

2. **Install dependencies**
   ```bash
   make setup
   ```

3. **Start local infrastructure** (Mongo, Qdrant, Redis)
   ```bash
   make start
   ```

## ✅ Testing

We use `pytest` for testing. Please ensure all tests pass before submitting a PR.

```bash
# Run all tests
make test
```

## 🎨 Code Style

We use `ruff` for linting and formatting. The CI pipeline will fail if code is not formatted.

```bash
# Format code
make format

# Check for linting errors
make lint
```

## 🏗️ Project Structure

The codebase follows **Domain-Driven Design (DDD)** principles:

- `llm_engineering/domain/`: Core business logic and interfaces (Chunks, Documents).
- `llm_engineering/application/`: Application logic (Crawlers, RAG pipelines).
- `llm_engineering/infrastructure/`: External implementations (Db, APIs).
- `pipelines/`: ZenML pipeline definitions.

## 🚀 Pull Request Process

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/amazing-feature`).
3. Commit your changes.
4. Push to the branch.
5. Open a Pull Request.

**Note:** Please update `docs/` if you change any architecture or setup steps.
