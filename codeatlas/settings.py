from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict
from zenml.client import Client
from zenml.exceptions import EntityExistsError


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # --- Required settings even when working locally. ---

    SKIP_TRAINING: bool = True

    # Ollama (Local)
    USE_OLLAMA: bool = False
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL_ID: str = "llama3:8b"

    # OpenAI API
    OPENAI_MODEL_ID: str = "gpt-4o-mini"
    OPENAI_API_KEY: str | None = None

    # Huggingface API
    HUGGINGFACE_ACCESS_TOKEN: str | None = None

    # Comet ML (during training)
    COMET_API_KEY: str | None = None
    COMET_PROJECT: str = "twin"

    # --- Required settings when deploying the code. ---
    # --- Otherwise, default values values work fine. ---

    # MongoDB database
    DATABASE_HOST: str = "mongodb://codeatlas:codeatlas@127.0.0.1:27017"
    DATABASE_NAME: str = "twin"

    # Neo4j graph database
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    NEO4J_DATABASE: str = "neo4j"

    # Ingestion (Phase 1)
    INGESTION_BATCH_SIZE: int = 1000

    # Ngưỡng confidence cho CALL edge. KHÔNG hardcode trong Cypher — mọi truy vấn
    # nhận ngưỡng qua tham số $min_confidence để Phase 5 quét được như một trục ablation.
    # Mặc định thấp vì chọn test là bài toán recall-first: thiếu edge -> bỏ sót test -> bug
    # lọt production; thừa edge -> chạy dư vài test. Xem docs/CODEATLAS_SPEC.md §2.2.
    CALL_EDGE_MIN_CONFIDENCE_IMPACT: float = 0.5
    # Ngưỡng cao cho phân tích cấu trúc (fan-in, dead code) — ở đó nhiễu mới là cái hại.
    CALL_EDGE_MIN_CONFIDENCE_STRUCTURAL: float = 0.9
    # Min hits for COVERS relationship to be considered in impact analysis
    COVERS_MIN_HITS: int = 1
    # Path to local repository clone used as sandbox for closed-loop refactoring
    SANDBOX_REPO_PATH: str = "/home/adminn/.cache/codeatlas-eval/fastapi"

    # Qdrant vector database
    USE_QDRANT_CLOUD: bool = False
    QDRANT_DATABASE_HOST: str = "localhost"
    QDRANT_DATABASE_PORT: int = 6333
    QDRANT_CLOUD_URL: str = "str"
    QDRANT_APIKEY: str | None = None

    # AWS Authentication
    AWS_REGION: str = "eu-central-1"
    AWS_ACCESS_KEY: str | None = None
    AWS_SECRET_KEY: str | None = None
    AWS_ARN_ROLE: str | None = None

    # vLLM Configuration
    USE_VLLM: bool = False
    VLLM_BASE_URL: str = "http://localhost:8000/v1"
    VLLM_MODEL_ID: str = "facebook/opt-125m"
    VLLM_API_KEY: str = "EMPTY"  # vLLM usually requires a dummy key

    # Groq (live demo — latency-bound, 8 sequential tool calls)
    USE_GROQ: bool = False
    GROQ_API_KEY: str | None = None
    GROQ_MODEL_ID: str = "openai/gpt-oss-120b"
    """spec §2.6 named llama-3.3-70b-versatile; Groq's catalog moved on and that model
    404s now (verified live, 2026-08-18 — see docs/codeatlas_roadmap.md "Đính chính").
    This is a reasoning model: it spends tokens on a hidden `reasoning` field before
    `content`, so short `max_tokens` budgets can truncate before any content appears."""
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"

    # Modal + vLLM (eval/ablation — throughput-bound, no rate limit, deterministic)
    #
    # Modal auth needs a *pair*: token_id ("ak-...") + token_secret ("as-...").
    # `MODAL_API_KEY*` below are token_ids only — `modal token set` / the SDK will
    # reject them without the matching `*_TOKEN_SECRET`. See spec §3.4: 4 accounts,
    # $1 each, one deploy per account per work-block to avoid burning cold-start time
    # on redeploys.
    MODAL_API_KEY: str | None = None
    MODAL_TOKEN_SECRET: str | None = None
    MODAL_API_KEY_2: str | None = None
    MODAL_TOKEN_SECRET_2: str | None = None
    MODAL_API_KEY_3: str | None = None
    MODAL_TOKEN_SECRET_3: str | None = None
    MODAL_API_KEY_4: str | None = None
    MODAL_TOKEN_SECRET_4: str | None = None

    MODAL_VLLM_BASE_URL: str | None = None
    """Base URL of the deployed Modal vLLM endpoint. Set after `modal deploy`."""
    MODAL_VLLM_MODEL_ID: str = "Qwen/Qwen2.5-7B-Instruct-AWQ"

    # LLM disk cache, shared by every provider
    LLM_CACHE_DIR: str = ".cache/llm"

    # LangGraph checkpointer (spec §2.3) — sqlite for dev, postgres for prod
    LANGGRAPH_CHECKPOINT_BACKEND: str = "sqlite"
    LANGGRAPH_SQLITE_PATH: str = ".cache/langgraph/checkpoints.sqlite"
    LANGGRAPH_POSTGRES_URI: str | None = None

    # --- Optional settings used to tweak the code. ---

    # AWS SageMaker
    HF_MODEL_ID: str = "mlabonne/TwinLlama-3.1-8B-DPO"
    GPU_INSTANCE_TYPE: str = "ml.g5.2xlarge"
    SM_NUM_GPUS: int = 1
    MAX_INPUT_LENGTH: int = 2048
    MAX_TOTAL_TOKENS: int = 4096
    MAX_BATCH_TOTAL_TOKENS: int = 4096
    COPIES: int = 1  # Number of replicas
    GPUS: int = 1  # Number of GPUs
    CPUS: int = 2  # Number of CPU cores

    SAGEMAKER_ENDPOINT_CONFIG_INFERENCE: str = "twin"
    SAGEMAKER_ENDPOINT_INFERENCE: str = "twin"
    TEMPERATURE_INFERENCE: float = 0.01
    TOP_P_INFERENCE: float = 0.9
    MAX_NEW_TOKENS_INFERENCE: int = 150

    # RAG
    TEXT_EMBEDDING_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKING_CROSS_ENCODER_MODEL_ID: str = "cross-encoder/ms-marco-MiniLM-L-4-v2"
    RAG_MODEL_DEVICE: str = "cpu"

    # LinkedIn Credentials
    LINKEDIN_USERNAME: str | None = None
    LINKEDIN_PASSWORD: str | None = None

    @property
    def OPENAI_MAX_TOKEN_WINDOW(self) -> int:
        official_max_token_window = {
            "gpt-3.5-turbo": 16385,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
        }.get(self.OPENAI_MODEL_ID, 128000)

        max_token_window = int(official_max_token_window * 0.90)

        return max_token_window

    @classmethod
    def load_settings(cls) -> "Settings":
        """
        Tries to load the settings from the ZenML secret store. If the secret does not exist, it initializes the settings from the .env file and default values.

        Returns:
            Settings: The initialized settings object.
        """

        try:
            logger.info("Loading settings from the ZenML secret store.")

            settings_secrets = Client().get_secret("settings")
            settings = Settings(**settings_secrets.secret_values)
        except Exception:
            logger.warning(
                "Failed to load settings from the ZenML secret store. Defaulting to loading the settings from the '.env' file."
            )
            settings = Settings()

        return settings

    def export(self) -> None:
        """
        Exports the settings to the ZenML secret store.
        """

        env_vars = settings.model_dump()
        for key, value in env_vars.items():
            env_vars[key] = str(value)

        client = Client()

        try:
            client.create_secret(name="settings", values=env_vars)
        except EntityExistsError:
            logger.warning(
                "Secret 'scope' already exists. Delete it manually by running 'zenml secret delete settings', before trying to recreate it."
            )


settings = Settings.load_settings()
