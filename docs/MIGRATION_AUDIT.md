# MIGRATION_AUDIT.md — NeuralTwin → CodeAtlas (Phase 0)

> Sinh bởi audit thủ công toàn repo, đối chiếu với `CODEATLAS_SPEC.md` §2.1 (bảng chuyển đổi) và `codeatlas_roadmap.md` Phase 0.
> Cột **Quyết định**: `KEEP` = giữ nguyên logic, `MODIFY` = giữ khung/pattern nhưng nội dung phải viết lại (Phase 1-4), `DELETE` = xoá, không mang sang CodeAtlas.
> Chưa xoá/đổi tên bất cứ thứ gì — bảng này để duyệt trước.

---

## 1. `llm_engineering/domain/` — Domain Layer

| Path | Vai trò hiện tại | Quyết định | Ghi chú |
|---|---|---|---|
| `domain/base/nosql.py` | `NoSQLBaseDocument` — ODM pattern cho Mongo (save/find/bulk_insert) | **KEEP** | Thuần pattern, không gắn digital-twin. Repository/Symbol/Edge sau này có thể kế thừa nếu cần Mongo. |
| `domain/base/vector.py` | `VectorBaseDocument` — ODM pattern cho Qdrant (to_point/search/bulk_insert) | **KEEP** | Thuần pattern, tái dùng trực tiếp cho code chunk embeddings. |
| `domain/base/patterns.py` | `SingletonMeta` — thread-safe singleton | **KEEP** | Infra thuần tuý, dùng cho Neo4jAdapter/Kafka/HydeGenerator. |
| `domain/base/__init__.py` | Re-export | **KEEP** | — |
| `domain/documents.py` | `UserDocument`, `Document`, `RepositoryDocument`, `PostDocument`, `ArticleDocument` | **MODIFY** | Domain-specific (digital twin author/platform). Spec §2.1: `domain/documents/` → `domain/code/` (Repository, Symbol, Edge) — sửa nhiều, không xoá pattern base. |
| `domain/chunks.py` | `Chunk`, `PostChunk`, `ArticleChunk`, `RepositoryChunk` | **MODIFY** | Cùng số phận với documents.py — thay bằng `CodeChunk` (function/method-level). |
| `domain/cleaned_documents.py` | `CleanedDocument` + 3 subclass | **MODIFY** | Ăn theo documents.py. |
| `domain/embedded_chunks.py` | `EmbeddedChunk` + Hierarchical Retrieval (`to_context`, parent/child) | **MODIFY** | Cơ chế Small-to-Big đáng giữ **nguyên logic**, chỉ đổi field domain (author_id/platform → qualified_name/file_path). |
| `domain/queries.py` | `Query`, `EmbeddedQuery` — có `author_id`/`author_full_name` | **MODIFY** | Bỏ khái niệm author, giữ pattern Query/EmbeddedQuery. |
| `domain/dataset.py` | `InstructDataset`, `PreferenceDataset`, HuggingFace export — phục vụ QLoRA finetuning | **DELETE** | Không nằm trong scope CodeAtlas (QA + Refactor agent, không finetune). Ghi chú: cần xác nhận với người dùng — xem mục "Cần quyết định" cuối file. |
| `domain/events.py` | `RawContentEvent`, `ProcessedDocumentEvent` — Kafka event schema | **MODIFY** | Pattern event-driven giữ (Kafka giữ theo spec), nhưng payload (source_url/platform) đổi sang code-ingestion event. |
| `domain/exceptions.py` | `LLMTwinException`, `ImproperlyConfigured` | **KEEP** (đổi tên) | Đổi `LLMTwinException` → tên trung lập hơn (vd `CodeAtlasException`) khi rename, giữ logic. |
| `domain/inference.py` | `DeploymentStrategy`, `Inference` (abstract, SageMaker) + `ReasoningStep*` (NeuralTwin reasoning breakdown) | **MODIFY** | `DeploymentStrategy`/`Inference` abstract gắn SageMaker — có thể DELETE cùng model/. `ReasoningStep*` là Pydantic contract chung chung, đổi docstring "NeuralTwin" → "CodeAtlas", có thể tái dùng cho `AtlasState`. |
| `domain/prompt.py` | `Prompt`, `GenerateDatasetSamplesPrompt` | **MODIFY** | `Prompt` base giữ, `GenerateDatasetSamplesPrompt` gắn dataset.py → xoá cùng. |
| `domain/types.py` | `DataCategory` enum (POSTS/ARTICLES/REPOSITORIES/...) | **MODIFY** | Đổi giá trị enum sang domain code (FUNCTION/CLASS/TEST/MODULE...). |
| `domain/__init__.py` | Re-export | **MODIFY** | Cập nhật theo các module đổi tên. |

---

## 2. `llm_engineering/application/` — Application Layer

### 2.1 `agents/` — 🎯 trọng tâm Phase 0 (xoá Mock Mode)

| Path | Vai trò hiện tại | Quyết định | Ghi chú |
|---|---|---|---|
| `application/agents/research_agent.py` | `ResearchAgent` — **toàn bộ** là `_mock_solve_loop()` hardcode, `use_mock_llm: bool = True` mặc định True | **MODIFY (xoá mock)** | ⚠️ Xoá `use_mock_llm`, `_mock_solve_loop` thì `solve()` không còn logic thật nào để chạy (nhánh `else` hiện tại là `pass`). Sau Phase 0, class này sẽ là **stub rỗng** cho tới khi Phase 3 (LangGraph mode QA) viết lại. Cần xác nhận việc này OK. |
| `application/agents/tools.py` | `AgentTools` — **100% mock**: `search_knowledge_base`, `web_search`, `search_graph` đều trả string hardcode theo keyword-match ("jwt", "oauth2", "fastapi"...), không gọi Qdrant/Neo4j/web thật | **MODIFY (xoá mock)** | Không có implementation thật nào để giữ lại — toàn bộ nội dung 3 method là mock. Sau khi xoá phải quyết định: để method rỗng/raise NotImplementedError, hay xoá file luôn và dựng lại ở Phase 3 (7 tool theo spec §2.4). Đề xuất: xoá file, Phase 3 viết lại từ đầu theo `AtlasState`/tool spec — tránh giữ xác chết. |
| *(thiếu)* `application/agents/__init__.py` | — | — | **Phát hiện phụ**: thư mục `agents/` không có `__init__.py` (có thể do implicit namespace package chạy được nhờ Python 3 hoặc là bug có sẵn, không liên quan migration). Nêu để lưu ý, không phải việc của Phase 0. |

### 2.2 `crawlers/` — 🎯 XOÁ theo yêu cầu item 2

| Path | Vai trò hiện tại | Quyết định | Ghi chú |
|---|---|---|---|
| `application/crawlers/base.py` | `BaseCrawler`, `BaseSeleniumCrawler` (Selenium driver setup) | **DELETE** | Chỉ được dùng bởi 3 crawler mạng xã hội bên dưới. |
| `application/crawlers/dispatcher.py` | `CrawlerDispatcher` — route URL → crawler theo domain | **DELETE** | Phụ thuộc trực tiếp cả 3 crawler bị xoá + `custom_article`. |
| `application/crawlers/github.py` | `GithubCrawler` — clone repo, lưu **toàn bộ nội dung file** như 1 `RepositoryDocument` phẳng (không AST) | **DELETE** | Đây chính là "GitHub profile crawler" trong yêu cầu — khác hoàn toàn ingestion AST-based của Phase 1. Không tái dùng được. |
| `application/crawlers/medium.py` | `MediumCrawler` — Selenium scrape bài Medium | **DELETE** | Digital-twin content collection thuần tuý. |
| `application/crawlers/linkedin.py` | `LinkedInCrawler` — Selenium scrape profile LinkedIn (đã deprecated nội bộ) | **DELETE** | Digital-twin content collection thuần tuý, đã tự đánh dấu deprecated. |
| `application/crawlers/custom_article.py` | `CustomArticleCrawler` — scrape blog bất kỳ qua `AsyncHtmlLoader` | **DELETE** | Fallback crawler của dispatcher, cùng domain "personal content ingestion", không cần cho repo ingestion. |

### 2.3 `graph/`, `rag/`, `networks/`, `utils/`, `dataset/`, `preprocessing/`

| Path | Vai trò hiện tại | Quyết định | Ghi chú |
|---|---|---|---|
| `application/graph/ingestor.py` | `GraphIngestor` — **mock** entity extraction (`_mock_extraction`: regex-match từ khoá "JWT", "FastAPI"...) rồi ghi Neo4j | **DELETE** (thay bằng Phase 1 `graph_builder.py`) | Chứa mock riêng, **ngoài phạm vi bắt buộc xoá của Phase 0** (không nằm trong `agents/`) nhưng cùng bản chất. Cách dùng Neo4j (MERGE, batch) có thể tham khảo, code thì thay hoàn toàn. |
| `application/ai_facade.py` | `NeuralTwinAIFacade` — orchestration layer, có `settings.MOCK_LLM`, `_mock_reasoning_response`, gọi `ResearchAgent`/`ContextRetriever` | **MODIFY** | Đổi tên class `NeuralTwinAIFacade` → `CodeAtlasFacade` (item 3). Có mock riêng ngoài `agents/` — không bắt buộc xoá ở Phase 0 nhưng nên ghi nhận nợ kỹ thuật. |
| `application/rag/base.py` | `RAGStep` (abstract, có `mock: bool` param), `PromptTemplateFactory` | **KEEP** (pattern) | Interface chung, giữ. |
| `application/rag/retriever.py` | `ContextRetriever` — HyDE + rerank + Hierarchical expansion; RRF method tồn tại nhưng **dead code** (dense-only đang chạy, sparse bị comment "SHOWCASE") | **MODIFY** | Spec giữ RRF nguyên — nhưng thực tế RRF chưa từng được gọi trong pipeline hiện tại. Cần biết trước khi Phase 2 "giữ nguyên RRF": nó phải được **kích hoạt thật**, không phải đã hoạt động sẵn. Toàn bộ gắn `EmbeddedPostChunk/ArticleChunk/RepositoryChunk` → đổi sang code chunk. |
| `application/rag/hyde_generator.py` | `HydeGenerator` — **100% mock**, trả string hardcode theo keyword ("jwt"/"oauth"/"kafka") | **MODIFY** | Spec Phase 2 mục 1: đổi hoàn toàn sang sinh code snippet giả định qua LLM thật. Hiện tại chưa từng gọi LLM thật (`_mock_generate` luôn được gọi kể cả khi `mock=False`). |
| `application/rag/reranking.py` | `Reranker` dùng `CrossEncoderModelSingleton` thật (không mock nếu `mock=False`) | **KEEP** (pattern) | Model thật, chỉ input/output đổi sang code chunk. |
| `application/rag/self_query.py` | `SelfQuery` — trích `author_full_name` từ câu hỏi, tạo/lookup `UserDocument` | **DELETE** | Khái niệm "tác giả" không tồn tại trong CodeAtlas domain. |
| `application/rag/query_expanison.py` | `QueryExpansion` — sinh N câu hỏi biến thể qua LLM | **KEEP** (pattern) | Domain-agnostic, tái dùng được nguyên vẹn. |
| `application/rag/prompt_templates.py` | `QueryExpansionTemplate`, `SelfQueryTemplate` | **MODIFY** | `SelfQueryTemplate` xoá cùng self_query.py; `QueryExpansionTemplate` giữ. |
| `application/networks/embeddings.py` | `EmbeddingModelSingleton`, `CrossEncoderModelSingleton` | **KEEP** | Thuần infra, không đổi. |
| `application/networks/base.py` | Re-export `SingletonMeta` | **KEEP** | — |
| `application/utils/llm_factory.py` | `get_llm()` — factory OpenAI/Ollama/vLLM | **MODIFY** | Spec §2.1: thêm `GroqProvider`, `ModalVLLMProvider` — mức "Vừa". |
| `application/utils/misc.py` | `flatten`, `batch`, `compute_num_tokens` | **KEEP** | Generic utils. |
| `application/utils/split_user_full_name.py` | Tách "First Last" cho UserDocument | **DELETE** | Chỉ dùng bởi self_query.py và steps/etl (cả hai đều bị xoá). |
| `application/preprocessing/dispatchers.py` | `CleaningDispatcher`, `ChunkingDispatcher`, `EmbeddingDispatcher` — factory theo `DataCategory` | **MODIFY** | Pattern factory tốt, giữ; nhánh theo POSTS/ARTICLES/REPOSITORIES → đổi theo loại code entity (hoặc bỏ nếu chunk theo function là chunking duy nhất). |
| `application/preprocessing/cleaning_data_handlers.py` | Clean Post/Article/Repository → strip HTML thô | **DELETE** | Code không cần "clean HTML" — cần AST parse (Phase 1 `python_parser.py`), khác bản chất hoàn toàn. |
| `application/preprocessing/chunking_data_handlers.py` | Chunk Post/Article/Repository theo char-length + parent_content cho Hierarchical Retrieval | **MODIFY** | Ý tưởng "lưu parent_content cho Small-to-Big" đáng giữ; cơ chế chunk (char/token splitter) thay bằng "1 chunk = 1 function" (spec §2.5/§Phase1 `chunker.py`). |
| `application/preprocessing/embedding_data_handlers.py` | Embed Post/Article/Repository/Query chunks | **MODIFY** | Pattern `embed_batch` giữ, field mapping đổi theo `CodeChunk`. |
| `application/preprocessing/operations/chunking.py` | `chunk_text`, `chunk_article` — char/sentence splitter | **DELETE** (thay bằng chunker.py mới) | Không phù hợp chunk theo function/AST. |
| `application/preprocessing/operations/cleaning.py` | `clean_text` — regex generic | **KEEP** | Đủ generic, có thể tái dùng cho docstring cleanup nếu cần. Có unit test (`tests/unit/.../test_cleaning.py`) — giữ cả hai. |
| `application/dataset/*.py` (4 file: `__init__`, `constants.py`, `generation.py`, `output_parsers.py`, `utils.py`) | Sinh instruct/preference dataset cho QLoRA finetuning; `constants.py` chứa `MOCKED_RESPONSE_INSTRUCT` | **DELETE** | Toàn bộ phục vụ finetuning showcase, ngoài scope CodeAtlas (QA + Refactor, không finetune). Có mock riêng (`MOCKED_RESPONSE_INSTRUCT`) ngoài `agents/`, không bắt buộc xoá ở Phase 0 theo nghĩa đen nhưng nên xoá cùng cả module vì cả module bị loại. |

---

## 3. `llm_engineering/infrastructure/` — Infrastructure Layer

| Path | Vai trò hiện tại | Quyết định | Ghi chú |
|---|---|---|---|
| `infrastructure/db/mongo.py` | `MongoDatabaseConnector` (Singleton) | **KEEP** | Thuần infra. |
| `infrastructure/db/qdrant.py` | `QdrantDatabaseConnector` | **KEEP** | Thuần infra, spec: "Qdrant client giữ, đổi chunking". |
| `infrastructure/graph/neo4j_adapter.py` | `Neo4jAdapter` — driver wrapper, `execute_query` | **KEEP** | Spec: "Neo4jAdapter giữ, đổi Cypher template" — file này không chứa Cypher cụ thể, chỉ transport. Giữ nguyên 100%. |
| `infrastructure/llm/vllm.py` | `VLLMClient` — OpenAI-compatible client, LangChain-invoke-compatible | **KEEP** | Tái dùng được cho `ModalVLLMProvider` hoặc giữ song song. |
| `infrastructure/streaming/kafka_config.py` | `KafkaProducer`, `KafkaConsumer` (Singleton) | **KEEP** | Spec: Kafka giữ nguyên. |
| `infrastructure/streaming/producers/data_collection_producer.py` | `DataCollectionProducer` — publish raw content lên topic `raw_content_stream` | **MODIFY** | Pattern publish giữ, nguồn dữ liệu đổi từ "crawled content" sang "AST-parsed code entity". |
| `infrastructure/streaming/consumers/document_processor_consumer.py` | Consume `raw_content_stream` → clean_text → forward `embedding_requests` | **MODIFY** | Logic clean_text (HTML) không còn phù hợp — nhưng khung consumer/producer Kafka giữ nguyên. |
| `infrastructure/streaming/consumers/embedding_consumer.py` | Consume `embedding_requests` — **`mock_embed()` trả vector ngẫu nhiên**, "simulate Qdrant upsert" (không thật sự lưu) | **MODIFY** | File này có mock riêng, ngoài `agents/`. Cần nối thật vào `EmbeddingModelSingleton` + Qdrant khi viết lại pipeline ingestion Kafka (nếu dùng). |
| `infrastructure/security/jwt.py` | JWT bearer verify — `SECRET_KEY` **hardcode trong code**, không đọc từ `settings` | **KEEP** (pattern), lưu ý bảo mật | Không phải việc migration nhưng đáng note: secret hardcode là smell có sẵn, không phải do Phase 0 gây ra. |
| `infrastructure/security/rate_limiter.py` | Redis sliding-window rate limiter, "fail open" nếu Redis down | **KEEP** | Generic, spec giữ Redis. |
| `infrastructure/monitoring/decorators.py` | `track_request_metrics` decorator | **KEEP** | Generic. |
| `infrastructure/monitoring/metrics.py` | Prometheus `Counter`/`Histogram`/`Gauge` definitions | **KEEP** | Generic, spec giữ Prometheus/Grafana/Jaeger. |
| `infrastructure/opik_utils.py` | `configure_opik()` — Comet/Opik tracing setup | **KEEP** | Có thể tái dùng để trace agent (JSONL + Prometheus theo spec Phase 3). |
| `infrastructure/files_io.py` | `JsonFileManager` — read/write JSON generic | **KEEP** | Generic, hữu ích cho export eval / gold set (Phase 5). |
| `infrastructure/api/__init__.py` | Docstring: "FastAPI routers for **NeuralTwin** capabilities" | **MODIFY** | Chỉ cần đổi tên trong docstring (item 3). |
| `infrastructure/api/facade_controller.py` | Router `/api/v1/ai/reasoning-breakdown`, import `NeuralTwinAIFacade` | **MODIFY** | Đổi tên import theo `ai_facade.py`, endpoint có thể giữ hoặc đổi route prefix. |
| `infrastructure/inference_pipeline_api.py` | FastAPI app chính — `/rag`, `/metrics`; có **"MOCK MODE FOR PORTFOLIO"** comment + `os.getenv("MOCK_LLM")` ở `call_llm_service`, `stream_rag`, `rag()` | **MODIFY** | Đây là entrypoint chính (`uvicorn llm_engineering.infrastructure.inference_pipeline_api:app` trong docker-compose và k8s). Có Mock Mode riêng, **ngoài phạm vi bắt buộc của item 2** (chỉ yêu cầu xoá mock trong `agents/`) nhưng liên quan trực tiếp — nêu rõ để quyết định có xoá cùng đợt hay để Phase 2/3. Cấu trúc FastAPI+SSE giữ nguyên theo spec. |
| `infrastructure/aws/**` (7 file: `deploy/autoscaling_sagemaker_endpoint.py`, `deploy/delete_sagemaker_endpoint.py`, `deploy/huggingface/{config,run,sagemaker_huggingface}.py`, `roles/create_execution_role.py`, `roles/create_sagemaker_role.py`) | Toàn bộ SageMaker/HuggingFace endpoint deploy tooling | **DELETE** | Spec §0 TL;DR: LLM serving = Groq (demo) + Modal+vLLM (eval) — không dùng SageMaker. Không nằm trong "giữ nguyên" list của spec §2.1. Cần xác nhận (xem mục cuối). |

---

## 4. `llm_engineering/model/` — Model Layer (Finetuning/Evaluation/Inference showcase)

| Path | Vai trò hiện tại | Quyết định | Ghi chú |
|---|---|---|---|
| `model/finetuning/finetune.py`, `finetuning/sagemaker.py` | QLoRA finetune script (`unsloth`) + SageMaker training job trigger | **DELETE** | Ngoài scope CodeAtlas (không finetune LLM). |
| `model/evaluation/evaluate.py`, `evaluation/sagemaker.py` | Đánh giá model đã finetune qua SageMaker Processor | **DELETE** | Cùng lý do — đây là eval cho **finetuned model**, khác hoàn toàn `codeatlas/eval/` (Phase 5, đánh giá retrieval/agent). |
| `model/inference/inference.py`, `inference/run.py`, `inference/test.py` | `LLMInferenceSagemakerEndpoint`, `InferenceExecutor` — gọi SageMaker endpoint | **DELETE** | Được dùng bởi `inference_pipeline_api.py` và `ai_facade.py` hiện tại — xoá sẽ phá 2 chỗ đó, cần thay bằng Groq/vLLM/Ollama trước. |
| `model/utils.py`, `model/Readme.md` | Utils cho model layer | **DELETE** (nếu model/ bị xoá toàn bộ) | Phụ thuộc các file trên. |

---

## 5. `pipelines/` (ZenML) & `steps/`

| Path | Vai trò hiện tại | Quyết định | Ghi chú |
|---|---|---|---|
| `pipelines/digital_data_etl.py` | `get_or_create_user` + `crawl_links` | **DELETE** | Phụ thuộc trực tiếp crawlers bị xoá. |
| `pipelines/end_to_end_data.py` | Orchestrate ETL → feature engineering → dataset gen | **DELETE** | Phụ thuộc digital_data_etl + generate_datasets, cả hai đều xoá. |
| `pipelines/feature_engineering.py` | query warehouse → clean → chunk+embed → load vector DB | **MODIFY** | Khung ZenML `@pipeline` giữ (spec: "ZenML giữ, đổi step" 🟢), nội dung step thay bằng ingestion pipeline Phase 1. |
| `pipelines/generate_datasets.py` | Sinh instruct/preference dataset | **DELETE** | Ăn theo `application/dataset/` bị xoá. |
| `pipelines/training.py` | Trigger QLoRA training (đã có sẵn `skip_execution` flag "Portfolio Mode") | **DELETE** | Ngoài scope finetuning. |
| `pipelines/evaluating.py` | Evaluate model qua SageMaker | **DELETE** | Ăn theo model/evaluation bị xoá — thay bằng `codeatlas/eval/run_ablation.py` (Phase 5, không phải ZenML pipeline). |
| `pipelines/export_artifact_to_json.py` | Export ZenML artifact → JSON | **KEEP** | Generic, hữu ích để export gold set/eval results (Phase 5). |
| `pipelines/__init__.py` | Re-export tất cả pipeline trên | **MODIFY** | Cập nhật theo danh sách pipeline còn lại. |
| `steps/etl/crawl_links.py`, `steps/etl/get_or_create_user.py` | Step ETL dùng `CrawlerDispatcher`/`UserDocument` | **DELETE** | Phụ thuộc trực tiếp thứ đã xoá. |
| `steps/feature_engineering/*.py` (4 file) | `clean_documents`, `load_to_vector_db`, `query_data_warehouse`, `chunk_and_embed` | **MODIFY** | Khung `@step` giữ; nội dung đổi theo domain code (Phase 1/2). `load_to_vector_db` khá generic (dùng `VectorBaseDocument.group_by_class`) — có thể giữ gần như nguyên. |
| `steps/generate_datasets/*.py` (5 file) | Sinh prompt/dataset instruct+preference, push HuggingFace | **DELETE** | Ăn theo application/dataset + domain/dataset bị xoá. |
| `steps/training/train.py` | Trigger `model.finetuning.sagemaker` | **DELETE** | Ăn theo model/finetuning bị xoá. |
| `steps/evaluating/evaluate.py` | Trigger `model.evaluation.sagemaker` | **DELETE** | Ăn theo model/evaluation bị xoá. |
| `steps/export/serialize_artifact.py`, `steps/export/to_json.py` | Serialize + write JSON, dùng `JsonFileManager` | **KEEP** | Generic, độc lập với domain. |

---

## 6. `configs/*.yaml` (ZenML pipeline configs)

| Path | Vai trò hiện tại | Quyết định | Ghi chú |
|---|---|---|---|
| `configs/digital_data_etl_maxime_labonne.yaml` | Params cho `digital_data_etl` (tên người, link blog cá nhân) | **DELETE** | "Config liên quan" tới crawler theo item 2. |
| `configs/digital_data_etl_paul_iusztin.yaml` | Params cho `digital_data_etl` (tên người dùng hiện tại, link cá nhân) | **DELETE** | Cùng lý do. |
| `configs/end_to_end_data.yaml` | Params cho `end_to_end_data` (gọi digital_data_etl bên trong) | **DELETE** | Phụ thuộc pipeline đã xoá. |
| `configs/feature_engineering.yaml` | `author_full_names` list | **MODIFY** | Đổi param sang `repo_url`/`repo_path` khi Phase 1/2 định nghĩa pipeline mới. |
| `configs/generate_instruct_datasets.yaml`, `configs/generate_preference_datasets.yaml` | Params dataset generation, push HuggingFace | **DELETE** | Ăn theo pipeline generate_datasets bị xoá. |
| `configs/training.yaml` | Params QLoRA finetuning | **DELETE** | Ăn theo pipeline training bị xoá. |
| `configs/evaluating.yaml` | `is_dummy` flag cho SageMaker eval | **DELETE** | Ăn theo pipeline evaluating bị xoá. |
| `configs/export_artifact_to_json.yaml` | Danh sách artifact cần export | **MODIFY** | Giữ pattern, đổi danh sách artifact (gold set, eval results...). |

Toàn bộ 9 file `configs/*.yaml` cũng đang hardcode `parent_image: ...amazonaws.com/zenml-rlwlcs:latest` (ECR image riêng của tác giả gốc) — cần thay dù pipeline có giữ hay không.

---

## 7. `tools/`

| Path | Vai trò hiện tại | Quyết định | Ghi chú |
|---|---|---|---|
| `tools/run.py` | CLI (`click`) chạy các ZenML pipeline | **MODIFY** | Khung CLI giữ, danh sách pipeline import cần cập nhật theo pipelines còn lại. |
| `tools/ml_service.py` | `uvicorn.run("tools.ml_service:app", ...)` — chạy FastAPI app | **KEEP** | Generic launcher, chỉ đổi nếu đổi module path (`llm_engineering` → `codeatlas`). |
| `tools/rag.py` | Demo script gọi `ContextRetriever` với query hardcode "My name is Paul Iusztin..." | **DELETE** | Query mẫu gắn chặt digital-twin persona; thay bằng demo CLI Phase 6. |
| `tools/agent_demo.py` | Demo CLI cho `ResearchAgent`, **tự mock luôn cả ZenML Client** (`sys.modules["zenml.client"] = mock_zenml`) để né lỗi kết nối | **DELETE** | Toàn bộ mục đích là demo agent mock hiện tại — hết ý nghĩa sau khi `ResearchAgent` bị strip mock. Viết lại ở Phase 6. |
| `tools/data_warehouse.py` | Export `ArticleDocument/PostDocument/RepositoryDocument/UserDocument` ra JSON | **DELETE** | Domain model bị xoá/modify hoàn toàn. |
| `tools/run_graph_ingestion.py` | Chạy `GraphIngestor(mock=True)` với "Mock Data" hardcode (4 câu về JWT/FastAPI/Kafka/Redis) | **DELETE** | Tên file/mục đích chính là demo mock ingestion — thay hoàn toàn bởi Phase 1 `python -m codeatlas.ingest`. |

---

## 8. `tests/`

| Path | Vai trò hiện tại | Quyết định | Ghi chú |
|---|---|---|---|
| `tests/conftest.py` | Thêm project root vào `sys.path` | **KEEP** | Generic. |
| `tests/unit/infrastructure/test_mongo.py` | Test `MongoDatabaseConnector` (mock qua `unittest.mock`, không phải "Mock Mode" của agent) | **KEEP** | Không liên quan digital-twin domain. |
| `tests/unit/application/preprocessing/operations/test_cleaning.py` | Test `clean_text()` | **KEEP** | Hàm được giữ (mục 2.3). |
| `tests/integration/integration_example_test.py` | Test placeholder vô nghĩa (`"x" == "x"`) | **KEEP** | Không hại gì, có thể xoá sau nếu muốn dọn nhưng không bắt buộc. |
| `tests/load/locustfile.py` | Load test POST `/rag` với query mẫu "How does OAuth2 compare to JWT?" | **MODIFY** | Endpoint `/rag` có thể vẫn tồn tại dạng nào đó — pattern Locust giữ, query mẫu đổi sang câu hỏi code. |

**Không tìm thấy test nào import trực tiếp `crawlers/`, `medium`, `linkedin`, hoặc `github.py`** (đã `grep -ril` toàn bộ `tests/`) → không có "test liên quan" nào cần xoá theo nghĩa đen của item 2; chỉ có **configs** liên quan (mục 6).

---

## 9. `training/`, `evaluation/`, `code_snippets/`

| Path | Vai trò hiện tại | Quyết định | Ghi chú |
|---|---|---|---|
| `training/lora_kaggle_training.py`, `training/README_TRAINING.md` | Script QLoRA finetune Llama 3 trên Kaggle T4 GPU | **DELETE** | Ngoài scope CodeAtlas. |
| `evaluation/rag_metrics.py` | `RAGMetrics` — Precision@K, Recall@K (đã viết sẵn, generic, không phụ thuộc domain) | **KEEP — tái dùng cao** | Trùng khớp trực tiếp với `codeatlas/eval/metrics.py` (Phase 5 spec §3.2: Precision@k, Recall@k, MRR, nDCG@k). Nên **di chuyển vào `codeatlas/eval/`** ở Phase 5 thay vì viết lại từ đầu. |
| `code_snippets/03_custom_odm_example.py` | Demo `UserDocument`/`ArticleDocument` | **DELETE** | Tutorial snippet gắn domain cũ. |
| `code_snippets/03_orm.py` | Demo SQLAlchemy ORM độc lập, không import gì từ `llm_engineering` | **DELETE** | Snippet giáo trình từ sách gốc, không liên quan runtime. |
| `code_snippets/08_instructor_embeddings.py`, `08_text_embeddings.py`, `08_text_image_embeddings.py` | Demo `sentence-transformers` độc lập | **DELETE** | Snippet giáo trình, không được import bởi bất kỳ module runtime nào. |

---

## 10. Cấu hình dự án & hạ tầng (rename NeuralTwin → CodeAtlas — item 3)

| Path | Vai trò hiện tại | Quyết định | Ghi chú |
|---|---|---|---|
| `pyproject.toml` | `[tool.poetry] name = "llm-engineering"` (chưa từng là "neuraltwin") | **MODIFY** | Đổi `name` → `codeatlas`. Không chứa chuỗi "NeuralTwin" hiện tại, nhưng cần đổi theo yêu cầu item 3 + rà lại toàn bộ dependency list (torch/unsloth/peft/trl cho finetuning có thể bỏ nếu model/ bị xoá). |
| `README.md` | Chứa nhiều chỗ nhắc "NeuralTwin" | **MODIFY** | Viết lại hoàn toàn ở Phase 6 theo spec, không chỉ tìm-thay tên. |
| `docker-compose.yml` | Container name prefix `llm_engineering_*` (không phải "neuraltwin"), service `rag-api` có `MOCK_LLM=true` | **MODIFY** | Đổi prefix container theo `codeatlas` nếu muốn nhất quán; **quan trọng hơn**: `MOCK_LLM=true` hardcode ở đây sẽ mâu thuẫn với việc xoá Mock Mode — cần quyết định đồng bộ. |
| `k8s/deployment.yaml`, `k8s/service.yaml`, `k8s/hpa.yaml` | `namespace: neural-twin`, `image: neuraltwin/api:latest`, `neuraltwin/consumer:latest`, host `api.neuraltwin.com` | **MODIFY** | Đổi namespace/image/host sang `codeatlas`. |
| `k8s/vllm-deployment.yaml` | `namespace: default`, không có "neuraltwin" | **KEEP** (không cần đổi tên) | Generic vLLM deployment, không gắn tên dự án. |
| `Makefile` | Header comment "LLM Twin - Portfolio Edition", target `start`/`stop`/`test`... | **MODIFY** | Đổi header, targets giữ nguyên logic. |
| `llm_engineering/domain/inference.py` | Docstring "four canonical phases of the **NeuralTwin** reasoning process" | **MODIFY** | Đổi tên trong docstring (đã liệt ở mục 1). |
| `llm_engineering/infrastructure/api/__init__.py` | Docstring "FastAPI routers for **NeuralTwin** capabilities" | **MODIFY** | Đã liệt ở mục 3. |
| `llm_engineering/infrastructure/api/facade_controller.py` | Import `NeuralTwinAIFacade`, module docstring nhắc NeuralTwin nhiều lần | **MODIFY** | Đã liệt ở mục 3. |
| `llm_engineering/application/ai_facade.py` | Class `NeuralTwinAIFacade`, docstring + system prompt "You are NeuralTwin — an expert AI reasoning engine" | **MODIFY** | Đã liệt ở mục 2.3 — đổi tên class **và** nội dung system prompt gửi cho LLM. |
| `CONTRIBUTING.md`, `AGENTS.md`, `CLAUDE.md`, `PORTFOLIO_SUMMARY.md` | Docs nhắc "NeuralTwin" | **MODIFY** | Không nằm trong danh sách bắt buộc của item 3 (chỉ liệt pyproject/README/docker-compose/k8s/Makefile) — nhưng để nhất quán nên cập nhật cùng đợt hoặc ở Phase 6. |
| `docs/ARCHITECTURE.md`, `docs/CODEBASE_CONTEXT.md`, `docs/RAG_PIPELINE.md`, `docs/API_REFERENCE.md`, `docs/INTERVIEW_GUIDE.md` | Docs mô tả kiến trúc NeuralTwin (một phần là tài liệu portfolio mang tính minh hoạ, không khớp 100% code thật — đã ghi nhận trước đó) | **DELETE hoặc viết lại hoàn toàn ở Phase 6** | Không sửa từng chữ — các doc này mô tả sai kiến trúc thật khá nhiều (AWS ECS/DocumentDB/Auth0 không tồn tại trong code). Giữ lại sẽ gây nhiễu cho interviewer đọc repo. |
| **Toàn bộ import path** `from llm_engineering...` / `from pipelines...` / `from steps...` / `from tools...` (~90+ file Python) | Package gốc tên `llm_engineering` | **MODIFY (mass rename)** | ⚠️ Đây là thay đổi có **blast radius lớn nhất** trong Phase 0 — đổi tên package `llm_engineering/` → `codeatlas/` kéo theo sửa import ở gần như mọi file `.py` trong repo, cộng cả `Dockerfile`, `docker-compose.yml` (`command: uvicorn llm_engineering...`), `k8s/*.yaml` (`command: ["python", "-m", "llm_engineering...`), `.pre-commit-config.yaml`, `ruff.toml` nếu có path reference. Cần xác nhận trước khi thực thi (xem mục cuối). |

---

## 11. Còn lại (chưa nhắc ở trên)

| Path | Vai trò hiện tại | Quyết định | Ghi chú |
|---|---|---|---|
| `.env`, `.env.example` | Env vars: `MOCK_LLM`, `LINKEDIN_USERNAME/PASSWORD`, `OPENAI_API_KEY`, `HF_MODEL_ID=mlabonne/TwinLlama-3.1-8B-DPO`... | **MODIFY** | Xoá `LINKEDIN_*`; `MOCK_LLM` cần quyết định đồng bộ với việc xoá Mock Mode; `HF_MODEL_ID` gắn tên model "TwinLlama" — đổi. |
| `llm_engineering/settings.py` | `Settings(BaseSettings)` — có `MOCK_LLM`, `SKIP_TRAINING`, `LINKEDIN_USERNAME/PASSWORD`, AWS/SageMaker fields, `HF_MODEL_ID` | **MODIFY** | Trung tâm cấu hình — xoá field crawler/finetuning liên quan, thêm field Groq/Modal (Phase 2). |
| `monitoring/prometheus/prometheus.yml` | Scrape config Prometheus | **KEEP** | Generic, không gắn tên dự án (chưa kiểm tra target hostnames — nên rà lại nếu đổi tên service). |
| `.github/workflows/ci.yaml` | Lint/format/test qua Poetry | **KEEP** | Generic, không gắn NeuralTwin. |
| `.github/workflows/cd.yaml` | Build & push Docker image lên ECR (chỉ chạy nếu có AWS creds) | **KEEP** | Generic, gắn AWS ECR nhưng đây là CI/CD deploy chung, không phải SageMaker finetuning — giữ được dù xoá `infrastructure/aws/`. |
| `Dockerfile` | Build image | **MODIFY** | Cần sửa nếu đổi package path `llm_engineering` → `codeatlas`. |
| `setup.sh`, `.pre-commit-config.yaml`, `ruff.toml`, `.python-version` | Tooling | **KEEP** | Generic, không gắn domain. |
| `.gitnexus/`, `.claude/`, `.vscode/`, `.zen/` | Tooling nội bộ (GitNexus index, Claude skills, editor config) | **KEEP** | Ngoài phạm vi migration, sẽ tự cập nhật khi code đổi. |

---

## Tổng kết theo yêu cầu Phase 0

### ✅ Khớp trực tiếp với item 2 (XOÁ) — sẵn sàng thực thi
- `llm_engineering/application/crawlers/` (6 file: base, dispatcher, github, medium, linkedin, custom_article)
- `steps/etl/crawl_links.py`, `steps/etl/get_or_create_user.py`, `steps/etl/__init__.py` (cập nhật rỗng)
- `configs/digital_data_etl_maxime_labonne.yaml`, `configs/digital_data_etl_paul_iusztin.yaml`, `configs/end_to_end_data.yaml` (config liên quan crawler)
- Test liên quan: **không có** (đã xác nhận bằng grep)
- Mock Mode trong `agents/`: `research_agent.py` (`use_mock_llm`, `_mock_solve_loop`), `tools.py` (toàn bộ — không có logic thật để giữ lại)

### ⚠️ Cần quyết định trước khi tôi thực thi (nằm ngoài phạm vi rõ ràng của prompt)

1. **Mock Mode ngoài `agents/`** — tồn tại ở 8 chỗ khác: `ai_facade.py`, `application/rag/{retriever,hyde_generator,reranking,self_query,query_expanison}.py`, `application/graph/ingestor.py`, `infrastructure/inference_pipeline_api.py`, `infrastructure/streaming/consumers/embedding_consumer.py`, `application/dataset/constants.py`, `tools/agent_demo.py`, `tools/run_graph_ingestion.py`, `docker-compose.yml` (`MOCK_LLM=true`). Yêu cầu chỉ bắt buộc `grep -ri mock llm_engineering/application/agents/` phải rỗng — **có xoá luôn các chỗ này trong Phase 0, hay để nguyên cho Phase 2/3?** Xoá hết sẽ khiến `/rag` endpoint và `ai_facade` hỏng ngay (không còn nhánh nào chạy được) cho tới khi có LLM provider thật.
2. **`ResearchAgent`/`AgentTools` sau khi xoá mock sẽ là stub rỗng** (không có bất kỳ logic thật nào thay thế cho tới Phase 3). Đồng ý để nó "trống" tạm thời, hay muốn tôi xoá hẳn 2 file này và để `agents/` rỗng cho tới Phase 3 dựng lại?
3. **Các subsystem ngoài scope CodeAtlas nhưng KHÔNG nằm trong danh sách XOÁ tường minh của prompt**: `llm_engineering/model/` (finetuning/evaluation/inference SageMaker), `llm_engineering/infrastructure/aws/`, `application/dataset/`, `domain/dataset.py`, `pipelines/{training,generate_datasets,evaluating}.py`, `steps/{training,generate_datasets,evaluating}/`, `configs/{training,generate_instruct_datasets,generate_preference_datasets,evaluating}.yaml`, `training/`, `code_snippets/`. Đây là ~30 file phục vụ QLoRA finetuning showcase của NeuralTwin gốc — theo `CODEATLAS_SPEC.md` §2.1 chúng không nằm trong danh sách "giữ nguyên" và toàn bộ TL;DR của spec (Groq/Modal, không SageMaker) ngụ ý loại bỏ, nhưng **prompt Phase 0 không liệt kê chúng trong mục XOÁ**. Xoá cùng đợt sẽ dọn repo sạch hơn nhiều cho mục đích phỏng vấn; không xoá thì an toàn hơn (không risk xoá nhầm thứ còn dùng) nhưng để lại ~30 file "xác chết" đến tận Phase 6. Bạn quyết định.
4. **Mass rename `llm_engineering` → `codeatlas`** (item 3, "mọi import path"): đây là thay đổi cơ học nhưng đụng tới ~90+ file. Có muốn tôi thực hiện trong Phase 0 này luôn (đúng theo yêu cầu), hay tách thành một bước riêng sau khi các quyết định 1-3 ở trên đã chốt — để tránh phải rename hai lần (rename rồi lại xoá thêm file sau).
