# Phase 2 — Kết quả đo

> Repo đo: `fastapi` (cùng clone dùng ở Phase 1, 1136 file, đã re-ingest sạch với UUID tất định).
> Ngày: 2026-08-16. Neo4j + Qdrant chạy qua `docker-compose up neo4j qdrant`.

---

## 0. Phát hiện trước khi làm được gì — domain retrieval chưa tồn tại

Trước khi viết HyDE/BM25/GraphRetriever theo đúng nghĩa đen của prompt, phát hiện hai lỗ hổng chặn đường:

1. **`chunker.py` (Phase 1) chưa từng được gọi.** `codeatlas/ingest.py` chỉ ghi Neo4j; `--no-write` chỉ tắt phần đó. Không có đường nào ghi chunk vào Qdrant.
2. **Domain retrieval vẫn là NeuralTwin cũ** — `domain/embedded_chunks.py` (`EmbeddedChunk` có `platform`, `author_id`, `author_full_name`), `preprocessing/dispatchers.py` (`DataCategory.POSTS/ARTICLES/REPOSITORIES`) — schema cho blog/LinkedIn, không phải code.

Quyết định (đã hỏi bạn trước khi code): migrate domain trước. Xây `codeatlas/domain/code/chunk.py` (`CodeChunkDocument`, tái dùng `VectorBaseDocument` — Qdrant client/collection logic giữ nguyên) + `codeatlas/ingestion/qdrant_writer.py` nối `chunker.py` → embed → Qdrant vào `ingest.py` (`--no-qdrant` để tắt).

**Kết quả:** `fastapi` → **5076 chunk**, embed thật bằng `sentence-transformers/all-MiniLM-L6-v2`, search thật trả kết quả đúng ngữ nghĩa (`FastAPI.post`, `test_jsonable_encoder` cho câu hỏi về JSON response).

### Bug tìm thấy khi verify: Qdrant không idempotent

`CodeChunkDocument.id` mặc định `uuid.uuid4()` — ngẫu nhiên mỗi lần, khác hẳn Neo4j's `MERGE`-on-key. Ingest lại cùng repo sẽ **nhân đôi** mọi chunk thay vì ghi đè. Sửa bằng `uuid5` tất định từ `(repo_id, qualified_name, part)`. Verify thật: ingest 2 lần liên tiếp trên mẫu 3-chunk, số điểm giữ nguyên 3 → 3 (không phải 3 → 6).

---

## 1. HyDE — sinh code, không sinh văn xuôi

Đổi từ mock keyword-match (`if "jwt" in query.lower(): return "JWT is..."`) sang gọi LLM thật, sinh một đoạn Python giả định trả lời câu hỏi, rồi embed đoạn đó.

Ví dụ thật (Groq, `llama-3.3-70b-versatile`):
```
Query: "how does jsonable_encoder convert pydantic models to JSON"
HyDE:  def jsonable_encoder(obj):
           """Convert a Pydantic model to a JSON-serializable dictionary.
           This function recursively traverses the object..."""
```

### Bug tìm thấy khi verify: `GroqProvider` không phải LangChain `Runnable`

`prompt | model` (LCEL) ném `TypeError: Expected a Runnable, callable or dict` — `GroqProvider`/`ModalVLLMProvider` là class tự viết, không kế thừa `Runnable`. Sửa bằng cách bỏ compose `|`, format prompt thủ công rồi gọi `model.invoke(formatted_prompt)` trực tiếp — chạy đồng nhất cho cả Ollama/OpenAI (native LCEL) lẫn Groq/Modal (custom). Áp dụng cho cả `self_query.py` và `query_expanison.py` (cùng lỗi tiềm ẩn, chưa từng bị exercise vì chưa có mock nào bị xoá trước đó).

---

## 2. BM25 tokenizer — 32 test, một quyết định ngược lại thiết kế ban đầu

`getUserById` → `get, user, by, id` (giữ token gốc `getuserbyid`). 14 kiểu identifier (camelCase, snake_case, PascalCase, SCREAMING_SNAKE, số dính chữ, acronym, hybrid...) — vượt yêu cầu tối thiểu 10.

**Quyết định rút lại:** ban đầu có filter stopword tiếng Anh (`a, the, to, in, for, with...`). Chạy thử phát hiện nó xoá mất `to` khỏi `toJSON`, và cùng logic sẽ phá `for_each`, `with_context`, `in_place` — những identifier rất phổ biến trong code, không phải từ đệm văn xuôi. Bỏ hẳn stopword list, để BM25's IDF tự hạ trọng số token phổ biến — đúng việc của nó, không phải việc của tokenizer.

---

## 3. GraphRetriever

Tokenize query → Cypher tìm `Function.name`/`docstring` chứa token, rank theo `match_count` rồi `fan_in`, hydrate content thật từ Qdrant bằng `qualified_name` filter (không dùng lại signature/docstring làm content trừ khi chunk không tồn tại).

Cùng interface `search(query, k) -> list[RetrievedChunk]` như dense/sparse để RRF fuse được — verify: câu hỏi "who calls APIRouter.add_api_route" graph trả đúng `get/post/put/head/options` (tất cả đều gọi `add_api_route` thật, confidence 1.0).

---

## 4. LLM Provider Factory

### GroqProvider — verify bằng lời gọi thật

Đọc header runtime, không hardcode: `limit_requests=1000, limit_tokens=12000` (khớp spec "30 RPM/1K RPD/12K TPM"). Duration string (`"2s"`, `"907ms"`, `"1m30s"`) parse đúng qua regex tự viết, có 9 test case.

Disk cache verify: lần 2 cùng prompt → `0.000s`, không có HTTP request log.

**Bug có sẵn tìm thấy khi verify:** `vllm.py` (code cũ) import `LLMGenerationError` từ `domain/exceptions.py` — **không tồn tại**. Nhánh lỗi chưa từng chạy nên không ai phát hiện. `GroqProvider` viết theo cùng pattern nên lộ ra ngay khi import. Đã thêm class còn thiếu, sửa lợi cho cả hai file.

### ModalVLLMProvider — viết xong, **chưa deploy**

`scripts/deploy_modal_vllm.py` dùng đúng API thật của `modal==1.5.4` đã cài (kiểm tra `inspect.signature` trên `modal.web_server`, `App.function`, `modal.concurrent`, `Image.pip_install` trước khi viết — không đoán theo trí nhớ vì API Modal đổi giữa các version). Client hỗ trợ `generate_batch` (ThreadPoolExecutor, server tự continuous-batch).

**Chưa chạy `modal deploy`** — đó là hành động cấp phát GPU container thật, có phí, cần bạn xác nhận trước. Không tự ý chạy.

### `settings.py` + `.env`

Thêm `GROQ_*`, `MODAL_*`, `LLM_CACHE_DIR`. Dọn luôn: `.env` (không phải `.env.example`) vẫn còn `MOCK_LLM=true` sót lại từ trước Phase 0.5 — set `USE_GROQ=true`, `USE_OLLAMA=false` để `get_llm()` không âm thầm rơi vào Ollama (chưa chắc chạy trên máy này).

---

## 5. Contextual retrieval — verify trên mẫu nhỏ, KHÔNG chạy full corpus

**Ràng buộc thật phải tôn trọng:** Groq free tier = 1000 request/ngày < 5076 chunk của `fastapi`. Chạy full corpus qua Groq sẽ chạm rate limit giữa chừng. Đây chính xác là lý do spec chỉ định Modal cho việc này ("throughput-bound, không rate limit") — chưa deploy Modal nên chưa thể chạy an toàn trên repo lớn.

**Đã làm:** build `ContextualEnricher` + `build_graph_context_batch` (Cypher UNWIND lấy module/parent_class/callers hàng loạt, không N+1 query), thêm `--contextual` flag **tắt mặc định** vào CLI kèm lý do trong `--help`.

**Verify trên repo 3 hàm** (`get_user_by_id` gọi bởi `deactivate_user` và `handle_request`):
```
service.UserService.get_user_by_id:
  "Belongs to service module, retrieves user data, and is called by
   deactivate_user and handle_request."
```
Câu sinh ra khớp đúng graph thật, không bịa. Cache theo `hash(content)` verify: lần 2 `0.002s`, không gọi Groq.

**Việc còn treo:** chạy contextual enrichment full `fastapi` (5076 chunk) chờ Modal deploy.

---

## 6. RRF (k=60) + Cross-encoder — giữ nguyên, tổng quát hoá

Công thức RRF y nguyên (`1/(k+rank+1)`), tách thành `rrf.py` để Phase 5 test/ablate độc lập. Dedupe theo `qualified_name` (không phải text) — verify: hàm `jsonable_encoder` dài, bị chia 9 phần trong Qdrant (Phase 1's chunker), BM25 trả về cả 9 phần riêng biệt trong top-15, nhưng sau RRF fusion chỉ còn **đúng 1** kết quả `jsonable_encoder` (dedup hoạt động đúng).

`Reranker` (cross-encoder) tổng quát hoá từ `list[EmbeddedChunk]` sang generic `ChunkT` (Protocol có `.content`) — dùng được với `RetrievedChunk` mà không cần domain-specific subclass.

---

## 7. Dọn mock — vượt phạm vi rag/*

Theo đúng mục "Phase 2 mới là chỗ xoá" trong roadmap:

| File | Trước | Sau |
|---|---|---|
| `base.py`, `reranking.py`, `self_query.py`, `query_expanison.py`, `hyde_generator.py` | `mock: bool` param + nhánh giả | Bỏ hẳn `mock`, gọi LLM thật |
| `ai_facade.py` | `self._mock`, `_mock_reasoning_response()`, `MOCK_LLM` check 4 chỗ | Bỏ hết, luôn gọi `get_llm()` thật |
| `settings.py` | `MOCK_LLM: bool = False` | Xoá field (không còn ai đọc) |
| `.env` | `MOCK_LLM=true` (2 dòng, sót từ trước Phase 0.5) | Xoá |

**Xoá code chết thay vì de-mock:** `application/graph/ingestor.py` (`GraphIngestor`, regex-based entity extraction cho schema `Entity/MENTIONS` cũ) + `tools/run_graph_ingestion.py`. De-mock file này vô nghĩa — Phase 1 đã thay bằng pipeline AST thật với schema hoàn toàn khác (`Function/Class/CALLS`). Verify: không còn nơi nào import trước khi xoá.

`ContextRetriever` (blog domain cũ, `EmbeddedPostChunk`/`author_id` filter) được **viết lại hoàn toàn** sang domain code — cùng tên class để 3 điểm gọi (`ai_facade.py`, `inference_pipeline_api.py`, `tools/rag.py`) không phải đổi cấu trúc, chỉ đổi tham số (`repo_id` bắt buộc thay vì `mock`).

---

## 8. Verify 10 câu hỏi trên `fastapi` đã index — kết quả thật, không chọn lọc

In đủ top-5 từng retriever + fused + reranked cho cả 10 câu (script + log đầy đủ, không cắt). Tự đánh giá trung thực theo 3 mức:

**Đúng ngay từ #1 (5/10):** `jsonable_encoder`, `FastAPI.get(Path)`, `FastAPI.openapi`, `_get_body_field` (request body), `BackgroundTasks.add_task`.

**Đúng nhưng không phải #1, hoặc lẫn kết quả liên quan (3/10):**
- "dependency injection" → top-1 là `OAuth2PasswordBearer.__init__` (liên quan nhưng lệch), `Depends`/`Security` đúng ở #2-3.
- "who calls add_api_route" → **GraphRetriever tìm đúng** (`get/post/put/head/options`, confidence 1.0 — đây chính xác là câu hỏi structural mà graph phải thắng), nhưng **cross-encoder rerank đẩy nó ra khỏi top-5 cuối**, thay bằng test file ít liên quan hơn. Ghi lại làm dữ liệu cho Phase 5 Bảng A — có thể là bằng chứng cho "+ Cross-encoder rerank" đôi khi làm giảm chất lượng câu hỏi structural, đúng hướng ablation cần đo.
- "CORS middleware" → không tìm ra `CORSMiddleware` (đúng, vì nó ở `starlette`, ngoài index), trả về `FastAPI.middleware`/`build_middleware_stack` — hợp lý nhất trong phạm vi index.

**Không tìm ra đúng (2/10):**
- "what validates the response model" → `fastapi.routing.serialize_response` xuất hiện ở DENSE #4 (0.644) nhưng bị rớt khỏi top-5 sau rerank; final toàn `FastAPI.post/head/patch/put/get` — các hàm bọc HTTP method, không phải hàm validate thật.
- "how does TestClient send a request" → thất bại rõ nhất. `TestClient` là re-export của `starlette.testclient` (ngoài index — đúng), nhưng cả 3 retriever đều không hội tụ về test setup code liên quan; final results gần như không liên quan tới câu hỏi.

**Không tinh chỉnh để 2 câu cuối trông đẹp hơn** — đúng nguyên tắc "kết quả âm vẫn phải báo đúng". Log đầy đủ (10 câu, không cắt) ở `docs/phase2_verification/10_query_verification.txt`. Chạy lại bằng `poetry run python scripts/verify_phase2_retrieval.py --repo-id fastapi` sau khi index.

---

## 9. Test

108 test xanh (Phase 1 kết thúc ở 57):

| File | Số | Phủ |
|---|---|---|
| `test_identifier_tokenizer.py` | 32 | 14 kiểu identifier + edge case |
| `test_llm_providers.py` | 12 | Duration parsing, cache key, message formatting |
| `test_qdrant_writer.py` | 7 | Batch/metadata/embedding + **idempotency** (mới thêm) |
| (Phase 1, không đổi) | 57 | symbol_resolver, repo_loader, chunker |

---

## 10. Việc còn treo — cần bạn quyết

1. **`modal deploy scripts/deploy_modal_vllm.py`** — chưa chạy, cấp phát GPU thật có phí. Cần trước khi: (a) chạy contextual enrichment full `fastapi`, (b) Phase 5 eval theo đúng backend spec chỉ định.
2. **Cross-encoder rerank đẩy đúng kết quả structural ra khỏi top-5** (câu 3, 9) — nghi vấn đáng đo ở Phase 5 Bảng A, không sửa vội vì có thể là đánh đổi hợp lý ở quy mô lớn hơn.
3. **2/10 câu miss hoàn toàn** (CORS ngoài index — chấp nhận được; TestClient — thật sự yếu) — cần gold set annotate tay ở Phase 5 mới đo được có hệ thống, không chỉ 10 câu quan sát bằng mắt.
