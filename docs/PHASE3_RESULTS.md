# Phase 3 — Kết quả đo

> Repo đo: `fastapi` (đã index từ Phase 2, 5076 chunk Qdrant + 4959 Function trong Neo4j).
> Ngày: 2026-08-18. Neo4j + Qdrant chạy qua `docker-compose up neo4j qdrant`. LLM: Groq `openai/gpt-oss-120b`.

---

## 0. Hai chặn hạ tầng phải giải quyết trước khi viết dòng code nào

### `langgraph==1.2` (spec ghi) không cài được cùng ZenML

`langgraph-sdk` (phụ thuộc bắc cầu của mọi `langgraph>=0.5`) đòi `orjson>=3.11.5`; `zenml[server]==0.74.0` (luật cứng — giữ nguyên ZenML) ghim `orjson>=3.10,<3.11`. `poetry add langgraph` với version bất kỳ `>=0.5` fail resolve. Downgrade xuống bản mới nhất cài được: **`langgraph==0.4.5`** + `langgraph-checkpoint-sqlite==2.0.11` + `langgraph-checkpoint-postgres==2.0.25` + `psycopg[binary]`. Verify import thật đủ API cần cho Phase 3/4: `StateGraph`, `add_conditional_edges`, `interrupt`/`Command`, `SqliteSaver`, `PostgresSaver`.

Hệ quả kỹ thuật: `StateGraph.add_node` ở 0.4.5 **từ chối tên node trùng tên field trong state**. Node dự định đặt tên `"plan"` phải đổi thành `"planning"` — field `AtlasState.plan` giữ nguyên theo spec.

### Model `llama-3.3-70b-versatile` (spec §2.6) không còn tồn tại

Gọi thật → 404. Groq đã đổi catalog. Tra `client.models.list()` thật thay vì đoán, chọn `openai/gpt-oss-120b` — đây là **reasoning model**, tốn token vào field `reasoning` ẩn trước `content`; `max_tokens` thấp có thể bị cắt trước khi có nội dung. TPM free-tier của model này chỉ **8000** (thấp hơn nhiều so với con số 12K TPM cũ trong spec cho Llama).

---

## 1. Bảy tool (`tools.py`) — verify sống từng cái trên `fastapi`

| Tool | Kết quả verify |
|---|---|
| `search_symbol` | Tìm đúng, difflib gợi ý đúng khi miss (vd `authX` → `authenticate`) |
| `get_callers` depth=1, depth=2 | Đúng, depth=2 trả nhiều hơn depth=1 hợp lý |
| `get_callees` | Đúng |
| `impact_analysis` | 8 impacted_symbols, 2 affected_tests cho `add_api_route` |
| `list_module_structure` | Lọc đúng — chỉ hàm/class trực tiếp trong module, loại method lồng trong class |
| `read_source` | Hydrate đúng từ Qdrant chunk, nối nhiều part khi hàm dài |
| `search_semantic` | Dùng `ContextRetriever` Phase 2 nguyên vẹn |

### Bug tìm thấy: `min(length(c))` sai kiểu Cypher

`get_callers`/`get_callees` dùng pattern biến-độ-dài `[c:CALLS*1..N]` — `c` là `LIST<RELATIONSHIP>`, không phải `PATH`. `length()` đòi `PATH`. Fake-adapter unit test **không bắt được** lỗi này (chỉ test shaping, không chạy Cypher thật) — chỉ lộ ra khi gọi Neo4j thật. Sửa: `size(c)`. **Bài học ghi vào quy trình:** mọi Cypher mới phải verify sống ít nhất 1 lần trước khi tin fake-adapter test.

---

## 2. Hai orchestrator — cùng tool, cùng retrieval, khác điều phối

### `custom_react.py` (~80 dòng logic)

Vòng lặp Thought/Action/Action Input/Observation dạng văn bản thuần, không dùng function-calling API. Verify sống: trả lời đúng, trích dẫn đúng dòng thật cho câu "who calls add_api_route" ngay từ vòng chạy đầu tiên.

### `langgraph_qa.py`

Đồ thị **fan-out thật của LangGraph** (3 node `retrieve_dense/sparse/graph` chạy song song qua `add_edge` từ `planning`, hội tụ ở `fuse_rerank`) — cố ý không tái dùng `ContextRetriever.search()` (vốn tự fan-out bằng Python thread ở Phase 2), để đúng tinh thần "graph tự fan-out" của roadmap. `tool_budget_used` được tái dùng làm bộ đếm vòng lặp retrieval (AtlasState là schema cố định theo spec, không thêm field).

### SSE streaming (`agent_controller.py`)

`compiled.stream(state, stream_mode="updates")` verify trực tiếp qua generator — đúng 8 event: `router → planning → 3 node fan-out → fuse_rerank → generate → done`. Custom ReAct không có generator từng-bước tự nhiên (đề bài chỉ nói "LangGraph streaming") nên stream theo từ như pattern `stream_rag` cũ.

---

## 3. Sáu bug thật tìm thấy khi verify sống — không có cái nào bị fake-adapter test bắt được

Đây là phần quan trọng nhất của Phase 3: **mọi bug dưới đây chỉ lộ ra khi chạy thật**, khẳng định lại nguyên tắc đã rút ra ở Phase 1 (Cypher) và giờ mở rộng sang toàn bộ pipeline agent.

| # | Bug | Cách phát hiện | Sửa |
|---|---|---|---|
| 1 | `min(length(c))` sai kiểu Cypher (mục 1) | Chạy `get_callers` thật | `size(c)` |
| 2 | `GroqProvider._backoff_seconds` luôn chờ `reset_requests_seconds` (~20 phút) dù giới hạn thật đang chạm là token/phút (`reset_tokens_seconds` ~4-5s) | So sánh header 429 thật giữa 2 lần request liên tiếp | Lấy `min()` của cả hai reset, có 3 test hồi quy |
| 3 | Groq's `openai/gpt-oss-120b` tự ý cố gọi tool dù request không khai `tools` param → server trả 400 "Tool choice is none, but model called a tool" | Crash live 3/5 câu ở lần chạy đầu | Bọc `LLMGenerationError`, chèn observation yêu cầu trả lời văn bản thuần, tối đa 3 lần retry |
| 4 | **(Bug do tự sửa #3 gây ra)** `ctx["error"]` trong `custom_react.py` trùng tên với kwarg `error` mà `AgentTracer.step()` tự truyền ở `finally` → `TypeError: got multiple values for keyword argument 'error'`, crash 2/5 câu ở lần chạy 2 | Chạy lại full 5 câu sau fix #3 | Đổi tên key thành `ctx["llm_error"]` |
| 5 | `plan()` gọi `tools.search_symbol(state["query"])` với **nguyên câu hỏi** ("who calls APIRouter.add_api_route") thay vì trích riêng tên symbol — exact-match không bao giờ khớp | So sánh kết quả `search_symbol` khi gọi thủ công vs trong graph | Thêm `_extract_symbol_mention()` (regex ưu tiên dotted-identifier), 7 test |
| 6 | Evidence từ `plan()`'s structural pre-check (`impact_analysis`) không bao giờ được đưa vào `fuse_rerank` — nằm trong trace nhưng không ảnh hưởng ranking | LangGraph trả "không tìm thấy trong codebase" cho câu có `impact_analysis` tìm đúng 8 kết quả | Hydrate content qua `read_source`, đưa vào RRF làm nguồn thứ 4 |

### Bug #7 — không phải bug code, mà là giới hạn thiết kế: reranker phá structural evidence (tái hiện lần 2, độc lập với Phase 2)

Sau khi sửa #6, RRF fusion xếp đúng `fastapi.applications.FastAPI.add_api_route` ở vị trí 4/35 — nhưng cross-encoder reranker **vẫn** loại hết mọi kết quả đúng, giữ lại 5 file test có tên bề mặt giống câu hỏi (`test_original_api_route_...is_called_after_inclusion`). Verify bằng cách spy trực tiếp input/output của `reciprocal_rank_fusion()`: xác nhận dứt khoát RRF đúng, reranker sai.

Đây là **cùng hiện tượng đã ghi nhận ở Phase 2** (`docs/PHASE2_RESULTS.md` §8, câu 3 và 9), giờ tái hiện độc lập lần thứ hai trong ngữ cảnh agent — bằng chứng đủ mạnh để không còn là nghi vấn. Sửa: khi `tool_hits` (structural evidence) khác rỗng, **bỏ qua cross-encoder rerank**, dùng thẳng thứ tự RRF. Đây chính là hàng `"Hybrid + RRF, không rerank"` đã thêm vào Bảng A (`CODEATLAS_SPEC.md` §3.3) từ cuối Phase 2 — giờ có bằng chứng thực thi thật, không chỉ là dự đoán.

Verify sau fix: câu "who calls APIRouter.add_api_route" → LangGraph trả lời đúng, trích dẫn `FastAPI.add_api_route` + `APIRouter.get` thật.

### Bug #8 — nghiêm trọng nhất: Custom ReAct bịa trích dẫn thật

Câu "where is request body parsing implemented", Custom ReAct trả lời:
> "Request body parsing is implemented in the `parse_body` function inside `src/http/request_parser.py`... 【src/http/request_parser.py:45-78】"

**File và hàm này không tồn tại trong FastAPI.** Model bịa hoàn toàn, vi phạm thẳng luật CLAUDE.md "Không bịa trích dẫn". System prompt đã có câu "If no source supports a claim, say 'không tìm thấy trong codebase'" — không đủ, model vẫn bịa khi tự đánh giá là "đủ bằng chứng" mà không thực sự kiểm chứng.

**Sửa — lớp kiểm tra cơ học, không phải "sửa hallucination nói chung" (bài toán chưa ai giải xong):** sau khi có Final Answer, trích mọi pattern `[file.py:N-M]` (và cả biến thể `【...】` mà model thật sự dùng) bằng regex, đối chiếu với tập source thật đã xuất hiện trong evidence — quét đệ quy toàn bộ tool result, không phụ thuộc shape từng tool.

**Trích dẫn không hợp lệ bị GỠ KHỎI câu trả lời, không chỉ gắn cờ.** Người đọc thấy cảnh báo vẫn có xu hướng tin phần câu chữ xung quanh nó; gỡ hẳn kèm một dòng ghi rõ đã gỡ gì mới thực sự tuân thủ luật "không bịa trích dẫn". Kiểm ở hai mức:
- **file-level**: file được trích có từng xuất hiện trong evidence không
- **line-level**: dải dòng có giao với dải dòng thật sự lấy về không (đúng file, sai dòng vẫn là bịa)

Dùng chung giữa cả hai orchestrator và eval harness Phase 5 qua `codeatlas/eval/citations.py`, xuất `citation_validity_rate` — đã thêm vào spec §3.2 làm metric chính thức. 11 test đơn vị. Verify trên chính câu trả lời bịa thật ở trên: `validity_rate = 0.0`, trích dẫn bị gỡ, ghi chú rõ ràng.

**Đây là bằng chứng cụ thể cho lựa chọn kiến trúc ở Phase 3:** LangGraph's `generate` node bị ép chỉ trả lời dựa trên context đã retrieve (constrained), trong khi Custom ReAct tin vào phán đoán "đủ bằng chứng chưa" của chính model — và phán đoán đó có thể sai. Đáng đưa vào Bảng B như một cột định tính, không chỉ số liệu latency/LOC.

---

## 4. Chi phí thật của rate limit — dữ liệu cho Bảng B cột Latency

| Câu hỏi | Custom ReAct | LangGraph |
|---|---|---|
| jsonable_encoder | crash (bug #3, trước fix) | 2.1s |
| who calls add_api_route | 3.4-3.9s, 6 tool call | 1.1-13.5s |
| BackgroundTasks.add_task | 3.2-335.4s (dao động cực lớn) | 2.3-14.5s |
| request body parsing | 4-20.3s, có lần bịa trích dẫn | 2.9-61.2s |
| OpenAPI schema | crash (bug #4, trước fix) | **1171.7s** (~20 phút, chạm reset_requests thật) |

**Quan sát thật, không tinh chỉnh:** Custom ReAct có phương sai latency cực lớn (3.2s → 335.4s cho cùng một câu ở các lần chạy khác nhau) vì mỗi turn là một lệnh gọi LLM riêng, transcript dài dần theo vòng lặp → dễ chạm rate limit giữa chừng. LangGraph có số lệnh gọi LLM cố định thấp hơn (HyDE + generate, tối đa 2 vòng) nên ổn định hơn ở đa số trường hợp — nhưng vẫn có thể chạm mức chờ cực đoan (1171.7s) khi tài khoản Groq đã cạn `remaining_requests` sau phiên làm việc dài. Free tier Groq (30 RPM/1K RPD/**8000 TPM** cho model hiện tại) là ràng buộc thật cho cả hai orchestrator ở quy mô demo, chưa nói tới eval quy mô lớn ở Phase 5 (đúng lý do spec chỉ định Modal cho eval).

---

## 5. Test

149 test xanh (Phase 2 kết thúc ở 108):

| File | Số | Phủ |
|---|---|---|
| `test_harness.py` | 8 | Trần 8 call, loop detection |
| `test_tools.py` | 16 | 7 tool, fake adapter, difflib suggestion |
| `test_state.py` | 4 | AtlasState shape, reducer additive |
| `test_langgraph_qa.py` | 7 | `_extract_symbol_mention` (bug #5) |
| `test_custom_react.py` | 6 | Citation verification (bug #8) |
| `test_llm_providers.py` (mở rộng) | +3 | Backoff min-of-both-resets (bug #2) |

**Giới hạn đã biết của fake-adapter test:** không bắt được lỗi cú pháp Cypher (bug #1) — chỉ verify sống mới lộ ra. Giữ nguyên chiến lược "logic thuần → fake adapter, wiring → verify sống" từ Phase 1/2, nhưng ghi rõ ranh giới này để không hiểu nhầm 149 test xanh là bằng chứng đủ.

---

## 6. Việc còn treo

1. **Groq quota cạn thật sau phiên làm việc dài** (327/8000 token còn lại lúc kết thúc phiên) — không chạy thêm live verification hôm nay. Con số Latency ở mục 4 dùng dữ liệu đã thu thập trong phiên, không giả định thêm.
2. **Custom ReAct thiếu safeguard chống hallucination có hệ thống hơn** — lớp kiểm tra citation hiện tại là cơ học (kiểm tra file_path xuất hiện trong evidence chưa), chưa kiểm tra nội dung câu trả lời có thực sự khớp với evidence hay không. Đủ để bắt bug #8 cụ thể, chưa đủ tổng quát.
3. ~~`PostgresSaver` chưa verify chạy thật~~ → **đã verify sống.** Thêm service `postgres` vào `docker-compose.yml`, chạy `scripts/verify_postgres_checkpoint.py` hai pha ở **hai process tách biệt**:
   - Process 1: chạy graph, `interrupt_before=["step_c"]` → state dừng ở `['a','b']`, `next=('step_c',)`
   - Process 2 (process mới hoàn toàn, connection mới): đọc lại từ Postgres được đúng `['a','b']`, `next=('step_c',)`, `invoke(None)` chạy tiếp ra `['a','b','c']`

   Import được không chứng minh resume được — đây mới là bằng chứng. Phase 4's interrupt/human-approval dựa hoàn toàn vào cơ chế này.

Log đầy đủ hai lần chạy 5-câu-hỏi (trước và sau vòng sửa bug) ở `docs/phase3_verification/run1_pre_fix.txt` và `run2_post_fix.txt`. Chạy lại bằng `poetry run python scripts/verify_phase3_agents.py --repo-id fastapi` sau khi index (cần `PYTHONPATH=.` nếu chạy ngoài `poetry run python -m`).
