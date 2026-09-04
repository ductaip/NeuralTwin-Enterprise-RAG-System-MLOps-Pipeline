# Design Decisions

> Ghi dần mỗi phase, Phase 6 biên tập lại. Format: Context → Options → Decision → Consequences (trung thực cả bất lợi).

---

## 1. LangGraph 1.2 → 0.4.5

**Phase:** 3  
**Ngày:** 2026-08-18

### Context

Spec §2.3 chỉ định LangGraph 1.2 cho orchestration. `langgraph>=0.5` (kể cả 1.2) kéo theo `langgraph-sdk`, đòi `orjson>=3.11.5`. ZenML 0.74.0 (luật cứng — giữ nguyên) ghim `orjson>=3.10,<3.11`. Hai ràng buộc không thể cùng thoả — `poetry add langgraph` với bất kỳ version `>=0.5` đều fail resolve.

### Options considered

1. **Nâng ZenML** → phá ràng buộc cứng, rủi ro regression trên toàn bộ pipeline đã chạy
2. **Patch orjson constraint** bằng `poetry add orjson --allow-prereleases` → hack, không tái lập được
3. **Dùng langgraph 0.4.5** — bản mới nhất cài được cùng ZenML

### Decision

Option 3. Verify import thật: `StateGraph`, `START`/`END`, `add_conditional_edges`, `interrupt`/`Command` (`langgraph.types`), `SqliteSaver`, `PostgresSaver` — đều có ở 0.4.5, không thiếu API nào Phase 3/4 cần.

### Consequences

**Thuận lợi:**
- Cài được, chạy được, không hack dependency
- API đủ dùng cho cả QA graph và refactor graph (đã verify)
- `StateGraph.add_node` ở 0.4.5 từ chối tên node trùng tên field trong state — sửa đổi nhỏ (node `"plan"` → `"planning"`)

**Bất lợi:**
- Không có API mới nếu 1.2 thêm gì sau này
- Phải ghi rõ version constraint trong README — người clone repo nếu tự nâng sẽ gặp lỗi không rõ nguyên nhân
- Bị khoá ở orjson <3.11 — ảnh hưởng nếu package khác cần orjson mới hơn

---

## 2. Model Groq: llama-3.3-70b-versatile → openai/gpt-oss-120b

**Phase:** 3  
**Ngày:** 2026-08-18

### Context

Spec §2.6 chỉ định `llama-3.3-70b-versatile` cho live demo (latency-bound). Gọi thật ngày 2026-08-18 → 404. Groq đã đổi catalog, không còn dòng Llama chat nào. Không có thông báo trước.

### Options considered

1. **Đoán tên model mới** → rủi ro đoán sai, mất thời gian debug
2. **Gọi `client.models.list()` thật** rồi chọn model phù hợp → chậm hơn nhưng chắc chắn
3. **Chuyển sang OpenAI/Anthropic** → mất ưu thế latency của Groq

### Decision

Option 2. Chạy `models.list()`, chọn `openai/gpt-oss-120b` — có sẵn, gọi thành công.

### Consequences

**Thuận lợi:**
- Chạy được thật, verify sống trên 5 câu
- Quy trình "tra danh sách thật, không đoán" áp dụng được cho mọi lần sau

**Bất lợi:**
- `openai/gpt-oss-120b` là **reasoning model**: tốn token vào field `reasoning` ẩn trước `content`. `max_tokens` thấp (vd 10) bị cắt trước khi có nội dung
- TPM free-tier chỉ **8000** (thấp hơn nhiều so với 12K TPM cũ) → latency demo tệ hơn, dễ chạm rate limit hơn
- Catalog Groq đổi không báo trước → phải gọi lại `models.list()` mỗi phase, không tin tên model trong spec/roadmap

---

## 3. Modal: L4 self-host vs serverless per-token

**Phase:** 3 (đo), chưa chốt (chờ verify seed)  
**Ngày:** 2026-09-03

### Context

Spec §2.6 chỉ định Modal + vLLM (Qwen2.5-7B-Instruct-AWQ trên L4) cho eval. 4 tài khoản × $1 = ~$4 tổng. Đo thật: L4 $0.80/giờ + CPU/RAM $0.32–0.63/giờ → **$1.12–1.43/giờ thực tế**, $1 mua 40–55 phút (không phải 75 phút).

Modal cũng cung cấp serverless LLM tính theo token: GLM-5.3-Flash $0.45/MTok prompt, $1.50/MTok completion, $0.09/MTok cached. Với 1.96M token (quy mô đã cắt) ≈ **$1.50 cho toàn bộ eval**.

### Options considered

1. **L4 self-host** (đúng spec): kiểm soát model/seed, nhưng $1/tài khoản ≈ 40–55 phút, cold start 10–15 phút mỗi lần, rủi ro quên tắt
2. **Serverless per-token** (GLM-5.3-Flash): rẻ hơn 2–3x, không cold start, không quên tắt, nhưng model khác spec và chưa verify seed

### Decision

Chưa chốt — điều kiện chặn: gọi thử endpoint với `seed` + `temperature=0`, cùng prompt 3 lần. Nếu output giống hệt → serverless. Nếu không → L4 self-host.

### Consequences

**Nếu chọn serverless:**
- Xoá sạch bài toán ngân sách — ~$1.50 thay vì chia 4 tài khoản thủ công
- Mất kiểm soát model (không có Qwen2.5-7B-AWQ) — phải verify chất lượng GLM-5.3-Flash trên 10 câu Phase 2
- Chạy lại rẻ gần bằng 0 (cached $0.09/MTok) nhưng cache là của Modal

**Nếu chọn L4 self-host:**
- Đúng spec, kiểm soát seed, reproducibility đảm bảo
- Ngân sách cực chặt — một lần deploy hỏng = mất 10–15 phút trong 40–55 phút

---

## 4. Recall > Precision: ngưỡng confidence ở tầng đọc, không phải tầng ghi

**Phase:** 1  
**Ngày:** 2026-08-12

### Context

Symbol resolver gán `confidence` cho mỗi call edge. Câu hỏi: lọc ngay lúc ghi vào graph (chỉ ghi edge confidence ≥ X), hay ghi hết rồi lọc lúc đọc?

### Options considered

1. **Lọc lúc ghi** (confidence ≥ 0.9): graph sạch, Cypher đơn giản hơn
2. **Ghi hết, lọc lúc đọc** qua `$min_confidence` parameter: graph lớn hơn nhưng linh hoạt

### Decision

Option 2. Mọi Cypher nhận `$min_confidence`. Mặc định: 0.5 cho chọn test (recall-first), 0.9 cho fan-in/dead-code (precision-first). Xem `settings.CALL_EDGE_MIN_CONFIDENCE_*`.

### Consequences

**Thuận lợi:**
- Đo trên fastapi: median test/function không đổi ở mọi ngưỡng (2, 2, 2, 2 cho 0.5/0.7/0.9/1.0) — recall-first gần như miễn phí ở trung vị
- Phase 5 quét ngưỡng như một trục ablation mà không cần re-ingest
- Thiếu edge (bỏ sót test → bug lọt production) nghiêm trọng hơn nhiều so với thừa edge (chạy dư vài test)

**Bất lợi:**
- Graph lớn hơn — 16528 relationships thay vì ~12K nếu lọc ở 0.9
- Cypher dài hơn chút (`WHERE ... r.confidence >= $min_confidence`)

---

## 5. `<locals>` trong qualified_name (tránh gộp node âm thầm)

**Phase:** 1  
**Ngày:** 2026-08-12

### Context

`qualified_name` là khoá `MERGE` trong Neo4j. Hai hàm cùng tên `inner` nhưng khác scope: `pkg.mod.outer.<locals>.inner` vs `pkg.mod.other.<locals>.inner`. Bỏ `<locals>` thì trùng khoá → graph gộp âm thầm, không báo lỗi, chỉ trả lời sai.

### Options considered

1. **Bỏ `<locals>`** cho qualified_name đẹp → nguy cơ gộp node
2. **Giữ `<locals>`** theo ngữ nghĩa `__qualname__` chuẩn Python → xấu trong Cypher nhưng đúng

### Decision

Option 2. Xấu nhưng đúng. Semantic correctness thắng aesthetics vì graph là load-bearing — mọi impact analysis, mọi test selection đi qua nó.

### Consequences

**Thuận lợi:** Không bao giờ gộp nhầm hai hàm khác nhau  
**Bất lợi:** `qualified_name` chứa `<` và `>`, cần escape trong Cypher literal, khó đọc trong log

---

## 6. `CALLS` có thể trỏ vào `:Class` (không chỉ `:Function`)

**Phase:** 1  
**Ngày:** 2026-08-12

### Context

`MyClass()` được resolve về `MyClass.__init__`. Khi class không định nghĩa `__init__` thì không có node `Function` để trỏ vào. Bỏ edge đó hay trỏ vào node `Class`?

### Options considered

1. **Bỏ edge** khi không tìm thấy `__init__` → mất thông tin về lời gọi khởi tạo
2. **Trỏ vào `:Class`** → schema `CALLS` linh hoạt hơn (`(:Function) -[:CALLS]-> (:Function|:Class)`)

### Decision

Option 2. Trên fastapi, `FastAPI.__init__` có fan-in 720 — nếu bỏ thì mất đúng quan hệ dày nhất trong repo. Nhất quán với nguyên tắc recall-first.

### Consequences

**Thuận lợi:** Impact analysis đầy đủ hơn — bắt mọi lời gọi khởi tạo  
**Bất lợi:** Cypher pattern phải match `(:Function|:Class)` — xem xét khi viết truy vấn mới

---

## 7. Bỏ cross-encoder rerank khi có structural evidence

**Phase:** 3 (quyết định), 2 (phát hiện đầu tiên)  
**Ngày:** 2026-09-03

### Context

Cross-encoder (`ms-marco-MiniLM`) được train trên cặp query–passage văn xuôi. Với câu hỏi structural ("ai gọi hàm X"), nó cần tín hiệu quan hệ đồ thị — không có trong bề mặt văn bản. Hai lần phát hiện độc lập:

1. **Phase 2** (§8): `serialize_response` đúng ở dense #4 nhưng bị rerank đẩy khỏi top-5; câu "who calls add_api_route" graph tìm đúng `get/post/put/head/options` (confidence 1.0), rerank thay bằng test file
2. **Phase 3** (§3, bug #7): spy trực tiếp input/output `reciprocal_rank_fusion()` — RRF xếp đúng `FastAPI.add_api_route` ở hạng 4/35, reranker loại sạch, giữ lại 5 file test trùng tên bề mặt

### Options considered

1. **Giữ rerank cho mọi câu** → structural evidence bị loại, câu trả lời sai
2. **Bỏ rerank khi có structural evidence** (tool_hits khác rỗng) → dùng thứ tự RRF
3. **Fine-tune reranker cho code** → tốn thời gian/dữ liệu, ngoài scope

### Decision

Option 2. Khi `tool_hits` (structural evidence từ `impact_analysis`) khác rỗng, bỏ qua cross-encoder rerank, dùng thẳng thứ tự RRF. Với câu semantic (không có structural evidence), vẫn rerank bình thường.

### Consequences

**Thuận lợi:**
- Sửa đúng vấn đề: verify sau fix, câu "who calls APIRouter.add_api_route" → LangGraph trả lời đúng
- Chỉ nhìn cột Overall thì hai hiệu ứng (rerank giúp semantic, hại structural) triệt tiêu nhau — tách theo loại câu mới thấy finding. Phase 5 Bảng A có hàng đo riêng hai cấu hình

**Bất lợi:**
- Heuristic: "có tool_hits → bỏ rerank" là điều kiện đơn giản — có thể có trường hợp semantic câu hỏi nhưng tool_hits vẫn xuất hiện, lúc đó rerank đúng ra là cần
- Phase 5 phải đo xem quyết định này đúng ở mức nào trên toàn bộ gold set, không chỉ 2 câu quan sát
