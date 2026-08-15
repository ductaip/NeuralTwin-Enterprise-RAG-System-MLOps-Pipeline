# CodeAtlas — Final Architecture Spec

> **Agentic Repository Intelligence.** Hiểu và sửa codebase lạ bằng GraphRAG + LangGraph agent có verification loop.
>
> Convert từ NeuralTwin. Giữ ~70% hạ tầng, thay tầng domain và tầng orchestration.

---

# 0. TL;DR — Chốt quyết định

| Hạng mục | Quyết định | Lý do một dòng |
|---|---|---|
| Domain | Repository intelligence | Graph là load-bearing, ground truth tự động, trúng nghiệp vụ FPT |
| Agent scope | QA **+ Refactor có verification** | Closed loop, test là ground truth tuyệt đối |
| Orchestration | **LangGraph 1.2** | Cần cycle + interrupt + checkpoint — đủ ba điều kiện |
| Demo serving | **Groq** (llama-3.3-70b-versatile) | Latency-bound: 8 tool call tuần tự, cần cảm giác tức thì |
| Eval serving | **Modal + vLLM** (Qwen2.5-7B-AWQ, L4) | Throughput-bound, không rate limit, deterministic |
| Local | ❌ Không dùng | Máy không đủ |
| Graph DB | Neo4j (giữ) | Call graph là graph thật |
| Vector | Qdrant (giữ) | |
| Ngôn ngữ hỗ trợ | Python (phase 1), thiết kế mở cho tree-sitter | |

---

# 1. Positioning

## 1.1 One-liner phỏng vấn

> "CodeAtlas giúp kỹ sư làm việc với codebase chưa từng đọc. Nó parse AST dựng call graph vào Neo4j, index code vào Qdrant, rồi dùng một LangGraph agent định tuyến giữa hai nguồn. Nó trả lời được câu 'đổi hàm này thì cái gì hỏng' — và ở chế độ refactor, nó tự sinh patch, chạy đúng những test bị ảnh hưởng, đọc lỗi và sửa lại cho tới khi xanh."

## 1.2 Hai chế độ

**Mode QA (read-only)** — trả lời câu hỏi về codebase, có trích dẫn `[file.py:12-30]`.

**Mode Refactor (write + verify)** — nhận yêu cầu thay đổi, thực thi có kiểm chứng:
```
impact_analysis (graph) → generate_patch → sandbox apply → run affected tests
   ├─ pass → interrupt: chờ người duyệt → commit
   └─ fail → repair (đọc stderr) → loop (tối đa 3 vòng)
```

🔑 **Điểm bán hàng cốt lõi:** graph cho phép chạy **12 test thay vì 3.000 test**. Đây là con số bạn nói ra và interviewer nhớ.

## 1.3 Phản biện & câu trả lời

**"Khác gì Claude Code / Cursor?"**
> Chúng làm việc theo ngữ cảnh cục bộ và grep. CodeAtlas dùng call graph để xác định phạm vi ảnh hưởng chính xác qua nhiều hop, và từ quan hệ TESTS chọn ra tập test tối thiểu cần chạy. Trên repo lớn, chạy toàn bộ test suite mất hàng giờ — đó là khác biệt thực tế, không phải khác biệt marketing. Ngoài ra CodeAtlas self-hosted, phù hợp khách hàng không cho code rời hạ tầng nội bộ.

**"Sao cần Neo4j, Qdrant không đủ?"**
> Hai loại truy vấn khác nhau. "Code retry logic ở đâu" là semantic → vector. "Hàm này bị ai gọi qua 2 hop" là truy vấn cấu trúc → traversal. Embedding không mã hoá được quan hệ bắc cầu. Em có ablation cho thấy vector-only sập ở nhóm câu structural và impact.

**"Sao dùng LangGraph, chẳng phải anh từng chê LangChain bloat?"**
> Em không tin vào lý lẽ suông nên viết cả hai rồi đo. Custom ReAct (~80 dòng) đủ và đúng cho mode QA — một vòng lặp, không branching, không cần framework. Nhưng mode refactor thì workflow có cycle (sửa–test–sửa lại), cần pause chờ người duyệt trước khi apply patch, và cần resume nếu crash giữa chừng. Ba thứ đó tự viết được, nhưng viết ra thì chính là viết lại checkpointer và interrupt. Đó là lúc framework đáng giá. Bảng B trong phần eval là số liệu so sánh hai cách, có cả cột LOC — kể cả khi nó bất lợi cho LangGraph.

**"Sao hai backend LLM?"**
> Chọn theo profile workload. Agent gọi 8 lượt tuần tự nên demo là latency-bound → Groq chạy 300–1.000 token/giây. Eval là throughput-bound, ~1.6M token mỗi lần chạy, vượt xa hạn mức free tier → Modal + vLLM, không rate limit và kiểm soát được seed.

**"Đo chất lượng thế nào?"**
> Ba tầng, và 35/60 câu trong gold set có ground truth sinh tự động từ AST. Mode refactor thì ground truth là test pass/fail — tuyệt đối, không cần LLM chấm.

---

# 2. Kiến trúc

## 2.1 Bản đồ chuyển đổi từ NeuralTwin

| NeuralTwin | CodeAtlas | Mức |
|---|---|---|
| `crawlers/` (social) | `ingestion/` — git + AST parser | 🔴 Mới |
| `domain/documents/` | `domain/code/` (Repository, Symbol, Edge) | 🟠 Sửa nhiều |
| Neo4j schema | Code graph schema | 🔴 Mới |
| `Neo4jAdapter` | Giữ, đổi Cypher template | 🟢 Nhẹ |
| Qdrant client | Giữ, đổi chunking | 🟢 Nhẹ |
| BM25 | Giữ, đổi tokenizer cho identifier | 🟠 Vừa |
| HyDE | Đổi prompt → sinh code snippet | 🟢 Nhẹ |
| RRF (k=60) | Giữ nguyên | 🟢 Giữ |
| Cross-encoder rerank | Giữ nguyên | 🟢 Giữ |
| ~~Custom ReAct loop~~ — thực chất là mock loop, đã xoá ở Phase 0.5 | **Viết mới** (~80 dòng) cho mode QA, làm baseline so sánh | 🔴 Mới |
| — | **LangGraph** cho mode Refactor | 🔴 Mới |
| LLM Factory | + `GroqProvider`, `ModalVLLMProvider` | 🟠 Vừa |
| **Mock Mode** | ❌ **XOÁ HOÀN TOÀN** | 🔴 Xoá |
| ZenML | Giữ, đổi step | 🟢 Nhẹ |
| Kafka | Giữ — incremental re-index on push | 🟢 Giữ |
| FastAPI + SSE | Giữ nguyên | 🟢 Giữ |
| Redis semantic cache | Giữ + thêm LLM response cache | 🟠 Vừa |
| Prometheus/Grafana/Jaeger | Giữ nguyên | 🟢 Giữ |
| Docker/K8s | Giữ nguyên | 🟢 Giữ |

## 2.2 Graph schema (Neo4j)

```cypher
// ===== NODES =====
(:Repository {name, url, commit_sha, indexed_at, language})
(:File       {path, language, loc, sha})
(:Module     {name, qualified_name})
(:Class      {name, qualified_name, docstring, start_line, end_line})
(:Function   {name, qualified_name, signature, docstring,
              start_line, end_line, is_async, is_public, complexity})
(:Test       {name, qualified_name, file_path, framework})

// ===== RELATIONSHIPS =====
(:File)     -[:IN_REPO]->        (:Repository)
(:Function) -[:DEFINED_IN]->     (:File)
(:Class)    -[:DEFINED_IN]->     (:File)
(:Function) -[:METHOD_OF]->      (:Class)
(:Function) -[:CALLS {line, confidence, reason}]-> (:Function|:Class)
(:File)     -[:IMPORTS {alias}]->(:Module)
(:Class)    -[:INHERITS]->       (:Class)
(:Test)     -[:TESTS]->          (:Function)
(:Test)     -[:COVERS {hits}]->  (:Function)   // Phase 4, xem mục "Bốn điều chỉnh"

// ===== INDEXES (bắt buộc) =====
CREATE INDEX fn_qn   IF NOT EXISTS FOR (f:Function) ON (f.qualified_name);
CREATE INDEX cls_qn  IF NOT EXISTS FOR (c:Class)    ON (c.qualified_name);
CREATE INDEX file_p  IF NOT EXISTS FOR (f:File)     ON (f.path);
```

### Bốn điều chỉnh sau Phase 1 (đã đo, không phải giả định)

**1. `CALLS` có thể trỏ vào `:Class`.** `MyClass()` được resolve về `MyClass.__init__`; khi class không định nghĩa `__init__` thì edge trỏ thẳng vào node `Class`. Bỏ edge đó thì impact analysis mù với **mọi** lời gọi khởi tạo — trên fastapi, `FastAPI.__init__` có fan-in 720, tức nếu bỏ thì mất đúng quan hệ dày nhất trong repo. Nhất quán với nguyên tắc recall-first.

**2. Ngưỡng confidence là tham số, không hardcode.** Mọi truy vấn nhận `$min_confidence`. Mặc định: `0.5` cho chọn test (thiếu edge → bỏ sót test → bug lọt production), `0.9` cho fan-in/dead-code (ở đó nhiễu mới là cái hại). Xem `settings.CALL_EDGE_MIN_CONFIDENCE_*`.

**3. `qualified_name` theo ngữ nghĩa `__qualname__`**, kể cả `<locals>`: `pkg.mod.outer.<locals>.inner`. Xấu trong Cypher nhưng bắt buộc vì đây là khoá `MERGE` — hai hàm `inner` khác nhau trùng khoá thì graph gộp âm thầm, không báo lỗi, chỉ trả lời sai.

**4. `TESTS` không đủ — cần `COVERS` bổ sung ở Phase 4.** Đo trên fastapi: **81% test node cô lập**, không có quan hệ `TESTS` nào. Nguyên nhân: test đi qua ranh giới HTTP (`client.get("/")`) thì chuỗi `CALLS` đứt tại `TestClient` (external) — static call graph không bắc được qua dynamic dispatch của một HTTP client. Test gọi thẳng hàm thì `TESTS` đúng (precision cao, đã kiểm bằng mắt); test integration thì không (recall thấp).

`COVERS {hits}` là quan hệ thứ hai, lấy từ `coverage run --context=test`, **không gộp** với `TESTS` — hai nguồn có đặc tính ngược nhau (`TESTS`: precision cao/recall thấp; `COVERS`: recall cao/precision thấp), giữ tách để Phase 5 đo riêng đóng góp từng nguồn (Bảng C thêm hàng). Xem `docs/PHASE1_RESULTS.md` §5 cho số đo đầy đủ và thiết kế chi tiết.

### Cypher lõi

```cypher
// [1] Ai gọi hàm này
MATCH (caller:Function)-[r:CALLS]->(f:Function {qualified_name:$qn})
RETURN caller.qualified_name, caller.signature, r.line, r.confidence;

// [2] Impact analysis — k hop ngược
MATCH path = (impacted:Function)-[:CALLS*1..3]->(f:Function {qualified_name:$qn})
RETURN DISTINCT impacted.qualified_name, min(length(path)) AS distance
ORDER BY distance;

// [3] Tập test tối thiểu cần chạy  ← TRUY VẤN QUAN TRỌNG NHẤT
// Phase 1: chỉ TESTS. Phase 4 trở đi: hợp nhất với COVERS (xem điều chỉnh 4 ở trên).
MATCH (t:Test)-[:TESTS]->(impacted:Function)-[:CALLS*0..3]->(f:Function {qualified_name:$qn})
RETURN DISTINCT t.qualified_name, t.file_path;

// [3'] Phase 4 — bản mở rộng với COVERS
MATCH (t:Test)-[r:TESTS|COVERS]->(impacted)-[:CALLS*0..3]->(f:Function {qualified_name:$qn})
WHERE type(r) = 'TESTS' OR r.hits >= $min_hits
RETURN DISTINCT t.qualified_name, t.file_path, type(r) AS source;

// [4] Fan-in cao — điểm nghẽn kiến trúc
MATCH (f:Function)<-[:CALLS]-(caller)
WITH f, count(DISTINCT caller) AS fan_in WHERE fan_in > 10
RETURN f.qualified_name, fan_in ORDER BY fan_in DESC LIMIT 20;

// [5] Dead code — không ai gọi, không phải entrypoint/test
MATCH (f:Function) WHERE NOT (f)<-[:CALLS]-() AND f.is_public = false
RETURN f.qualified_name, f.start_line;
```

## 2.3 LangGraph design

```python
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import interrupt, Command

class AtlasState(TypedDict):
    # Input
    query: str
    mode: Literal["qa", "refactor"]
    repo_id: str
    # Working
    plan: str
    evidence: Annotated[list[dict], add]      # reducer: gộp thay vì ghi đè
    tool_budget_used: int
    # Refactor branch
    impacted_symbols: list[str]
    affected_tests: list[str]
    patch: str | None
    test_output: str | None
    repair_iteration: int
    # Output
    answer: str
    citations: list[dict]
```

### Đồ thị

```
                    START
                      │
                  [router]
              ┌───────┴────────┐
         mode=qa          mode=refactor
              │                │
        ┌─────▼──────┐   ┌─────▼──────────┐
        │   plan     │   │ impact_analysis│  ← Cypher [2]+[3]
        └─────┬──────┘   └─────┬──────────┘
              │                │
        ┌─────▼──────┐   ┌─────▼──────────┐
        │  retrieve  │   │ generate_patch │
        │ (parallel: │   └─────┬──────────┘
        │  graph /   │         │
        │  vector /  │   ┌─────▼──────────┐
        │  bm25)     │   │ sandbox_apply  │
        └─────┬──────┘   └─────┬──────────┘
              │                │
        ┌─────▼──────┐   ┌─────▼──────────┐
        │ fuse+rerank│   │  run_tests     │  ← chỉ chạy affected_tests
        └─────┬──────┘   └─────┬──────────┘
              │                │
       [enough evidence?] [tests pass?]
          │        │         │       │
         no       yes       no      yes
          │        │         │       │
      back to   ┌──▼─────┐ ┌─▼────┐ ┌▼──────────────┐
       plan     │generate│ │repair│ │ interrupt():   │
                └──┬─────┘ └─┬────┘ │ human approve  │
                   │         │      └───┬────────────┘
                   │    (max 3, rồi     │
                   │     báo thất bại)  │
                   │         │      ┌───▼────┐
                   │         └──────│ commit │
                   │                └───┬────┘
                   └────────┬───────────┘
                           END
```

### Bốn tính năng LangGraph được dùng — và vì sao cần

| Tính năng | Dùng ở đâu | Nếu tự viết thì sao |
|---|---|---|
| **Conditional edge** | router, `enough_evidence?`, `tests_pass?` | Viết được, nhưng rối |
| **Cycle** | repair loop, plan↔retrieve | Viết được |
| **Checkpointer** (PostgresSaver) | Resume khi crash giữa refactor | **Phải tự viết state serialization** |
| **`interrupt()`** | Pause chờ người duyệt patch | **Phải tự viết middleware pause/resume** |

Hai hàng cuối là lý do chính đáng để dùng framework. Nói đúng như vậy khi phỏng vấn.

## 2.4 Agent tools (mode QA)

```python
search_semantic(query, top_k)         # Qdrant + BM25 + RRF + rerank
search_symbol(name)                   # tra chính xác, có fuzzy fallback
get_callers(qualified_name, depth)    # Cypher [1]
get_callees(qualified_name, depth)
impact_analysis(qualified_name)       # Cypher [2] + [3]
read_source(file_path, start, end)
list_module_structure(module)
```

**Nguyên tắc harness:**
- Trần 8 tool call/query, vượt thì trả lời bằng bằng chứng đã có
- Loop detection: hash `(tool_name, args)`, lặp lần 2 → chèn observation cảnh báo
- Mọi tool trả JSON có field `source` (file + line range)
- Error message actionable: `"Symbol 'authX' not found. Closest: 'authenticate' (difflib 0.82)"`
- Trace mỗi bước ra JSONL + Prometheus

## 2.5 Chunking & retrieval

**Chunk = 1 function/method** (kèm docstring). >100 dòng thì chia theo block, giữ header.

Metadata: `qualified_name`, `file_path`, `start_line`, `end_line`, `language`, `parent_class`, `repo_id`.

**Contextual retrieval:** trước khi embed, prepend một câu ngữ cảnh do LLM sinh:
```
Ngữ cảnh: Hàm này thuộc module xác thực, được gọi bởi middleware FastAPI,
trả về JWT payload đã decode.
<code>
```
Cache theo hash chunk để không gọi lại. Chạy trên Modal (batch).

**BM25 tokenizer:** phải tách `getUserById` → `get, user, by, id` **đồng thời giữ token gốc**. Đây là điểm mấu chốt để tìm symbol chính xác.

**HyDE:** sinh một đoạn code Python giả định khớp câu hỏi, rồi embed đoạn code đó (không embed văn xuôi).

## 2.6 LLM serving

| Workload | Backend | Model | Ghi chú |
|---|---|---|---|
| Live demo | Groq | `llama-3.3-70b-versatile` | 30 RPM / 1K RPD / 12K TPM / 100K TPD |
| Eval + ablation | Modal + vLLM | Qwen2.5-7B-Instruct-AWQ trên L4 24GB | Không rate limit, seed cố định |
| Contextual enrichment | Modal + vLLM | Qwen2.5-7B-AWQ | Batch, ~1.2M token |
| Fallback nhanh/rẻ | Groq | `llama-3.1-8b-instant` | RPD cao nhất free tier |

**Bắt buộc:**
- Đọc header rate limit lúc runtime: `x-ratelimit-limit-requests` là RPD, `x-ratelimit-limit-tokens` là TPM. Backoff theo header, không hardcode.
- Prompt caching: token đã cache không tính vào rate limit → giữ system prompt cố định.
- Disk cache theo `(provider, model, prompt_hash)` — chạy lại ablation gần như miễn phí.
- `temperature=0` + seed cố định cho eval; vLLM continuous batching vẫn dao động nhẹ → **median of 3 runs** (đúng giao thức đã dùng ở Viettel Challenge).

---

# 3. Evaluation

> Đây là phần tạo khác biệt. Nếu thiếu thời gian, cắt phần khác chứ đừng cắt phần này.

## 3.1 Gold set — 60 câu QA + 20 task refactor

| Loại | Số | Ground truth | Ví dụ |
|---|---|---|---|
| Structural | 20 | ✅ Tự động (Cypher) | "Hàm X được gọi từ đâu?" |
| Impact | 15 | ✅ Tự động (Cypher) | "Đổi X thì test nào phải chạy?" |
| Semantic | 15 | Annotate tay | "Code xử lý rate limiting ở đâu?" |
| Multi-hop | 10 | Annotate tay | "Giải thích luồng request → DB" |
| **Refactor task** | 20 | ✅ **Tuyệt đối (test pass/fail)** | "Đổi tên X thành Y", "Thêm param có default" |

🔑 **55/80 mục có ground truth khách quan.** Nói con số này khi phỏng vấn.

## 3.2 Metrics

```
Retrieval:   Precision@5, Recall@10, MRR, nDCG@10
Impact:      Exact set match, Jaccard(pred, gold)
QA E2E:      Faithfulness, Answer correctness (LLM-as-judge, swap order
             khử position bias, đo Cohen's κ với human trên 20 mẫu)
Refactor:    Success rate (test xanh), số vòng repair trung bình,
             test selection precision (chạy đúng test cần chạy),
             % test tiết kiệm được so với chạy full suite
System:      p50/p95 latency, tokens/query, index time /1k LOC
```

## 3.3 Ablation — "money table"

### Bảng A — Retrieval

| Cấu hình | Structural | Semantic | Impact | Overall |
|---|---|---|---|---|
| Vector-only | | | | |
| BM25-only | | | | |
| Graph-only | | | | |
| Hybrid + RRF | | | | |
| + Cross-encoder rerank | | | | |
| + Contextual retrieval | | | | |
| **+ Agent routing** | | | | |

**Giả thuyết:** vector-only sập ở Structural/Impact; graph-only sập ở Semantic; chỉ agent routing tốt đều cả ba.

### Bảng B — Orchestration (đây là bảng biện minh cho LangGraph)

| Cấu hình | QA acc | Refactor success | Latency p95 | LOC |
|---|---|---|---|---|
| Custom ReAct loop (viết mới ở Phase 3) | | ❌ không hỗ trợ | | |
| LangGraph (QA mode) | | ❌ | | |
| LangGraph (Refactor + verify) | | | | |

Cột LOC (số dòng orchestration code) cho thấy chi phí thật của framework. Trung thực cả khi nó bất lợi.

⚠️ **Đính chính sau Phase 0.5:** loop ReAct trong NeuralTwin là mock (`_mock_solve_loop`, tool trả string hardcode) nên đã bị xoá — **không còn baseline kế thừa**. Cả hai nhánh trong bảng này đều được viết mới ở Phase 3, cùng một bộ tool và cùng một retrieval layer, để so sánh chỉ khác nhau ở tầng orchestration. Điều này làm bảng B **công bằng hơn** so với kế hoạch cũ (so code cũ với code mới là so sai), nhưng phải nói rõ khi trình bày: đây là so sánh hai implementation viết cùng lúc, không phải hành trình tiến hoá theo thời gian.

### Bảng C — Test selection

| | Full suite | Graph-selected (TESTS only) | Graph-selected (COVERS only) | Graph-selected (Union) |
|---|---|---|---|---|
| Số test chạy | | | | |
| Thời gian | | | | |
| Bug bắt được | | | | |

Cột cuối quan trọng: nếu graph-selected bỏ sót bug thì phải nói ra. **Kết quả âm cũng là kết quả.**

Ba cột graph-selected tách theo nguồn quan hệ, không gộp làm một — đo ở Phase 1 trên fastapi cho thấy `TESTS` (suy từ AST) chỉ phủ **19.2%** test node (81% cô lập vì test đi qua ranh giới HTTP), trong khi `COVERS` (từ coverage run) recall cao hơn hẳn nhưng chưa biết cái giá về số test chạy — đo bảng median/p95/%suite cho cả ba nguồn **trước khi** công bố con số "N thay vì M test" ra bên ngoài. Chi tiết thiết kế và số đo ở `docs/PHASE1_RESULTS.md` §5.

## 3.4 Ngân sách token

| Hạng mục | Token ước tính | Backend |
|---|---|---|
| Contextual enrichment (1.500 chunk) | ~1.2M | Modal |
| QA eval: 60 câu × 7 config × 3 lần (non-agent ~1 call) | ~0.4M | Modal |
| QA eval: agent config (8 call/câu) | ~1.6M | Modal |
| Refactor eval: 20 task × 3 vòng × 3 lần | ~1.5M | Modal |
| Demo | <50K | Groq |
| **Tổng** | **~4.7M** | |

L4 trên Modal ≈ $0.80/giờ, $30 credit ≈ 37 giờ. Với Qwen2.5-7B-AWQ, ~4.7M token nằm gọn trong vài giờ chạy. **Có cache thì chạy lại gần như miễn phí.**

---

# 4. Talking points

## 4.1 Kể project (2–3 phút)

1. **Bài toán** — "Onboard vào codebase 500k dòng mất hàng tuần. Câu khó nhất không phải 'code này làm gì' mà 'sửa chỗ này thì cái gì hỏng'."
2. **Vì sao khó** — "Không phải bài toán similarity. Vector search trả về code *giống* câu hỏi, không trả về code *phụ thuộc* vào nó. Quan hệ bắc cầu không mã hoá được trong embedding."
3. **Cách tiếp cận** — graph cho cấu trúc, vector cho ngữ nghĩa, agent định tuyến. Mode refactor đóng vòng lặp bằng test.
4. **Chỗ thất bại** — symbol resolution. Ban đầu resolve theo tên đơn, sai nhiều với import alias và `self.method()`. Phải viết lại theo scope chain và chấp nhận đánh dấu `unresolved` thay vì đoán. Độ chính xác quan trọng hơn độ phủ vì downstream là impact analysis — sai một edge là bỏ sót một test.
5. **Kết quả** — ba bảng ablation, số cụ thể.
6. **Nếu làm lại** — dùng tree-sitter từ đầu thay vì `ast` để đa ngôn ngữ ngay; và build gold set trước khi tối ưu, vì em đã tối ưu vài vòng theo cảm tính trước khi có baseline.

## 4.2 Bốn con số phải thuộc

- Precision@5 của cấu hình tốt nhất
- Chênh lệch vector-only vs hybrid ở nhóm Impact
- Refactor success rate + số vòng repair trung bình
- Số test chạy: graph-selected vs full suite

Không nhớ số → mất hết độ tin cậy.

## 4.3 Đừng làm

- ❌ Liệt kê công nghệ như đọc CV
- ❌ Nói project không có điểm yếu
- ❌ Demo mà không test trước trên máy sạch
- ❌ Để sót Mock Mode ở bất kỳ đâu

## 4.4 Nếu bị hỏi "em làm bao lâu"

Trả lời thật: kế thừa kiến trúc từ một project trước (NeuralTwin), phần mới là ingestion, LangGraph orchestration và eval harness. Trung thực về kế thừa mạnh hơn nhiều so với bị phát hiện.

---

# 5. Checklist trước khi đem đi phỏng vấn

- [ ] `docker compose up` chạy được trên máy sạch
- [ ] Demo hoàn chỉnh dưới 90 giây, đã bấm giờ thử
- [ ] **Quay sẵn video demo dự phòng** (wifi phòng phỏng vấn hỏng là chuyện thường)
- [ ] Ba bảng ablation có số thật, không placeholder
- [ ] README có GIF demo ở ngay đầu
- [ ] Thuộc 4 con số ở mục 4.2
- [ ] Trả lời trơn 5 phản biện ở mục 1.3
- [ ] Có một điểm thất bại thật để kể
- [ ] `grep -ri "mock" .` ra rỗng ở phần agent
- [ ] Repo public, commit history sạch
- [ ] `docs/DESIGN_DECISIONS.md` viết theo format Context/Options/Decision/Consequences