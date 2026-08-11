# CodeAtlas — Roadmap & Claude Code Prompts

> Đọc kèm `CODEATLAS_SPEC.md`. File này là thứ tự thi công + prompt copy thẳng vào Claude Code.
>
> **Cách dùng:** mỗi phase là một session Claude Code riêng. Đưa cả `CODEATLAS_SPEC.md` vào context ở mọi phase.

---

# Bảng tổng — Model cho từng phase

| Phase | Nội dung | Model | Vì sao model đó | Ước lượng |
|---|---|---|---|---|
| **0** | Audit & dọn dẹp | **Sonnet** | Việc cơ học: đọc, xoá, đổi tên. Không có quyết định kiến trúc | 1 session |
| **1** | AST ingester + graph schema | **OPUS** ⭐ | Symbol resolution là phần khó nhất cả dự án. Sai ở đây thì impact analysis sai, eval sai, mọi thứ sai | 2–3 session |
| **2** | Retrieval layer | **Sonnet** | Sửa code có sẵn theo spec rõ ràng | 1–2 session |
| **3** | LangGraph — mode QA | **Sonnet** | LangGraph có API rõ, spec đã có sẵn state schema và đồ thị | 1–2 session |
| **4** | LangGraph — mode Refactor | **OPUS** ⭐ | Sandbox an toàn + cycle + interrupt + repair loop. Nhiều edge case, dễ sai âm thầm | 2–3 session |
| **5** | Eval harness + ablation | **OPUS** ⭐ | Thiết kế eval quyết định độ tin cậy của toàn bộ project. Sai eval = số liệu vô nghĩa | 2 session |
| **6** | Demo, docs, CI | **Sonnet** | Viết lách và cấu hình | 1 session |

**Quy tắc chung:** mặc định Sonnet. Escalate lên Opus khi (a) phase được đánh dấu ⭐, hoặc (b) Sonnet loay hoay quá 2 lượt ở cùng một chỗ.

**Ba phase Opus (1, 4, 5) là ba chỗ mà sai lầm không lộ ra ngay** — code vẫn chạy, chỉ là kết quả sai. Đó là tiêu chí chọn.

---

# Thứ tự ưu tiên nếu thiếu thời gian

```
BẮT BUỘC:  0 → 1 → 2 → 3 → 5 → 6
NÂNG CAO:  + 4 (mode Refactor)
```

Phase 4 là thứ tạo ấn tượng mạnh nhất, nhưng Phase 5 (eval) mới là thứ tạo **độ tin cậy**. Nếu chỉ làm được một trong hai, chọn 5.

---

# PHASE 0 — Audit & dọn dẹp
### 🤖 Sonnet

```
Đây là repo NeuralTwin — agentic GraphRAG system. Tôi đang chuyển domain
sang "repository intelligence" (tên mới: CodeAtlas). Đọc file
CODEATLAS_SPEC.md tôi đính kèm để hiểu đích đến.

Kiến trúc hạ tầng giữ nguyên. Chỉ thay tầng domain.

NHIỆM VỤ PHASE 0 — không viết logic mới:

1. Đọc toàn bộ repo. Xuất ra `docs/MIGRATION_AUDIT.md` một bảng:
   path | vai trò hiện tại | KEEP / MODIFY / DELETE | ghi chú
   Phân loại rõ: cái nào là domain-specific (gắn digital twin),
   cái nào là infrastructure dùng lại được.

2. XOÁ:
   - Toàn bộ crawler mạng xã hội (Medium, LinkedIn, GitHub profile)
   - Test và config liên quan
   - MỌI dấu vết của Mock Mode trong ResearchAgent — kể cả flag,
     env var, và nhánh code fallback. Sau phase này, lệnh
     `grep -ri "mock" llm_engineering/application/agents/` phải ra rỗng.

3. ĐỔI TÊN NeuralTwin -> CodeAtlas trong: pyproject.toml, README.md,
   docker-compose.yml, k8s/*, Makefile, và mọi import path.
   Package name: `codeatlas`.

4. GIỮ NGUYÊN, không đụng vào: Qdrant client, Neo4jAdapter, RRF,
   cross-encoder reranker, custom ReAct loop, ZenML, Kafka, FastAPI,
   Redis, Prometheus/Grafana/Jaeger, Docker/K8s manifests.

5. Chạy `pytest` sau khi xong. Báo cáo test nào fail và nguyên nhân.
   ĐỪNG tự sửa test fail ở phase này — chỉ báo cáo.

Bắt đầu bằng việc đưa tôi bảng audit ở mục 1 trước khi xoá bất cứ thứ gì.
```

**✅ Xong phase khi:** có `MIGRATION_AUDIT.md`, `grep -ri mock` sạch, repo import được, biết chính xác test nào đang fail.

---

# PHASE 1 — AST Ingester + Graph Schema
### 🧠 OPUS — phase quan trọng nhất

```
Phase 1: xây tầng ingestion cho CodeAtlas.

MỤC TIÊU: từ git repo URL -> code graph trong Neo4j + code chunks
trong Qdrant.

=== YÊU CẦU ===

Module `codeatlas/ingestion/`:

1. `repo_loader.py`
   - Clone repo (hoặc nhận local path), checkout commit cụ thể
   - Liệt kê file .py, tôn trọng .gitignore
   - Bỏ qua: .git, node_modules, venv, .venv, build, dist, __pycache__
   - Trả về danh sách (path, content, sha)

2. `python_parser.py`
   - Dùng module `ast` chuẩn thư viện (KHÔNG dùng thư viện ngoài ở phase này)
   - Trích xuất: Module, Class, Function, imports, call sites,
     inheritance, decorator, docstring, line range, is_async
   - Nhận diện test function: tên bắt đầu `test_`, hoặc trong file
     `test_*.py` / `*_test.py`, hoặc có decorator pytest

3. `symbol_resolver.py`  ← PHẦN KHÓ NHẤT, ĐỌC KỸ
   Resolve tên gọi -> qualified_name. Phải xử lý:
   - `import x.y as z` rồi gọi `z.foo()`
   - `from a.b import c` rồi gọi `c()`
   - `from . import x` (relative import), `from ..pkg import y`
   - `self.method()` -> resolve trong class hierarchy, kể cả kế thừa
   - `super().method()`
   - Nested function, nested class
   - Method gọi qua instance biến local (best-effort)
   - Shadowing: biến local trùng tên module

   RÀNG BUỘC TUYỆT ĐỐI: chỗ nào không resolve chắc chắn thì gán
   `confidence` thấp và đánh dấu `unresolved=true`. TUYỆT ĐỐI KHÔNG
   ĐOÁN BỪA. Downstream là impact analysis — một edge sai là một test
   bị bỏ sót, và đó là lỗi nghiêm trọng hơn nhiều so với thiếu edge.

   Ghi ra `unresolved_report.json` để tôi xem tỉ lệ.

4. `graph_builder.py`
   - Batch upsert Neo4j bằng UNWIND, KHÔNG loop từng node
   - Idempotent: MERGE theo key (repo_id, qualified_name)
   - Tạo index như trong spec
   - Chạy lại trên cùng commit không tạo node trùng

5. `chunker.py`
   - 1 chunk = 1 function/method, kèm docstring
   - >100 dòng thì chia theo block logic, giữ header
   - Metadata: qualified_name, file_path, start_line, end_line,
     language, parent_class, repo_id

6. CLI: `python -m codeatlas.ingest --repo <url|path> --lang python`
   In thống kê: #file, #function, #class, #test, #call_edge,
   #unresolved (và %), thời gian.

=== GRAPH SCHEMA ===
Dùng CHÍNH XÁC schema trong CODEATLAS_SPEC.md mục 2.2. Không tự đổi.

=== RÀNG BUỘC KỸ THUẬT ===
- Type hints đầy đủ, pydantic v2 cho data model
- Thiết kế `Parser` protocol để sau này thêm tree-sitter cho ngôn ngữ
  khác chỉ cần implement protocol, không sửa graph_builder
- Unit test cho symbol_resolver: ít nhất 15 case khó, cover hết
  danh sách ở mục 3

=== QUY TRÌNH ===
BƯỚC 1: Trình bày thiết kế của symbol_resolver — cấu trúc scope chain,
cách xử lý từng case ở mục 3, cách tính confidence. CHỜ TÔI DUYỆT.
BƯỚC 2: Sau khi duyệt mới implement.
BƯỚC 3: Test trên repo `fastapi`. Báo cáo: tỉ lệ call edge resolve
được (%), và 10 ví dụ unresolved điển hình để tôi đánh giá.
```

**✅ Xong phase khi:** index được `fastapi`, tỉ lệ resolve ≥ 80%, chạy lại không nhân đôi node, 15 unit test xanh.

⚠️ **Đừng sang phase 2 nếu tỉ lệ resolve < 70%.** Mọi số liệu sau đó sẽ vô nghĩa.

---

# PHASE 2 — Retrieval Layer
### 🤖 Sonnet

```
Phase 2: chuyển tầng retrieval sang domain code.

1. HyDE prompt: thay vì sinh câu trả lời văn xuôi giả định, sinh một
   đoạn code Python giả định khớp câu hỏi, rồi embed đoạn code đó.
   Giữ nguyên phần còn lại của pipeline.

2. BM25 tokenizer: tách identifier camelCase và snake_case thành
   sub-token (getUserById -> get, user, by, id) ĐỒNG THỜI giữ token
   gốc. Viết test cho ít nhất 10 identifier kiểu khác nhau.

3. Contextual retrieval: trước khi embed mỗi chunk, gọi LLM sinh một
   câu ngữ cảnh (thuộc module nào, vai trò gì, ai gọi nó — lấy từ
   graph) và prepend vào chunk trước khi embed.
   - Cache theo hash(chunk_content) ra đĩa, không gọi lại
   - Chạy batch qua ModalVLLMProvider (xem mục 4 bên dưới)

4. `GraphRetriever`: implement CÙNG interface với các retriever hiện có,
   để RRF fuse được kết quả graph cùng dense và sparse.

5. LLM Provider Factory — thêm 2 adapter:
   - `GroqProvider`: OpenAI-compatible endpoint. BẮT BUỘC đọc header
     rate limit lúc runtime (x-ratelimit-limit-requests = RPD,
     x-ratelimit-limit-tokens = TPM) và backoff theo header, không
     hardcode con số. Retry với exponential backoff + jitter.
   - `ModalVLLMProvider`: kèm script deploy vLLM lên Modal chạy
     Qwen2.5-7B-Instruct-AWQ trên L4 24GB. Hỗ trợ batch inference.
   - Disk cache dùng chung cho cả hai, key = (provider, model,
     prompt_hash, temperature, seed)

6. Giữ nguyên: RRF k=60, cross-encoder reranker.

KIỂM CHỨNG: chạy 10 query mẫu trên fastapi đã index. In ra top-5 của
TỪNG retriever riêng lẻ (dense / sparse / graph) và sau khi fuse+rerank,
để tôi kiểm tra bằng mắt trước khi sang phase sau.
```

**✅ Xong phase khi:** ba retriever chạy độc lập được, fuse ra kết quả hợp lý, Groq và Modal đều gọi được thật, cache hoạt động.

---

# PHASE 3 — LangGraph: Mode QA
### 🤖 Sonnet

```
Phase 3: dựng LangGraph cho mode QA. Dùng LangGraph 1.2.

1. State schema: dùng CHÍNH XÁC `AtlasState` trong CODEATLAS_SPEC.md
   mục 2.3. Chú ý reducer `Annotated[list[dict], add]` cho evidence.

2. Đồ thị nhánh QA:
   START -> router -> plan -> retrieve -> fuse_rerank
        -> [conditional: enough_evidence?] -> plan (loop) | generate -> END

3. Node `retrieve` chạy SONG SONG 3 retriever (graph/vector/bm25)
   bằng fan-out, rồi merge state.

4. Tool set 7 tool như spec mục 2.4. Harness:
   - Trần 8 tool call, vượt thì generate bằng evidence đã có
   - Loop detection: hash (tool_name, args), lặp lần 2 chèn observation
     cảnh báo agent thay vì fail
   - Mọi tool trả JSON có field `source` = {file_path, start, end}
   - Error message actionable, dùng difflib gợi ý symbol gần đúng

5. Checkpointer: SqliteSaver cho dev, PostgresSaver cho prod.
   Cấu hình qua env.

6. Streaming: dùng LangGraph streaming để đẩy từng node transition
   và token ra SSE endpoint FastAPI có sẵn.

7. GIỮ LẠI custom ReAct loop cũ, cho chạy song song được qua flag
   `--orchestrator=custom|langgraph`. Tôi cần so sánh hai cái ở
   phase eval.

8. Câu trả lời cuối BẮT BUỘC có trích dẫn [file.py:12-30].
   Không có source thì nói "không tìm thấy trong codebase",
   TUYỆT ĐỐI không bịa.

9. Trace mỗi node ra JSONL + Prometheus counter/histogram.

KIỂM CHỨNG: chạy 5 câu hỏi mẫu ở cả hai orchestrator, in ra trace
đầy đủ để tôi so sánh.
```

**✅ Xong phase khi:** cả hai orchestrator chạy được cùng bộ câu hỏi, trace đọc được, trích dẫn đúng dòng.

---

# PHASE 4 — LangGraph: Mode Refactor
### 🧠 OPUS ⭐ — phase gây ấn tượng mạnh nhất

```
Phase 4: mode refactor có verification loop. Đây là tính năng khác
biệt của project.

=== LUỒNG ===
impact_analysis -> generate_patch -> sandbox_apply -> run_tests
  ├─ pass -> interrupt(chờ người duyệt) -> commit -> END
  └─ fail -> repair (đọc stderr) -> generate_patch (loop, max 3)
             quá 3 vòng -> báo thất bại kèm log, KHÔNG im lặng bỏ qua

=== NODE ===

1. `impact_analysis`
   Cypher [2] + [3] trong spec. Xuất ra:
   - impacted_symbols: hàm bị ảnh hưởng, kèm distance
   - affected_tests: TẬP TEST TỐI THIỂU cần chạy
   Log rõ: chọn N test trong tổng số M test của repo.
   Con số N/M này là metric quan trọng nhất của phase.

2. `generate_patch`
   LLM sinh unified diff. Ràng buộc:
   - Chỉ sửa file trong impacted_symbols, KHÔNG đụng file khác
   - Output phải là valid unified diff, parse được bằng `patch`
   - Nếu parse fail thì retry với error message, không im lặng bỏ

3. `sandbox_apply`  ← ƯU TIÊN AN TOÀN
   - Copy repo sang thư mục tạm (KHÔNG apply lên repo gốc, bao giờ)
   - Apply patch trong thư mục tạm
   - Chạy trong Docker container có: network=none, read-only mount
     cho code gốc, giới hạn CPU/RAM, timeout cứng 120s
   - Dọn dẹp thư mục tạm ở finally, kể cả khi exception

4. `run_tests`
   - CHỈ chạy affected_tests, không chạy full suite
   - `pytest <danh sách test cụ thể> -x --tb=short --timeout=60`
   - Capture stdout/stderr đầy đủ vào state

5. `repair`
   - Đưa test output (stderr, traceback) vào context
   - Cho LLM đọc lại source của hàm lỗi qua read_source
   - Sinh patch mới
   - Tăng repair_iteration

6. Cổng người duyệt
   - Dùng `interrupt()` của LangGraph
   - Trả về cho người dùng: diff, danh sách test đã chạy, kết quả
   - Resume bằng Command(resume={"approved": true/false})
   - Nếu từ chối: rollback sạch, trả lý do

7. `commit`
   - Chỉ chạy sau khi approved
   - Apply patch lên repo thật, tạo branch mới `codeatlas/<slug>`
   - KHÔNG tự push, KHÔNG tự merge

=== METRIC PHẢI LOG ===
- refactor success rate (test xanh sau <= 3 vòng)
- số vòng repair trung bình
- test selection: |affected_tests| / |all_tests|
- thời gian: graph-selected vs nếu chạy full suite (đo thật cả hai)

=== QUY TRÌNH ===
BƯỚC 1: Trình bày thiết kế sandbox và cơ chế rollback. CHỜ TÔI DUYỆT.
An toàn quan trọng hơn tốc độ — tôi không muốn nó ghi đè repo gốc.
BƯỚC 2: Implement.
BƯỚC 3: Test với 3 task refactor thật trên fastapi:
  (a) đổi tên một hàm private
  (b) thêm một param có default value
  (c) đổi kiểu trả về của một hàm
In ra full trace từng vòng lặp.
```

**✅ Xong phase khi:** 3 task mẫu chạy hết luồng, có ít nhất 1 task cần repair và hồi phục được, sandbox không bao giờ chạm repo gốc, interrupt/resume hoạt động.

---

# PHASE 5 — Eval Harness & Ablation
### 🧠 OPUS ⭐ — phase tạo độ tin cậy

```
Phase 5: eval harness. Đây là phần quyết định project được coi là
engineering nghiêm túc hay chỉ là demo.

1. `codeatlas/eval/gold_generator.py`
   Sinh TỰ ĐỘNG từ graph đã index:
   - 20 câu structural: chọn function có fan-in >= 2, hỏi "hàm nào
     gọi X", đáp án từ Cypher [1]
   - 15 câu impact: chọn function có test, hỏi "đổi X thì test nào
     phải chạy", đáp án từ Cypher [3]
   - 20 task refactor: rename / add-default-param / change-return-type
     trên các function có test coverage
   Xuất JSONL: {id, question, type, gold_answer, gold_sources}
   Seed cố định để tái lập được.

2. `codeatlas/eval/gold_manual.jsonl`
   Template cho 25 câu semantic + multi-hop. Sinh sẵn câu hỏi gợi ý
   dựa trên module lớn nhất trong repo, để trống phần đáp án cho
   tôi annotate.

3. `codeatlas/eval/metrics.py`
   - Retrieval: Precision@k, Recall@k, MRR, nDCG@k
   - Impact: exact set match, Jaccard
   - QA E2E: LLM-as-judge PAIRWISE, có swap order để khử position
     bias. Tính Cohen's kappa giữa judge và human trên 20 mẫu.
   - Refactor: success rate, số vòng repair, test selection precision,
     % test tiết kiệm

4. `codeatlas/eval/run_ablation.py`
   Ba bảng như CODEATLAS_SPEC.md mục 3.3:
   - Bảng A retrieval: 7 cấu hình
   - Bảng B orchestration: custom vs langgraph vs langgraph+refactor,
     CÓ CỘT LOC (đếm số dòng orchestration code của mỗi cách)
   - Bảng C test selection: full suite vs graph-selected
   Mỗi cấu hình chạy 3 lần, báo cáo MEDIAN và độ lệch.
   Xuất markdown vào docs/EVALUATION.md.

5. Reproducibility BẮT BUỘC:
   - temperature=0, seed cố định
   - Ghi vào metadata mỗi run: model name, model version, index
     commit_sha, config hash, timestamp
   - Toàn bộ chạy qua ModalVLLMProvider (KHÔNG dùng Groq — sẽ chạm
     rate limit; xem ngân sách token trong spec mục 3.4)
   - Disk cache bật, để chạy lại không tốn token

=== QUY TRÌNH ===
BƯỚC 1: Trình bày thiết kế eval — cụ thể là cách sinh gold set tự
động có tránh được bias không (ví dụ: chỉ chọn hàm dễ resolve thì
kết quả sẽ đẹp giả tạo). CHỜ TÔI DUYỆT.
BƯỚC 2: Implement.
BƯỚC 3: Chạy full ablation, xuất docs/EVALUATION.md.

QUAN TRỌNG: nếu kết quả KHÔNG khớp giả thuyết (ví dụ graph không
thắng vector ở nhóm impact), BÁO CÁO ĐÚNG NHƯ VẬY kèm phân tích
nguyên nhân. Không được tinh chỉnh gold set cho ra số đẹp.
```

**✅ Xong phase khi:** ba bảng có số thật, chạy lại ra kết quả tương đương, `EVALUATION.md` đọc được như một mục trong paper.

---

# PHASE 6 — Demo, Docs, CI
### 🤖 Sonnet

```
Phase 6: hoàn thiện để trình diễn.

1. Demo CLI (rich/textual):
   - Nhập query -> hiện LIVE từng node transition của LangGraph
     (plan / retrieve / rerank / generate) với spinner
   - Mode refactor: hiện diff có màu, danh sách test đang chạy,
     kết quả từng vòng repair, prompt duyệt y/n
   - Đây là thứ tôi share screen khi phỏng vấn -> ưu tiên trực quan
   - Bấm giờ: một demo QA phải xong dưới 30s, refactor dưới 90s

2. README.md:
   - One-liner + GIF demo NGAY ĐẦU FILE
   - Vấn đề giải quyết
   - Kiến trúc: mermaid diagram (cả pipeline + LangGraph state machine)
   - Ba bảng ablation
   - Quickstart 3 lệnh: docker compose up -> ingest -> query
   - Link tới DESIGN_DECISIONS.md

3. `docs/DESIGN_DECISIONS.md` — format Context / Options considered /
   Decision / Consequences cho mỗi quyết định:
   - Tại sao Neo4j (và khi nào Qdrant là đủ)
   - Tại sao RRF thay vì weighted sum
   - Tại sao LangGraph cho refactor nhưng giữ custom loop cho QA
   - Tại sao chunk theo function
   - Tại sao Groq cho demo và Modal cho eval
   - Tại sao đánh dấu unresolved thay vì đoán
   Mỗi mục phải có phần Consequences trung thực, kể cả bất lợi.

4. GitHub Actions:
   - lint (ruff), type check (mypy), test (pytest)
   - job chạy eval rút gọn trên repo nhỏ để bắt regression

5. Kiểm chứng cuối:
   - Chạy quickstart trên container TRỐNG hoàn toàn
   - `grep -ri "mock" codeatlas/` -> phải rỗng
   - Quay video demo dự phòng (phòng khi wifi hỏng lúc phỏng vấn)
```

**✅ Xong phase khi:** người lạ clone repo và chạy được trong 5 phút chỉ đọc README.

---

# Theo dõi tiến độ

| Phase | Model | Xong | Tiêu chí nghiệm thu |
|---|---|---|---|
| 0 | Sonnet | ☐ | `grep -ri mock` sạch, có MIGRATION_AUDIT.md |
| 1 | **Opus** | ☐ | Resolve rate ≥ 80% trên fastapi, 15 test xanh |
| 2 | Sonnet | ☐ | 3 retriever chạy độc lập, Groq + Modal gọi thật được |
| 3 | Sonnet | ☐ | Hai orchestrator chạy song song so sánh được |
| 4 | **Opus** | ☐ | 3 task refactor, ≥1 task cần repair và hồi phục |
| 5 | **Opus** | ☐ | Ba bảng ablation có số thật, tái lập được |
| 6 | Sonnet | ☐ | Người lạ chạy được trong 5 phút |

---

# Ba cạm bẫy hay gặp

**1. Bỏ qua bước "chờ tôi duyệt" ở Phase 1, 4, 5.**
Ba phase Opus đều có bước trình bày thiết kế trước. Đừng để Claude Code nhảy thẳng vào code — đó chính là chỗ sai lầm đắt nhất phát sinh.

**2. Tối ưu trước khi có baseline.**
Đừng tinh chỉnh retrieval ở Phase 2 khi chưa có gold set ở Phase 5. Nếu lỡ làm, hãy thành thật kể chuyện đó khi phỏng vấn — nó là bài học thật.

**3. Để số liệu đẹp giả tạo.**
Nếu gold generator chỉ chọn hàm dễ, kết quả sẽ đẹp mà vô nghĩa. Đã yêu cầu Claude Code tự phản biện điểm này ở Phase 5. Kết quả âm vẫn là kết quả — và kể được kết quả âm là dấu hiệu của engineer thật.