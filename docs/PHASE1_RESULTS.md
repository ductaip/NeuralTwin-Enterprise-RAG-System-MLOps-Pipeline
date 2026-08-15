# Phase 1 — Kết quả đo

> Repo đo: `fastapi` (clone `--depth 1`, 1136 file .py). Mọi số dưới đây chạy thật, không phải ước lượng.
> Ngày: 2026-08-12. Ingest chạy trên máy local, Neo4j qua `docker-compose up neo4j`.

---

## 1. Ba rổ và tỉ lệ resolve

Mẫu số được **chốt trước khi đo**: `internal + unresolved`. Lời gọi `external` (stdlib, thư viện ngoài) là **kết quả đúng** — `os.path.join` resolve thành công ra một thứ nằm ngoài index. Gộp chúng vào bất kỳ hướng nào cũng làm méo con số.

| | Toàn repo | Chỉ thư viện lõi `fastapi/` |
|---|---|---|
| File | 1136 | 48 |
| Function/method | 4959 | 387 |
| Class | 692 | 115 |
| Call site | 15793 | 1845 |
| → internal | 4919 | 675 |
| → external | 9448 | 830 |
| → unresolved | 1426 | 340 |
| **Resolve rate (nội bộ)** | **77.5%** | **66.5%** |

**Chưa đạt mục tiêu 80%, vượt sàn 70% ở toàn repo, nhưng dưới sàn ở thư viện lõi.**

Giả thuyết ban đầu của tôi sai ngược: tôi đoán `docs_src/` (hơn 1000 file tutorial) kéo tỉ lệ xuống, nên tách ra đo riêng. Thực tế tutorial **kéo tỉ lệ lên** — chúng dùng `app.get`/`app.post` đơn giản, còn code thư viện dùng dynamic dispatch dày đặc. Khi trình bày phải nói cả hai số và giải thích chênh lệch, không chọn số đẹp.

---

## 2. Bốn lần sửa, mỗi lần do histogram chỉ ra

Tỉ lệ đi qua: **30.6% → 47.9% → 59.3% → 76.0% → 77.5%**. Mỗi bước nhảy là một pattern chiếm đa số trong `unresolved_report.json` — đây chính là lý do báo cáo theo pattern thay vì chỉ tổng số.

| # | Pattern chiếm đa số | Nguyên nhân gốc | Sửa |
|---|---|---|---|
| 1 | `self_attr_not_found` (66) | `self._client.send()` bị tra như một method | Bản đồ instance attribute từ `self.x = Foo()`, tra cả theo MRO. Còn **6** |
| 2 | `constructor_unresolvable` (5107, 71.7%) | **Hai đường resolve tách rời**: `_resolve_type_expression` không lần theo re-export, nên *mọi* `app = FastAPI()` fail — `fastapi.FastAPI` là re-export, class thật ở `fastapi.applications.FastAPI` | Gộp về dùng chung `_resolve_module_symbol` |
| 3 | `constructor_unresolvable` (3314) | Externality chỉ xét hop đầu; `TestClient`/`Request` là re-export của **starlette** | Lần hết chuỗi re-export rồi mới phân loại rổ |
| 4 | `assign_from_method_return` (1916, 56.8%) | `client = TestClient()` → `response = client.get()` → `response.json()` | Giá trị dẫn xuất từ object ngoài index thì cũng ở ngoài index — cùng luật với `os.path.join` |

Sửa thêm: literal typing (`data = {}` → `data.get()` là lời gọi `dict`, thuộc rổ external).

### Kết quả âm: return-type inference cải thiện **đúng 0**

Đã implement suy kiểu từ return annotation (`x = build()` với `def build() -> Engine`), có unit test chứng minh chạy đúng (`test_return_annotation_types_the_result`). Đo trên fastapi: tỉ lệ **không đổi** (77.5%, 1426 unresolved trước và sau).

Giữ lại vì đúng nguyên tắc và sẽ hữu ích trên codebase có type hint dày. Báo cáo vì nó không giúp gì ở đây.

### Phần còn lại là giới hạn cố hữu

62% số unresolved còn lại là hai pattern cần dataflow analysis thật, không sửa vặt được:

| Pattern | Số | % unresolved |
|---|---|---|
| `opaque_binding` | 525 | 36.8% |
| `dynamic_callee` | 358 | 25.1% |
| `annotation_unresolvable` | 110 | 7.7% |
| `assign_from_method_return` | 90 | 6.3% |
| `param_untyped` | 63 | 4.4% |

---

## 3. Graph và tính idempotent

Ingest thật vào Neo4j: **8177 node, 16528 relationship**.
Chạy lại trên **cùng commit**: `+0 nodes, +0 relationships`. MERGE theo khoá `(repo_id, qualified_name)`.

| Node | Số | Relationship | Số |
|---|---|---|---|
| Function (+Test) | 4959 | DEFINED_IN | 5651 |
| Module | 1389 | CALLS | 4902 |
| File | 1136 | IMPORTS | 3438 |
| Class | 692 | IN_REPO | 1136 |
| | | TESTS | 1021 |
| | | METHOD_OF | 323 |

### Cypher [4] — fan-in cao

Trả về đúng thứ phải đúng với một web framework, tức graph không rác:

| Hàm | fan-in |
|---|---|
| `fastapi.applications.FastAPI.__init__` | 720 |
| `fastapi.applications.FastAPI.get` | 436 |
| `fastapi.applications.FastAPI.post` | 148 |
| `fastapi.param_functions.Depends` | 104 |
| `fastapi.routing.APIRouter.__init__` | 102 |

---

## 4. Cypher [3] — chọn tập test tối thiểu

Tổng 2340 test node (sau khi loại 199 fixture — xem §5).

| Hàm | conf≥0.5 | conf≥0.9 | % suite |
|---|---|---|---|
| `fastapi.encoders.jsonable_encoder` | 125 | 125 | 5.3% |
| `fastapi.applications.FastAPI.get` | 72 | 57 | 3.1% |
| `fastapi.param_functions.Depends` | 18 | 18 | 0.8% |
| `fastapi.routing.APIRouter.add_api_route` | 2 | 0 | 0.1% |

### Chi phí của ngưỡng thấp — gần như bằng 0

Câu hỏi đặt ra trước khi đo: "recall-first (conf≥0.5) có kéo theo hàng trăm test không?" **Không.**

| Ngưỡng | median | p95 | max |
|---|---|---|---|
| 0.50 | 2 | 97 | 228 |
| 0.70 | 2 | 97 | 228 |
| 0.90 | 2 | 95 | 204 |
| 1.00 | 2 | 66 | 204 |

Median không đổi ở mọi ngưỡng; chỉ p95 dịch (97 → 66). Nghĩa là quyết định "ghi hết edge kèm confidence, quyết ngưỡng ở tầng đọc" gần như miễn phí ở trung vị.

---

## 5. ⚠️ Giới hạn nghiêm trọng của quan hệ TESTS

Kiểm chứng bằng mắt 5 hàm phát hiện **hai vấn đề**, một đã sửa, một là giới hạn thiết kế.

### Đã sửa: fixture bị đánh dấu là Test

199 hàm có decorator `pytest.*` nhưng là **fixture**, không phải test — `tests.test_arbitrary_types.get_client`, `tests.benchmarks...client`. `pytest tests/x.py::get_client` không phải node id chạy được, nên nếu lọt vào `affected_tests` thì Phase 4 sẽ gọi một lệnh không thể thành công. Đã loại trừ decorator chứa `fixture`; sau khi sửa, số fixture lọt vào nhãn Test là **0** (2539 → 2340 test node).

### Chưa sửa: 81% test không nối tới gì

| | Số | % |
|---|---|---|
| Test node | 2340 | 100% |
| Có ≥1 quan hệ TESTS | 449 | **19.2%** |
| Không nối tới bất cứ đâu | 1891 | **80.8%** |

**Nguyên nhân:** test suite của fastapi chạy qua ranh giới HTTP:

```
test_foo()  →  client.get("/")  →  [ranh giới TestClient/HTTP]  →  route handler  →  jsonable_encoder
                                    ↑ đứt ở đây
```

`TestClient` là external, nên chuỗi CALLS đứt tại đó. Test nào gọi thẳng hàm (`tests.test_jsonable_encoder.*` gọi `jsonable_encoder()`) thì nối đúng; test nào đi qua HTTP thì không.

**Hệ quả phải nói thẳng:**
- Tập test được chọn có **precision cao** — kiểm bằng mắt, các test trả về đúng là test của hàm đó
- Nhưng **recall thấp với integration test**, và recall mới là hướng nguy hiểm: thiếu test → bug lọt production
- Con số "5.3% suite" đúng về mặt tính toán nhưng được tính trên một graph mà 81% test bị cô lập

**Đây là giới hạn cố hữu của static call graph**, không phải bug: không có phân tích tĩnh nào bắc được qua dynamic dispatch của một HTTP client. Hướng xử lý tiêu chuẩn trong ngành là **bổ sung dữ liệu coverage** (chạy suite một lần với `coverage`, ghi nhận test nào chạm dòng nào) rồi hợp nhất với graph.

**Cần quyết trước Phase 4**, vì mode Refactor phụ thuộc hoàn toàn vào `affected_tests`.

---

## 6. Test của chính tầng ingestion

57 test xanh (trước Phase 1 là 9):

| File | Số | Phủ |
|---|---|---|
| `test_symbol_resolver.py` | 37 | 8 nhóm case khó + phân loại rổ + fixture |
| `test_repo_loader.py` | 6 | module naming, skip dirs, .gitignore, sha |
| `test_chunker.py` | 5 | 1 chunk/function, metadata, tách hàm dài |
| (có sẵn) | 9 | clean_text, Mongo connector |

---

## 7. Lệch so với spec, cố ý và đã cập nhật vào §2.2

1. `CALLS` có thể trỏ vào `:Class` khi class không có `__init__` — bỏ edge đó thì mất quan hệ dày nhất repo (fan-in 720).
2. Ngưỡng confidence là tham số `$min_confidence`, không hardcode.
3. `qualified_name` mang ngữ nghĩa `__qualname__` kể cả `<locals>` — bắt buộc vì là khoá MERGE.
