# CodeAtlas

Agentic **repository intelligence**: parse AST → call graph trong Neo4j + code chunk trong Qdrant → agent định tuyến giữa hai nguồn để trả lời "đổi hàm này thì cái gì hỏng", và ở mode refactor thì tự sinh patch, chạy đúng tập test bị ảnh hưởng, đọc lỗi và sửa lại.

Repo này đang **chuyển domain từ NeuralTwin** (digital twin cá nhân) sang CodeAtlas. Giữ ~70% hạ tầng, thay tầng domain và tầng orchestration.

## Đọc trước khi làm bất cứ việc gì

| File | Nội dung |
|---|---|
| `docs/CODEATLAS_SPEC.md` | Kiến trúc đích. Graph schema §2.2, `AtlasState` §2.3, tool set §2.4, eval §3. **Là nguồn sự thật.** |
| `docs/codeatlas_roadmap.md` | Thứ tự thi công 7 phase + prompt từng phase. Mục **"⚠️ Đính chính sau Phase 0/0.5"** ở đầu file **thắng** mọi chỗ mâu thuẫn trong prompt phía dưới. |
| `docs/MIGRATION_AUDIT.md` | Bảng KEEP/MODIFY/DELETE từ Phase 0 |

## Trạng thái hiện tại

**Xong:** Phase 0 (audit, xoá crawler social, rename `llm_engineering`→`codeatlas`), Phase 0.5 (xoá finetuning/SageMaker/AWS, xoá agent stub, sửa điểm gãy import).

**Tiếp theo:** Phase 1 — AST ingester + symbol resolver.

**Test:** `pytest` xanh 9/9 — nhưng chỉ phủ `clean_text` (5) + Mongo connector (3) + 1 example. **Không có test nào chạm retrieval, graph, hay agent.** Đây không phải lưới an toàn.

**Còn mock có chủ ý:** 6 file `application/rag/*`, `application/ai_facade.py`, `application/graph/ingestor.py`. Xoá ở **Phase 2** khi đã có `GroqProvider`/`ModalVLLMProvider` thật — xoá sớm hơn thì `/rag` chết mà chưa có gì thay thế.

## Luật cứng

- **Không thêm mock mới.** Không `MOCK_*` flag, không nhánh fallback trả dữ liệu giả, không hardcode kết quả theo keyword. Chưa implement được thì `raise NotImplementedError` với thông báo rõ. Repo này đã một lần suýt mang số liệu đo từ pipeline có mock ở giữa đi phỏng vấn — đừng lặp lại.
- **Mọi module mới phải kèm test của chính nó.** Không giả định có sẵn gì đỡ phía dưới.
- **Không đoán bừa khi resolve symbol.** Không chắc thì gán `confidence` thấp + `unresolved=true`. Downstream là impact analysis: một edge sai = một test bị bỏ sót, tệ hơn nhiều so với thiếu edge.
- **Không bịa trích dẫn.** Câu trả lời phải có `[file.py:12-30]` trỏ đúng dòng thật; không tìm thấy thì nói "không tìm thấy trong codebase".
- **Kết quả âm vẫn phải báo đúng.** Nếu số liệu không khớp giả thuyết, báo cáo như vậy kèm phân tích. Không tinh chỉnh gold set cho ra số đẹp.
- **Phase 1, 4, 5 có bước "trình bày thiết kế, chờ duyệt" trước khi code.** Đừng nhảy thẳng vào implement.

## Môi trường

```bash
export PATH="$HOME/.local/bin:$PATH"   # poetry KHÔNG có sẵn trên PATH mặc định
poetry install --no-root
poetry run pytest tests/ -v
```

venv: `~/.cache/pypoetry/virtualenvs/codeatlas-k_0uQjOe-py3.11`. Python 3.11.
`torch` vẫn được cài như transitive dependency của `sentence-transformers` (đã gỡ khỏi `pyproject.toml` dưới dạng explicit dep) — lần install đầu tải khá lâu.

**Báo cáo trung thực:** `py_compile` không chứng minh được gì về runtime. Nói "xong" chỉ khi đã chạy `pytest` thật và dán kết quả.

## Giữ nguyên, đừng đụng vào

Qdrant client, `infrastructure/graph/neo4j_adapter.py`, RRF (k=60), cross-encoder reranker, ZenML, Kafka, FastAPI + SSE, Redis, Prometheus/Grafana/Jaeger, Docker/K8s manifests.
