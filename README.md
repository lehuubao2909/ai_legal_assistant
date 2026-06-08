# AI Legal Assistant — Vietnamese Legal IR & QA

Hệ thống AI **truy hồi & hỏi đáp văn bản pháp luật tiếng Việt** cho doanh nghiệp SME, xây dựng cho cuộc thi *Vietnamese Legal Information Retrieval & Question Answering*.

Pipeline xử lý hàng loạt offline: nhận câu hỏi pháp lý → truy hồi đúng **Điều luật** liên quan → sinh câu trả lời có dẫn nguồn → xuất `results.json` để nộp.

> 📄 Đề bài chi tiết: [docs/competition-overview.md](docs/competition-overview.md)
> 🏗️ Kiến trúc & giải pháp: [docs/pipeline-architecture.md](docs/pipeline-architecture.md)

---

## Tổng quan

- **Đầu vào:** câu hỏi pháp lý tiếng Việt (`data/test_questions.json`).
- **Đầu ra:** `results.json` — câu trả lời + danh sách văn bản/điều luật liên quan, nén `submission.zip` (zip phẳng).
- **Chấm điểm:** F2 macro (truy hồi, ưu tiên recall) + 5 tiêu chí QA (LLM-judge + chuyên gia).
- **Ràng buộc thi:** chỉ dùng mô hình **mã nguồn mở, < 14B, công bố trước 01/03/2026**. Cấm mô hình đóng (Gemini/GPT).

## Kiến trúc (rút gọn)

```
HF vbpl-vn → build_corpus (--append) → corpus_articles.jsonl → embed_corpus → corpus_emb.npy
                                                                                  │
test_questions → [Dense numpy cosine (Vietnamese_Embedding) ‖ BM25] → RRF → Reranker → cutoff
                                                                                  │
              ┌──── relevant_docs/articles (chân lý từ retrieval) ────────────────┤
              └──── Qwen2.5 (Ollama local / transformers Colab) → answer ── ép 'Điều X' ─┴→ results.json + zip
```

Chi tiết từng giai đoạn + lý do chọn công nghệ: xem [docs/pipeline-architecture.md](docs/pipeline-architecture.md).

## Tech stack

| Lớp | Công nghệ | Lý do |
|---|---|---|
| Embedding | `AITeamVN/Vietnamese_Embedding` (BGE-M3, v1) | SOTA retrieval VN, mạnh domain legal (v1 > v2 ở legal) |
| Reranker | `AITeamVN/Vietnamese_Reranker` | Cross-encoder, tăng precision/recall mạnh nhất |
| LLM | `Qwen2.5-7B-Instruct` (Ollama, local) | Open Apache, <14B |
| Sparse + Fusion | `rank_bm25` + RRF + `pyvi` | Khớp từ khóa luật + gộp rank ổn định |
| Vector store | numpy `.npy` (cosine brute-force) | Corpus ~15MB float32 → tức thì, incremental (append), không cần ChromaDB/FAISS |

Tất cả mô hình tuân thủ ràng buộc cuộc thi (mở, <14B, pre-2026-03-01).

## Cấu trúc thư mục

```
.
├── backend/
│   # ── Pipeline thi (ĐANG TẬP TRUNG) ──
│   ├── local_models_config.py     # Config: model, allowlist luật SME, max_seq_len
│   ├── legal_text_parser.py       # Parse tên luật + tách markdown → Điều
│   ├── build_corpus.py            # HF vbpl-vn → corpus_articles.jsonl (--append: thêm luật mới)
│   ├── embed_corpus.py            # Embed incremental → corpus_emb.npy (chỉ embed Điều mới)
│   ├── local_rag_engine.py        # Hybrid retrieval (numpy cosine + BM25 + RRF + rerank)
│   ├── local_reranker.py          # Cross-encoder reranker
│   ├── local_llm_client.py        # Ollama client + prompt (build_prompt tái dùng cho Colab)
│   ├── requirements-local.txt     # Deps pipeline
│   # ── Web app (GIAI ĐOẠN SAU, chưa dùng) ──
│   ├── main.py · agent.py · rag_engine.py · ingestion.py · document_generator.py
│   ├── requirements.txt           # Deps web-app (FastAPI...)
│   └── static/                    # docx đã upload
├── frontend/ · frontend_vanilla/  # UI web app (giai đoạn sau)
├── scratch/
│   ├── generate_submission.py     # → results.json + submission.zip (local, Ollama)
│   ├── eval_f2.py                 # Đo P/R/F2 nội bộ vs gold_dev
│   └── tune_retrieval.py          # Quét ngưỡng cutoff (top_k/min_score/margin)
├── colab/
│   └── full_pipeline_colab.ipynb  # Notebook full GPU: retrieve + Qwen sinh answer 2000 câu
├── data/
│   ├── corpus_articles.jsonl · corpus_emb.npy       # corpus + embeddings (pipeline thi)
│   ├── test_questions.json · gold_dev.json · stage1_questions.json
│   └── law.json · document_registry.json · legal_list_id.json  # (web-app)
├── docs/                          # Tài liệu (đề bài, kiến trúc)
└── plans/                         # Kế hoạch + báo cáo research
```

## Quick start

```bash
# 1. Cài deps + Ollama
pip install -r backend/requirements-local.txt
ollama serve &
ollama pull qwen2.5:7b-instruct-q4_K_M

# 2. Build corpus + embed (numpy .npy, incremental)
python backend/build_corpus.py            # → data/corpus_articles.jsonl (xem log coverage)
python backend/embed_corpus.py            # → data/corpus_emb.npy (chỉ embed Điều mới khi chạy lại)

# 3. Test retrieval + đo F2 nội bộ
python backend/local_rag_engine.py        # smoke test
python scratch/generate_submission.py --no-llm   # IR-only (không cần Ollama)
python scratch/eval_f2.py

# Thêm luật mới sau này (KHÔNG chạy lại từ đầu):
#   thêm mã vào SME_LAW_ALLOWLIST → python backend/build_corpus.py --append → python backend/embed_corpus.py

# 4. Sinh submission đầy đủ
python scratch/generate_submission.py     # → results.json + submission.zip
```

> Dùng Python interpreter phù hợp (vd `./backend/venv/bin/python`). `--no-llm` cho phép sinh submission hợp lệ để chấm IR trước khi cài Ollama.

## Trạng thái

- ✅ Pipeline code hoàn chỉnh, đã review (2 Critical + 4 High đã fix).
- ⏳ Chưa chạy runtime thực (cần dataset HF + Ollama trên máy).
- ⏳ `gold_dev.json` là gold tạm (Claude tạo) — chỉ để so sánh tương đối.

## Lưu ý

- **`mã văn bản` là khóa chấm điểm** — corpus phải có mã chính xác.
- **Corpus coverage = trần recall** — thiếu luật gold → mất điểm câu đó. Mở rộng allowlist trong `local_models_config.py`.
- **Bảo mật:** không commit `.env` (đã gitignore). `.env.example` chỉ chứa placeholder.
- **Cần làm rõ với BTC:** mâu thuẫn quy định "không dùng dữ liệu ngoài" vs "tự thu thập corpus" (xem [docs/competition-overview.md](docs/competition-overview.md) §9).
