# Cải thiện IR retrieval (F2) — toàn diện

**Bối cảnh:** Bài nộp đầu tiên (real 50-Q dev gold): ARTICLES_F2 **0.317** (P0.41/R0.32), DOCS_F2 **0.377** (P0.42/R0.38). QA=0 (chưa chấm, đúng spec). Mục tiêu: kéo F2 lên bằng recall (F2 nặng recall 2×) + đúng luật hơn.

**Chẩn đoán (data từ retrieved.json):**
- 87% câu chỉ trả 1 điều (avg 1.19) → cutoff quá chặt → giết recall.
- ~60% câu ra sai văn bản (DOCS P/R ~0.4) → retrieval quality + coverage.
- Không có gold thật → dùng **leaderboard (10 bài/ngày, tới 30/06)** làm oracle.

## Phases

### Phase 1 — Tách cutoff khỏi retrieval (backbone) ✅
Lưu top-N candidate KÈM điểm rerank → cutoff thành bước rẻ, sweep offline (không GPU).
- [x] `backend/retrieval_cutoff.py` — hàm cutoff thuần (dùng chung engine/submission/sweep)
- [x] config: nới cutoff mặc định (top6/margin6/min None) + `RETRIEVE_CAND_SAVE=12`
- [x] `local_rag_engine` → gọi `apply_cutoff` chung
- [x] Kaggle Phase A: lưu top-12 + `score` (bỏ cutoff inline)
- [x] Kaggle gen (B1b/B2): áp cutoff qua `get_ctx` (CUT_TOP_K/MARGIN/MIN ở cell 5b)
- [x] `generate_submission --retrieved`: áp cutoff (cache giàu)
- [x] `scratch/sweep_cutoff.py`: sinh biến thể results_*.json (giữ prose LLM, gắn lại căn cứ)
- [x] + **weighted RRF (Phase 3 #1)** bundle vào Phase A: BM25 0.65 / dense 0.35 (re-run 1 lần được cả 2)
- [x] validate: py_compile OK, notebook 0 syntax err, smoke-test cutoff/reattach OK

**→ User chạy lại Phase A 1 lần (rich + weighted RRF) → Phase B (base answers) → sweep cutoff → nộp ~5 biến thể → chọn tốt nhất.**

### Phase 2 — Coverage (data) ⏳ — NÚT THẮT SỐ 1 (xác nhận qua leaderboard)
Bằng chứng: sweep cutoff đỉnh 0.3877 ở 2-3 điều/câu (nới = tụt) + **57% câu rerank top-1 ≤ 0** (không khớp tự tin) + corpus chỉ 32 vb. Report: `researcher-260609-1506-ir-bottleneck-coverage-diagnosis.md`.
- [x] Chẩn đoán: coverage 32 vb quá hẹp là trần recall.
- [x] Chốt hướng (user): **quét rộng vbpl-vn theo chủ đề SME** (`build_corpus --mode keywords`).
- [x] Mở rộng `SME_TITLE_KEYWORDS` (12 → 35 từ khóa: + việc làm/tiền lương/công đoàn/ATVSLĐ/BHYT/BHTN/lệ phí/kiểm toán/cạnh tranh/quảng cáo/NTD/XNK/hải quan/chứng khoán/xây dựng/nhà ở/BĐS/môi trường/PCCC/ATTP/xử phạt VPHC).
- [⏳] Chạy `build_corpus --mode keywords` (đang build nền — local stream 158K vb).
- [ ] Upload corpus mới lên Kaggle → re-embed (KHÔNG upload corpus_emb.npy cũ) → Phase A → retrieved.json mới.
- [ ] Sweep cutoff lại trên corpus mới → nộp đo F2.
- [ ] (sau) dọn clean_name trùng hoa/thường (cosmetic; join theo mã).

### Phase 3 — Retrieval quality ⏳ (report `researcher-260609-1340-vn-legal-retrieval-quality.md`)
Ước tính combine #1+#2+#3 → F2 0.317 → **0.41-0.46**.
- [x] #1 **weighted RRF** BM25 0.65/dense 0.35 (+0.08-0.12) — đã bundle vào Phase 1.
- [ ] #2 rerank pool 10→15-20 + softer margin (sweep ở Phase 1) + sigmoid-normalize điểm (+0.05-0.09).
- [ ] #3 **offline HyDE** — Qwen viết lại câu hỏi sang văn phong luật trước khi embed (+0.03-0.07). `backend/query_expansion.py`.
- [ ] #6 boost BM25 khi query có số luật/Điều (regex) — ít giá trị vì câu hỏi SME hiếm khi cite số.
- [ ] Giữ article-level chunking (report khuyến nghị — khớp grader mã|Điều).
- [ ] Verify format join `relevant_docs/articles` vs grader (chuẩn hóa mã văn bản).

### Phase 4 — Validation loop ⬜
- [ ] Mỗi thay đổi → 1 bài leaderboard → ghi P/R/F2 → giữ/bỏ. Ghi log biến thể (no silent caps).

## Quyết định
- vLLM bỏ (lệch CUDA-build Kaggle) → transformers fp16 2×T4.
- numpy cosine giữ (corpus nhỏ). Không ChromaDB/turbovec.

## Câu hỏi mở
1. 50 câu gold là câu nào (id 1-50? random?) — không biết → chỉ có tín hiệu tổng hợp.
2. Grader ghép article identifier từ `relevant_articles` field hay parse `answer`? (spec mơ hồ §6.1) — ảnh hưởng cách tối ưu.
