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
- [x] Build keywords (local stream 158K) → 248K điều thô.
- [x] Phân tích: 53% là Quyết định + Nghị quyết (nhiễu hành chính) → **lọc còn 93K điều / 5131 vb** (Luật/Bộ luật/NĐ/TT/Pháp lệnh). Bỏ idea dedup-version (clean_name cắt cụt → nguy hiểm).
- [x] `KEEP_LEGAL_TYPES` bỏ QĐ/NQ (build sau ra 93K trực tiếp).
- [x] **Build trên Kaggle** (Phase 0 cell: clone repo + stream) — tránh upload 209MB qua mạng yếu. Committed.
- [x] User chạy Kaggle corpus 93K → retrieved.json (time2). Top-1 rerank: median −1.05→**0.11**, tự tin >0: 42.7%→**50.6%**.
- [x] **Leaderboard vòng corpus-93K: ARTICLES_F2 0.3877 → 0.4616 (+19%)**, DOCS_F2 0.492. Recall vượt precision (R0.51/P0.41) → nút thắt mới = precision.
- [ ] (sau) targeted vbpl.vn scrape cho vb null-markdown.

### Phase 2b — Lọc HIỆU LỰC văn bản (precision) ⏳
Report: `researcher-260610-1155-validity-filter-precision-tuning.md`. Dataset KHÔNG có trường hiệu lực → heuristic 2 lớp trong `retrieval_cutoff.py`:
- [x] `SUPERSEDED_DOCS` curated 18 cặp (LDN 68/2014→59/2020, BLLĐ 10/2012→45/2019, BHXH 58/2014→41/2024, Đất đai 45/2013→31/2024, TCTD 47/2010→32/2024...) — đo được 1543 candidate / 735 câu dính.
- [x] `drop_superseded()`: curated drop + same-name-keep-newest (trong top-12/câu), never-empty, lọc TRƯỚC cutoff.
- [x] Sweep vòng 2: `f_t3m3 / f_t4m35 / f_t5m4 / f_t6m6` + `raw_t4m35` (đối chứng). Bản f_ sạch 100% superseded; raw còn 440.
- [x] **Vòng 2 leaderboard: filter hiệu lực THẮNG** — f_t3m3 **0.4887** (P0.467/R0.519, DOCS_F2 0.5162) vs t3m3 0.4616 (+0.027); A/B sạch f_t4m35 0.4705 vs raw_t4m35 0.4544 (+0.016). Đỉnh vẫn cutoff chặt t3m3.
- [x] Bake `drop_superseded` + cutoff t3m3 vào notebook Kaggle (cell 5b get_ctx) → Phase B dùng config thắng.
- [x] Sweep vòng 3 (vi chỉnh quanh t3m3): `f_t2m25` (1.64 đ/c) / `f_t3m2` (1.92) / `f_t3m4` (2.38) / `f_t3m3_sib` (2.67 đ/c, same docs — test giả thuyết DOCS_R>ART_R = thiếu điều cùng văn bản).
- [x] **Vòng 3: f_t3m2 = 0.4975 ĐỈNH MỚI** (P0.49/R0.519, DOCS_F2 0.5348) — margin 2.0 tăng precision +0.023, recall giữ nguyên. **sib-expand BỊ BÁC** (0.4766: P sập 0.467→0.418, R chỉ +0.007 → điều "anh em" kéo thêm đa số sai).
- [x] Chốt f_t3m2 → bake vào notebook (cell 5b) + backend config (TOP_K=3, MARGIN=2.0). Cutoff KỊCH TRẦN trên phễu CAND-20.
- [x] HyDE BỊ LOẠI (user): embed văn bản LLM bịa = rủi ro ảo giác + không hợp sản phẩm thật. Thay bằng:

### Phase 4 — Mở phễu rerank + (tùy chọn) query rewriting ⏳
Trần recall 0.519 = giới hạn của top-12 từ phễu RRF-20. Điều đúng hạng 21-50 chưa bao giờ được rerank.
- [x] Notebook Phase A: **CAND 20→50** (rerank 50 ứng viên/câu, ~+10-15p; corpus_emb tái dùng — corpus KHÔNG đổi).
- [x] Sweep GRID vòng 4: `c50_t3m2` (anchor vs 0.4975) / `c50_t3m15` / `c50_t3m25` / `c50_t4m2`.
- [ ] User: Kaggle xóa retrieved.json cũ → chạy Phase A (Add Input corpus_emb.npy time2 để skip embed) → tải retrieved.json → sweep → nộp 4 bản.
- [ ] Nếu c50 plateau: **query rewriting runtime** (Qwen viết lại câu hỏi sang thuật ngữ pháp lý — KHÔNG bịa nội dung; retrieve bằng cả gốc+viết lại qua RRF; rerank vẫn dùng câu GỐC → chống ảo giác; chạy mọi query → tổng quát cho user thật).
- [ ] Backup precision: LLM-verify từng candidate (tác động sau thuần túy, ~1-2h GPU).

### Phase 5 — Phase B (LLM answers) cho QA promote ⬜ — deadline 30/06
QA (4 tiêu chí) chấm bài được promote, MỖI TUẦN 1 LẦN → cần ÍT NHẤT 1 bài có answer LLM thật trước hạn.
- [ ] Khi retrieval chốt (sau vòng 4): chạy Phase B 1 lần (2-4h, get_ctx đã bake config thắng) → nộp + promote. Mục tiêu: xong trước ~20/06 để còn 1-2 kỳ chấm QA.

**Leaderboard journey: 0.317 → 0.3877 (cutoff) → 0.4616 (corpus 93K) → 0.4887 (lọc hiệu lực) → 0.4975 (t3m2) = +57%.**

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
