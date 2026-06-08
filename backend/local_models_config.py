"""Central configuration for the offline competition pipeline.

Keeps model names, paths, and the curated SME law allowlist in ONE place so
ingestion / retrieval / corpus-building stay in sync (DRY).

Compliance (competition rules): all models are open-source, <14B params, and
released before 2026-03-01.
  - Embedding : AITeamVN/Vietnamese_Embedding (BGE-M3, 0.6B, 1024-dim, MIT, 2025)
  - Reranker  : AITeamVN/Vietnamese_Reranker  (BGE-reranker-v2-m3, 0.6B, MIT, 2025)
  - LLM       : Qwen2.5-7B-Instruct           (7.6B, Apache-2.0, 2024-09) via Ollama
"""
import os

# ---- Paths ------------------------------------------------------------------
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BACKEND_DIR, "..", "data"))
CHROMA_PATH = os.path.join(DATA_DIR, "chroma_db")   # legacy, no longer used
CORPUS_JSONL = os.path.join(DATA_DIR, "corpus_articles.jsonl")
CORPUS_EMB = os.path.join(DATA_DIR, "corpus_emb.npy")        # float32 (N×dim), row i ↔ jsonl line i
CORPUS_EMB_IDS = os.path.join(DATA_DIR, "corpus_emb_ids.json")  # id order in .npy (incremental check)
TEST_QUESTIONS = os.path.join(DATA_DIR, "test_questions.json")
GOLD_DEV = os.path.join(DATA_DIR, "gold_dev.json")

# ---- Models -----------------------------------------------------------------
EMBEDDING_MODEL = "AITeamVN/Vietnamese_Embedding"
RERANKER_MODEL = "AITeamVN/Vietnamese_Reranker"
EMBEDDING_DIM = 1024
# BGE-M3 default max_seq_length=8192 (fine-tuned at 1024). NOT capping it makes
# encoding O(n²)-heavy & very slow. Legal articles fit well under 1024 tokens.
EMBED_MAX_SEQ_LEN = 1024   # embedding (bi-encoder)
RERANK_MAX_LEN = 512       # reranker (query + article snippet)

# ---- Retrieval cutoff (đẩy precision) ---------------------------------------
# Thay vì luôn trả top_k cố định → cắt theo điểm reranker. Luôn giữ top-1 (sàn recall),
# giữ thêm điều nào điểm còn cao. Giá trị tinh chỉnh trên dev set (scratch/tune_retrieval.py).
# Tuned trên 20-mock dev (scratch/tune_retrieval.py): F2 0.586→~0.71, precision 0.285→~0.70.
# Chọn cấu hình ROBUST (không phải max-F2-dev) để khỏi overfit + giữ recall trên tập 2000 câu.
RETRIEVE_TOP_K = 8          # cap cứng số điều trả về (rộng cho câu nhiều điều)
RETRIEVE_MIN_SCORE = 0.0    # chỉ giữ điều có điểm reranker >= 0 (cắt điều rõ ràng không liên quan)
RETRIEVE_MARGIN = 4.0       # giữ điều có điểm >= (điểm top - 4)
OLLAMA_MODEL = "qwen2.5:7b-instruct-q4_K_M"
OLLAMA_HOST = "http://localhost:11434"

CHROMA_COLLECTION = "vietnamese_laws"


def get_device() -> str:
    """Best available torch device: CUDA (Colab/Kaggle GPU) > MPS (Mac) > CPU."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# ---- HuggingFace corpus datasets (recommended by organizers) ----------------
HF_VBPL = "tmquan/vbpl-vn"            # config="documents" — primary statutory corpus
HF_PHAPDIEN = "tmquan/phapdien-moj-gov-vn"   # config="articles" — codified (secondary)
HF_ANLE = "tmquan/anle-toaan-gov-vn"         # config="documents" — case law (optional)

# ---- SME-relevant law allowlist (mã văn bản) --------------------------------
# Curated core laws covering the competition's SME scope (doanh nghiệp, lao động,
# thuế, thương mại, BHXH...). doc_number in vbpl-vn is matched case-insensitively.
# Extend freely — coverage here is the ceiling for retrieval recall.
SME_LAW_ALLOWLIST = {
    # ── Core (đã verify trong dataset, 2411 Điều) ──
    "59/2020/QH14",   # Luật Doanh nghiệp
    "04/2017/QH14",   # Luật Hỗ trợ doanh nghiệp nhỏ và vừa
    "80/2021/NĐ-CP",  # NĐ hướng dẫn Luật Hỗ trợ DNNVV
    "01/2021/NĐ-CP",  # NĐ về đăng ký doanh nghiệp
    "45/2019/QH14",   # Bộ luật Lao động
    "145/2020/NĐ-CP", # NĐ hướng dẫn BLLĐ về điều kiện & quan hệ lao động
    "36/2005/QH11",   # Luật Thương mại
    "58/2014/QH13",   # Luật Bảo hiểm xã hội (2014)
    "38/2019/QH14",   # Luật Quản lý thuế
    "61/2020/QH14",   # Luật Đầu tư
    "91/2015/QH13",   # Bộ luật Dân sự
    "88/2015/QH13",   # Luật Kế toán
    # ── Bổ sung theo phân tích 2000 câu test (~400 câu chưa phủ) ──
    "50/2005/QH11",   # Luật Sở hữu trí tuệ            (~134 câu)
    "07/2022/QH15",   # Luật sửa đổi, bổ sung Luật SHTT 2022
    "123/2020/NĐ-CP", # NĐ về hóa đơn, chứng từ        (~121 câu)
    "47/2010/QH12",   # Luật các tổ chức tín dụng 2010 (~74 câu)
    "32/2024/QH15",   # Luật các tổ chức tín dụng 2024
    "45/2013/QH13",   # Luật Đất đai 2013              (~24 câu)
    "31/2024/QH15",   # Luật Đất đai 2024
    "200/2014/TT-BTC",# TT chế độ kế toán doanh nghiệp (có thể KHÔNG có trên vbpl-vn)
    "133/2016/TT-BTC",# TT chế độ kế toán DN nhỏ và vừa (có thể KHÔNG có trên vbpl-vn)
}

# Document types to keep when scanning vbpl-vn (legal_type field, lowercased contains).
# Only gates --mode keywords; allowlist mode bypasses this (matches by doc_number).
KEEP_LEGAL_TYPES = ("luật", "bộ luật", "nghị định", "thông tư", "nghị quyết",
                    "quyết định", "pháp lệnh")

# Keyword fallback for --mode keywords (broader sweep beyond the allowlist).
SME_TITLE_KEYWORDS = (
    "doanh nghiệp", "lao động", "thương mại", "đầu tư", "thuế",
    "bảo hiểm xã hội", "kế toán", "hỗ trợ doanh nghiệp",
    "sở hữu trí tuệ", "hóa đơn", "tổ chức tín dụng", "đất đai",
)
