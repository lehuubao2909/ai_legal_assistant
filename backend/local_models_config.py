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
CHROMA_PATH = os.path.join(DATA_DIR, "chroma_db")
CORPUS_JSONL = os.path.join(DATA_DIR, "corpus_articles.jsonl")
TEST_QUESTIONS = os.path.join(DATA_DIR, "test_questions.json")
GOLD_DEV = os.path.join(DATA_DIR, "gold_dev.json")

# ---- Models -----------------------------------------------------------------
EMBEDDING_MODEL = "AITeamVN/Vietnamese_Embedding"
RERANKER_MODEL = "AITeamVN/Vietnamese_Reranker"
EMBEDDING_DIM = 1024
OLLAMA_MODEL = "qwen2.5:7b-instruct-q4_K_M"
OLLAMA_HOST = "http://localhost:11434"

CHROMA_COLLECTION = "vietnamese_laws"

# ---- HuggingFace corpus datasets (recommended by organizers) ----------------
HF_VBPL = "tmquan/vbpl-vn"            # config="documents" — primary statutory corpus
HF_PHAPDIEN = "tmquan/phapdien-moj-gov-vn"   # config="articles" — codified (secondary)
HF_ANLE = "tmquan/anle-toaan-gov-vn"         # config="documents" — case law (optional)

# ---- SME-relevant law allowlist (mã văn bản) --------------------------------
# Curated core laws covering the competition's SME scope (doanh nghiệp, lao động,
# thuế, thương mại, BHXH...). doc_number in vbpl-vn is matched case-insensitively.
# Extend freely — coverage here is the ceiling for retrieval recall.
SME_LAW_ALLOWLIST = {
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
}

# Document types to keep when scanning vbpl-vn (legal_type field, lowercased contains).
# Only gates --mode keywords; allowlist mode bypasses this (matches by doc_number).
KEEP_LEGAL_TYPES = ("luật", "bộ luật", "nghị định", "thông tư", "nghị quyết",
                    "quyết định", "pháp lệnh")

# Keyword fallback for --mode keywords (broader sweep beyond the allowlist).
SME_TITLE_KEYWORDS = (
    "doanh nghiệp", "lao động", "thương mại", "đầu tư", "thuế",
    "bảo hiểm xã hội", "kế toán", "hỗ trợ doanh nghiệp",
)
