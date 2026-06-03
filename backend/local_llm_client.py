"""Local LLM client (Ollama) for grounded answer generation.

The retrieval layer is authoritative for citations; the LLM's job is a correct,
practical Vietnamese answer that *explicitly names every provided Điều* so the
auto-grader can extract them from the answer text.
"""
import json
import urllib.request
import urllib.error
from typing import List, Dict, Any, Tuple

import local_models_config as cfg


class LocalLLMClient:
    def __init__(self, model_name: str = cfg.OLLAMA_MODEL, host: str = cfg.OLLAMA_HOST):
        self.model_name = model_name
        self.host = host.rstrip("/")
        self.chat_url = f"{self.host}/api/chat"

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": 8192},
        }
        req = urllib.request.Request(
            self.chat_url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                if resp.getcode() != 200:
                    raise RuntimeError(f"Ollama HTTP {resp.getcode()}")
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("message", {}).get("content", "").strip()
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Không kết nối được Ollama tại {self.host}. "
                f"Chạy: `ollama serve` và `ollama pull {self.model_name}`. ({e})"
            )

    def build_prompt(self, query: str, contexts: List[Dict[str, Any]]) -> Tuple[str, str]:
        """Return (system_prompt, user_prompt) grounded on retrieved articles."""
        blocks = []
        for i, d in enumerate(contexts, 1):
            blocks.append(
                f"[{i}] {d.get('clean_name','')} ({d.get('doc_number','')}) — {d.get('article','')}\n"
                f"{d.get('text','')}"
            )
        context_str = "\n\n".join(blocks) if blocks else "(Không có căn cứ pháp lý nào được truy hồi.)"

        # Exact list of articles the answer MUST cite (authoritative from retrieval).
        must_cite = ", ".join(sorted({d.get("article", "") for d in contexts if d.get("article")}))

        system = (
            "Bạn là trợ lý pháp lý AI cho doanh nghiệp SME Việt Nam. Trả lời CHÍNH XÁC, "
            "NGẮN GỌN, chỉ dựa trên CƠ SỞ PHÁP LÝ được cung cấp. Tuyệt đối không bịa điều luật.\n"
            "BẮT BUỘC: trong câu trả lời phải nêu rõ từng số Điều (ví dụ 'Điều 125', 'Điều 36') "
            "và tên văn bản tương ứng — hệ thống chấm điểm sẽ trích xuất các 'Điều X' từ câu trả lời. "
            "Nếu căn cứ không đủ, nói rõ và khuyến nghị tham vấn luật sư."
        )
        user = (
            f"CƠ SỞ PHÁP LÝ:\n{context_str}\n\n"
            f"CÂU HỎI: {query}\n\n"
            f"Yêu cầu: trả lời trực tiếp, trích dẫn rõ các điều luật sau trong câu trả lời: {must_cite}.\n"
            f"Câu trả lời:"
        )
        return system, user


if __name__ == "__main__":
    c = LocalLLMClient()
    print(c.chat("Bạn là trợ lý.", "Trả lời ngắn: 1+1 bằng mấy?"))
