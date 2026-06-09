"""Sweep cutoff configs OFFLINE → nhiều biến thể results.json cho leaderboard (không cần GPU).

Đọc `retrieved.json` GIÀU (top-N + score, từ Kaggle Phase A bản mới) + `results.json` gốc
(đã có answer LLM) → với mỗi cấu hình cutoff: dựng lại relevant_docs/articles + gắn lại khối
"Căn cứ pháp lý áp dụng" vào answer (giữ prose LLM) → results_<tag>.json + submission_<tag>.zip.

Nộp lần lượt các zip lên leaderboard (10 bài/ngày vòng public) để chọn cutoff tốt nhất —
KHÔNG chạy lại retrieval/LLM. (Cutoff chặt hơn mặc định: prose có thể còn "Điều N" ngoài tập
cutoff → tính vào predicted, hơi giảm precision; sweep chủ yếu để NỚI nên không sao.)

Run:
    python scratch/sweep_cutoff.py --retrieved backup/retrieved.json --base results.json
"""
import argparse
import json
import os
import re
import sys
import zipfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
from retrieval_cutoff import apply_cutoff

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_CIT_MARKER = "\n\nCăn cứ pháp lý áp dụng:"

# (tag, top_k, margin, min_score) — lưới cutoff. Nới dần để đẩy recall (F2 nặng recall 2×).
GRID = [
    ("t3m3",   3, 3.0,  None),
    ("t5m4",   5, 4.0,  None),
    ("t6m6",   6, 6.0,  None),   # = mặc định notebook
    ("t8m10",  8, 10.0, None),
    ("t5flat", 5, None, None),   # top-5 phẳng (recall tối đa, precision thấp)
]


def build_fields(ctx):
    docs, arts = [], []
    for d in ctx:
        c, nm, a = d.get("doc_number", ""), d.get("clean_name", ""), d.get("article", "")
        if not c or not a:
            continue
        if f"{c}|{nm}" not in docs:
            docs.append(f"{c}|{nm}")
        if f"{c}|{nm}|{a}" not in arts:
            arts.append(f"{c}|{nm}|{a}")
    return docs, arts


def reattach_citations(base_answer, ctx):
    """Bỏ khối căn cứ cũ → đảm bảo mọi 'Điều N' trong ctx xuất hiện trong answer text."""
    i = base_answer.find(_CIT_MARKER)
    body = (base_answer[:i] if i != -1 else base_answer).rstrip()
    present = set(re.findall(r"Điều\s+(\d+)", body))
    miss = {}
    for d in ctx:
        m = re.match(r"Điều\s+(\d+)", d.get("article", ""))
        if m and m.group(1) not in present:
            miss.setdefault(d.get("clean_name", ""), []).append(d["article"])
    if not miss:
        return body
    parts = [f"{', '.join(a)} ({nm})" for nm, a in miss.items()]
    return body + _CIT_MARKER + " " + "; ".join(parts) + "."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieved", default=os.path.join(REPO_ROOT, "backup", "retrieved.json"))
    ap.add_argument("--base", default=os.path.join(REPO_ROOT, "results.json"),
                    help="results.json gốc (có answer LLM). Thiếu → answer = chỉ liệt kê căn cứ.")
    ap.add_argument("--questions", default=os.path.join(REPO_ROOT, "data", "stage1_questions.json"))
    ap.add_argument("--outdir", default=os.path.join(REPO_ROOT, "sweep"))
    args = ap.parse_args()

    cache = {int(k): v for k, v in json.load(open(args.retrieved, encoding="utf-8")).items()}
    questions = json.load(open(args.questions, encoding="utf-8"))

    # cache phải GIÀU (có 'score') mới sweep được
    sample = next((v for v in cache.values() if v), [])
    if sample and "score" not in sample[0]:
        sys.exit("⚠ retrieved.json KHÔNG có 'score' (cache cũ post-cutoff, ~1.19 điều). Chạy lại Phase A "
                 "bản mới (lưu top-12+score) rồi mới sweep được.")

    base = {}
    if os.path.exists(args.base):
        base = {int(r["id"]): r.get("answer", "") for r in json.load(open(args.base, encoding="utf-8"))}
        print(f"base answers: {len(base)} câu (giữ prose LLM, gắn lại căn cứ theo cutoff)")
    else:
        print("(không có results.json gốc → answer = chỉ liệt kê căn cứ; QA sẽ kém, chỉ để đo IR)")

    os.makedirs(args.outdir, exist_ok=True)
    print(f"\n{'tag':8} {'avg_art':>8} {'avg_doc':>8}")
    for tag, tk, mg, mn in GRID:
        rows, n_art, n_doc = [], 0, 0
        for q in questions:
            qid = int(q["id"])
            ctx = apply_cutoff(cache.get(qid, []), tk, mn, mg, score_key="score")
            rd, ra = build_fields(ctx)
            n_art += len(ra); n_doc += len(rd)
            if not ctx:
                ans = "Chưa tìm thấy căn cứ pháp lý phù hợp. Khuyến nghị tham vấn luật sư."
            elif base.get(qid):
                ans = reattach_citations(base[qid], ctx)
            else:
                ans = "Căn cứ pháp lý liên quan: " + "; ".join(ra)
            rows.append({"id": qid, "question": q["question"], "answer": ans,
                         "relevant_docs": rd, "relevant_articles": ra})
        rows.sort(key=lambda r: r["id"])
        rj = os.path.join(args.outdir, f"results_{tag}.json")
        json.dump(rows, open(rj, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        with zipfile.ZipFile(os.path.join(args.outdir, f"submission_{tag}.zip"), "w", zipfile.ZIP_DEFLATED) as z:
            z.write(rj, arcname="results.json")
        print(f"{tag:8} {n_art/len(questions):8.2f} {n_doc/len(questions):8.2f}  → submission_{tag}.zip")

    print(f"\n✓ {len(GRID)} biến thể trong {args.outdir}/ — nộp lần lượt lên leaderboard, ghi F2 lại.")


if __name__ == "__main__":
    main()
