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
from retrieval_cutoff import apply_cutoff, drop_superseded

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_CIT_MARKER = "\n\nCăn cứ pháp lý áp dụng:"

# (tag, top_k, margin, min_score, validity_filter, sibling_expand) — lưới VÒNG 4 (phễu CAND=50).
# Vòng 3 chốt: f_t3m2 = 0.4975 (P0.49/R0.519) — đỉnh mới; sib-expand THUA (0.4766, P sập) → bác.
# Vòng 4 chạy trên retrieved.json MỚI (Phase A CAND 20→50): cùng cutoff f_t3m2 làm anchor để
# đo riêng tác dụng mở phễu; vi chỉnh margin quanh 2.0 vì phễu rộng đổi phân bố điểm.
GRID = [
    ("c50_t3m2",  3, 2.0, None, True, False),  # ANCHOR — so trực tiếp với f_t3m2/CAND-20 = 0.4975
    ("c50_t3m15", 3, 1.5, None, True, False),  # chặt hơn (phễu rộng → nhiều noise điểm gần nhau?)
    ("c50_t3m25", 3, 2.5, None, True, False),  # lỏng hơn 1 nấc
    ("c50_t4m2",  4, 2.0, None, True, False),  # cap 4 (phễu rộng có thể thêm điều đúng thứ 4)
]

# Sibling expand: sau cutoff, với mỗi văn bản đã giữ → thêm tối đa 1 điều TỐT NHẤT còn lại
# của cùng văn bản (từ top-12 đã lọc hiệu lực) nếu điểm >= top - SIB_MARGIN. Cap tổng SIB_CAP.
SIB_MARGIN, SIB_CAP = 5.0, 5


def expand_siblings(ctx, cands):
    if not ctx:
        return ctx
    top = ctx[0].get("score", 0.0)
    have = {(d.get("doc_number"), d.get("article")) for d in ctx}
    kept_docs = [d.get("doc_number") for d in ctx]
    out = list(ctx)
    for dn in kept_docs:
        if len(out) >= SIB_CAP:
            break
        sib = next((c for c in cands
                    if c.get("doc_number") == dn
                    and (c.get("doc_number"), c.get("article")) not in have
                    and c.get("score", -99) >= top - SIB_MARGIN), None)
        if sib:
            out.append(sib); have.add((sib.get("doc_number"), sib.get("article")))
    return out


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
    print(f"\n{'tag':10} {'avg_art':>8} {'avg_doc':>8}")
    for tag, tk, mg, mn, filt, sib in GRID:
        rows, n_art, n_doc = [], 0, 0
        for q in questions:
            qid = int(q["id"])
            cands = cache.get(qid, [])
            if filt:
                cands = drop_superseded(cands)   # lọc hiệu lực TRƯỚC cutoff → slot đôn lên
            ctx = apply_cutoff(cands, tk, mn, mg, score_key="score")
            if sib:
                ctx = expand_siblings(ctx, cands)  # kéo thêm điều cùng văn bản đã giữ
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
        print(f"{tag:10} {n_art/len(questions):8.2f} {n_doc/len(questions):8.2f}  → submission_{tag}.zip")

    print(f"\n✓ {len(GRID)} biến thể trong {args.outdir}/ — nộp lần lượt lên leaderboard, ghi F2 lại.")


if __name__ == "__main__":
    main()
