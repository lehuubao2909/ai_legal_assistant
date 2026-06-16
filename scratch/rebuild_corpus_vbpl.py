"""Rebuild corpus SME từ API CHÍNH THỐNG vbpl (thay HF dataset) — coverage + hiệu lực + parse sạch.

3 cái lợi so với corpus HF hiện tại (93K điều, regex-parse, heuristic hiệu lực):
  1. PHỤC HỒI 7611 doc null-markdown bị HF bỏ → recall ceiling (đòn chính, top-1 R0.72 vs ta 0.55).
  2. effStatus CHÍNH THỨC → chỉ giữ "Còn hiệu lực" (bỏ doc hết hiệu lực như 02/2016/TT-BCT) → precision.
  3. prov-article/clause/item → parse Điều chuẩn (khỏi regex đoán).

Chạy trên KAGGLE (mạng nhanh; fetch ~12K doc). KHÔNG chạy local (máy user lag + mạng yếu).

Quy trình:
    python rebuild_corpus_vbpl.py ids       # stream HF keywords → data/sme_doc_ids_all.json {docNum: vbpl_id}
    python rebuild_corpus_vbpl.py rebuild    # fetch tất cả id (đa luồng) → data/corpus_articles.jsonl (mới)
→ rồi embed_corpus.py để sinh corpus_emb.npy mới (corpus đã đổi → PHẢI re-embed).
"""
import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
import local_models_config as cfg
from legal_text_parser import parse_legal_name
from vbpl_fetch import fetch_doc, parse_articles, extract_id, is_in_force
import build_corpus as bc   # tái dùng _keep / _pick_doc_number (DRY)

IDS_ALL = os.path.join(cfg.DATA_DIR, "sme_doc_ids_all.json")


def collect_ids():
    """Stream HF keywords → {docNum: vbpl_id} cho MỌI doc khớp SME (kể cả null-markdown)."""
    from datasets import load_dataset
    ds = load_dataset(cfg.HF_VBPL, "documents", split="train", streaming=True)
    try:
        ds = ds.select_columns(["doc_number", "title", "legal_type", "source_url", "summary"])
    except Exception as e:
        print("(select_columns skip:", e, ")")
    ids, n, miss = {}, 0, 0
    for row in ds:
        n += 1
        if n % 20000 == 0:
            print(f"  scanned {n} | matched {len(ids)} | no-id {miss}")
        if not bc._keep(row, "keywords"):
            continue
        code = bc._pick_doc_number(row.get("doc_number"))
        if not code or code in ids:
            continue
        vid = extract_id(row.get("source_url", ""))
        if vid:
            ids[code] = vid
        else:
            miss += 1
    json.dump(ids, open(IDS_ALL, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"\nMatched SME docs: {len(ids)} có id (+{miss} thiếu id) / {n} scanned → {IDS_ALL}")


def rebuild(ids_path, workers=12, out_path=None):
    out_path = out_path or cfg.CORPUS_JSONL
    ids = json.load(open(ids_path, encoding="utf-8"))
    items = list(ids.items())     # (docNum, vbpl_id)
    print(f"fetch {len(items)} doc qua vbpl API ({workers} luồng)...")

    rows, stat = [], {"ok": 0, "expired": 0, "fail": 0, "noart": 0}
    lock = threading.Lock()

    def work(pair):
        docNum, vid = pair
        d = fetch_doc(vid)
        if not d:
            with lock: stat["fail"] += 1
            return
        if not is_in_force(d.get("effStatus")):     # GROUND-TRUTH hiệu lực
            with lock: stat["expired"] += 1
            return
        arts = parse_articles(d.get("content", ""))
        if not arts:
            with lock: stat["noart"] += 1
            return
        meta = parse_legal_name(d.get("title", ""), docNum, d.get("docType", "") or "")
        yr = (re.search(r"(\d{4})", d.get("issueDate", "") or "") or [None, ""])[0] if d.get("issueDate") else ""
        slug = re.sub(r"[^A-Za-z0-9]", "", docNum)
        out = []
        for a in arts:
            out.append({
                "id": f"{slug}_{a['article'].replace(' ', '')}",
                "doc_number": docNum, "clean_name": meta["clean_name"], "legal_type": meta["type"],
                "year": yr, "article": a["article"], "title": a["title"], "text": a["text"][:4000],
                "source_url": f"https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc/{vid}",
            })
        with lock:
            rows.extend(out); stat["ok"] += 1
            if stat["ok"] % 500 == 0:
                print(f"  ok {stat['ok']} | expired {stat['expired']} | fail {stat['fail']} | điều {len(rows)}")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, items))

    # dedup id (cùng docNum+Điều) + ghi
    seen, uniq = set(), []
    for r in rows:
        if r["id"] in seen:
            r["id"] = r["id"] + "_" + str(len(uniq))
        seen.add(r["id"]); uniq.append(r)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in uniq:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    docs = len({r["doc_number"] for r in uniq})
    print(f"\n✓ {out_path}: {len(uniq)} điều / {docs} văn bản CÒN HIỆU LỰC")
    print(f"  fetch: ok {stat['ok']} | hết hiệu lực (bỏ) {stat['expired']} | no-article {stat['noart']} | fail {stat['fail']}")
    print("  → chạy embed_corpus.py để sinh corpus_emb.npy MỚI (corpus đã đổi).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ids", "rebuild"])
    ap.add_argument("--ids", default=IDS_ALL, help="file id cho mode rebuild")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=None, help="đường dẫn corpus ra (mặc định data/corpus_articles.jsonl)")
    a = ap.parse_args()
    if a.mode == "ids":
        collect_ids()
    else:
        rebuild(a.ids, a.workers, a.out)
