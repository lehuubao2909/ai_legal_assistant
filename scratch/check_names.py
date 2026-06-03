import json
import re
import os

registry_path = "/Users/huubao/Documents/GOKU/Dev/DEV_AI/ai_legal_assistant/data/document_registry.json"

with open(registry_path, "r", encoding="utf-8") as f:
    registry = json.load(f)

print(f"Total documents in registry: {len(registry)}")

def parse_legal_name(law_name: str):
    law_name = re.sub(r'\s+', ' ', law_name).strip()
    
    # Try to find code, e.g., 46/2019/QH14, 80/2021/NĐ-CP, 04/2017/QH14, 21-LCT/HDNN8
    code_match = re.search(r'\b(\d+/\d+/[A-Z0-9\-đĐgG]+|\d+-[A-Z0-9\-đĐgG/]+)\b', law_name)
    code = ""
    if code_match:
        code = code_match.group(1)
    else:
        # Fallback regex for codes without slash, or custom numbers like "Không số"
        code_match_fallback = re.search(r'\bsố\s+([^\s,]+)', law_name, flags=re.IGNORECASE)
        if code_match_fallback:
            code = code_match_fallback.group(1)
            
    law_types = ["Bộ luật", "Luật", "Nghị định", "Thông tư", "Quyết định", "Hiến pháp"]
    doc_type = ""
    for t in law_types:
        if law_name.lower().startswith(t.lower()):
            doc_type = t
            break
    if not doc_type:
        doc_type = "Văn bản"
        
    subject = law_name
    if doc_type and subject.lower().startswith(doc_type.lower()):
        subject = subject[len(doc_type):].strip()
        
    # Remove "số <code_part>" specifically (with or without 'số')
    if code:
        # Remove "số [code]" or "số:[code]" or "[code]"
        subject = re.sub(rf'\b(số\s+|số:\s*)?{re.escape(code)}\b', '', subject, flags=re.IGNORECASE).strip()
    
    # Remove any trailing "số " or "số" that was left behind
    subject = re.sub(r'\b(số)\s*$', '', subject, flags=re.IGNORECASE).strip()
    # Remove trailing/leading punctuation
    subject = subject.strip(",. ").strip()
    
    clean_name = f"{doc_type} {subject}".strip()
    
    return code, doc_type, subject, clean_name

no_code_count = 0
for idx, doc in enumerate(registry[:30]):
    name = doc.get("law_name", "")
    code, doc_type, subject, clean_name = parse_legal_name(name)
    if not code:
        no_code_count += 1
    print(f"Original: {name}")
    print(f" -> Code: {code} | Clean Name: {clean_name}")
    print("-" * 40)
