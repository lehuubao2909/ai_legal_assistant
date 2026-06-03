import json
import os

db_path = "/Users/huubao/Documents/GOKU/Dev/DEV_AI/ai_legal_assistant/data/legal_db.json"

with open(db_path, "r", encoding="utf-8") as f:
    db = json.load(f)

print("Total items:", len(db))

# Search for "hình sự" and "117"
matches = []
for idx, item in enumerate(db):
    law_name = item.get("law_name", "")
    article = item.get("article", "")
    title = item.get("title", "")
    
    if "hình sự" in law_name.lower():
        if "117" in article or "117" in title:
            matches.append(item)

print(f"Found {len(matches)} matches:")
for m in matches[:5]:
    print(f"\nID: {m.get('id')}")
    print(f"Law: {m.get('law_name')}")
    print(f"Article: {m.get('article')}")
    print(f"Title: {m.get('title')}")
    print(f"Text snippet: {m.get('text')[:200]}")
