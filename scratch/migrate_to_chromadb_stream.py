import json
import os
import sys
import re

# Ensure we can import from backend
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

import chromadb

db_path = "/Users/huubao/Documents/GOKU/Dev/DEV_AI/ai_legal_assistant/data/legal_db.json"
chroma_path = "/Users/huubao/Documents/GOKU/Dev/DEV_AI/ai_legal_assistant/data/chroma_db"

print("Starting memory-efficient STREAM migration from JSON to ChromaDB...")
print(f"Reading and streaming JSON database from {db_path}...")

# 1. Verify files
if not os.path.exists(db_path):
    print(f"Error: {db_path} does not exist!")
    sys.exit(1)

# 2. Connect to ChromaDB
print(f"Initializing ChromaDB at {chroma_path}...")
client = chromadb.PersistentClient(path=chroma_path)

# Delete existing collection if it exists to start fresh and clean
try:
    client.delete_collection("vietnamese_laws")
    print("Deleted old vietnamese_laws collection to start fresh.")
except Exception:
    pass

collection = client.create_collection(
    name="vietnamese_laws",
    metadata={"hnsw:space": "cosine"} # Use cosine similarity for embeddings
)

# 3. Stream parser function (Zero RAM overhead)
def stream_legal_db(file_path):
    buffer = []
    in_record = False
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped == "{":
                in_record = True
                buffer = [line]
            elif in_record:
                buffer.append(line)
                if stripped == "}" or stripped == "},":
                    in_record = False
                    record_str = "".join(buffer).rstrip()
                    if record_str.endswith(","):
                        record_str = record_str[:-1]
                    try:
                        yield json.loads(record_str)
                    except Exception as e:
                        print("Error parsing block:", e)

# 4. Stream and Batch insert
batch_size = 5000
ids = []
embeddings = []
documents = []
metadatas = []

migrated_count = 0
failed_count = 0

print("Streaming and batching items to ChromaDB...")
for idx, item in enumerate(stream_legal_db(db_path)):
    item_id = item.get("id")
    embedding = item.get("embedding")
    text = item.get("text", "")
    title = item.get("title", "")
    
    if not item_id or not embedding:
        failed_count += 1
        continue
        
    ids.append(item_id)
    embeddings.append(embedding)
    # Combine title and text for rich document representation
    documents.append(f"{title}\n{text}")
    
    metadata = {
        "id": item_id,
        "law_name": item.get("law_name", ""),
        "title": title,
        "text": text,
        "type": item.get("type", "parent"),
        "parent_id": item.get("parent_id") or "",
        "effective_date": item.get("effective_date", ""),
        "expiration_date": item.get("expiration_date") or "",
        "status": item.get("status", "active"),
        "source_url": item.get("source_url") or "",
        "article": item.get("article") or "",
        "clause": item.get("clause") or "",
        "point": item.get("point") or ""
    }
    metadatas.append(metadata)
    
    if len(ids) == batch_size:
        try:
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            migrated_count += len(ids)
            print(f"Migrated {migrated_count} items...")
        except Exception as e:
            print(f"Error adding batch to ChromaDB: {e}")
            failed_count += len(ids)
            
        ids = []
        embeddings = []
        documents = []
        metadatas = []

# Flush remaining items
if ids:
    try:
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        migrated_count += len(ids)
        print(f"Migrated remaining {len(ids)} items...")
    except Exception as e:
        print(f"Error adding final batch: {e}")
        failed_count += len(ids)

print("\n=== Stream Migration Complete ===")
print(f"Successfully migrated: {migrated_count}")
print(f"Failed/Skipped: {failed_count}")
print(f"ChromaDB collection count: {collection.count()}")
