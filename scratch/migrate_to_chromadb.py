import json
import os
import sys

# Ensure we can import from backend
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

import chromadb

db_path = "/Users/huubao/Documents/GOKU/Dev/DEV_AI/ai_legal_assistant/data/legal_db.json"
chroma_path = "/Users/huubao/Documents/GOKU/Dev/DEV_AI/ai_legal_assistant/data/chroma_db"

print("Starting migration from JSON to ChromaDB...")
print(f"Reading JSON database from {db_path}...")

# 1. Load the JSON database
if not os.path.exists(db_path):
    print(f"Error: {db_path} does not exist!")
    sys.exit(1)

with open(db_path, "r", encoding="utf-8") as f:
    legal_db = json.load(f)

total_items = len(legal_db)
print(f"Successfully loaded {total_items} items from JSON.")

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

# 3. Batch migrate
batch_size = 5000
ids = []
embeddings = []
documents = []
metadatas = []

migrated_count = 0
failed_count = 0

print("Migrating items in batches...")
for idx, item in enumerate(legal_db):
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
    
    # Chroma metadata keys must be simple types (str, int, float, bool)
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
        "source_url": item.get("source_url") or ""
    }
    metadatas.append(metadata)
    
    # When batch is full, add to collection
    if len(ids) == batch_size or idx == total_items - 1:
        try:
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            migrated_count += len(ids)
            print(f"Migrated {migrated_count}/{total_items} items...")
        except Exception as e:
            print(f"Error adding batch to ChromaDB at index {idx}: {e}")
            failed_count += len(ids)
            
        # Reset lists
        ids = []
        embeddings = []
        documents = []
        metadatas = []

print("\n=== Migration Complete ===")
print(f"Total processed: {total_items}")
print(f"Successfully migrated: {migrated_count}")
print(f"Failed/Skipped: {failed_count}")
print(f"ChromaDB collection count: {collection.count()}")
