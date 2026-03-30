import pandas as pd
from openai import OpenAI
import chromadb
import os
import sys
import argparse
from tqdm import tqdm

# ====================== ARGS ======================
parser = argparse.ArgumentParser(description="Replace one CSV file's vectors in ChromaDB")
parser.add_argument("--old_file", required=True, help="Old CSV filename to remove (e.g. Inventory-OLD.csv)")
parser.add_argument("--new_file", required=True, help="New CSV filename to index (e.g. Inventory-NEW.csv)")
args = parser.parse_args()

if not os.path.exists(args.new_file):
    print(f"❌ New file not found: {args.new_file}")
    sys.exit(1)

# ====================== CLIENTS ======================
EMBED_URL = "https://rnk392h3d7rcmjm3.us-east4.gcp.endpoints.huggingface.cloud/v1"
embed_client = OpenAI(base_url=EMBED_URL.rstrip('/'), api_key="hf_dummy")

client_db = chromadb.PersistentClient(path="./shopify_chroma")
collection = client_db.get_or_create_collection(
    name="shopify_reports",
    metadata={"hnsw:space": "cosine"}
)

# ====================== STEP 1: DELETE OLD VECTORS ======================
print(f"🔍 Scanning for vectors from: {args.old_file}")
existing = collection.get(include=["documents"])

old_ids = [
    existing["ids"][i]
    for i, doc in enumerate(existing["documents"])
    if f"source_file: {args.old_file}" in doc
]

if old_ids:
    collection.delete(ids=old_ids)
    print(f"🗑️  Deleted {len(old_ids)} vectors from '{args.old_file}'")
else:
    print(f"⚠️  No vectors found matching '{args.old_file}'")
    print("    Make sure you ran reindex_all.py first to tag vectors with source filenames.")
    sys.exit(1)

# ====================== STEP 2: INDEX NEW FILE ======================
print(f"\n📂 Reading new file: {args.new_file}")
df = pd.read_csv(args.new_file)

texts = []
for _, row in df.iterrows():
    row_text = (
        f"source_file: {args.new_file} | "
        + " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
    )
    texts.append(row_text)

print(f"📝 {len(texts)} rows to embed")

# Get current max ID to avoid collisions
existing_ids = collection.get()["ids"]
if existing_ids:
    max_id = max(int(id.split("_")[1]) for id in existing_ids if id.startswith("doc_"))
    start_index = max_id + 1
else:
    start_index = 0

# ====================== VECTORIZE ======================
batch_size = 80
success = 0

for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    try:
        response = embed_client.embeddings.create(
            input=batch,
            model="Qwen3-Embedding-4B",
            encoding_format="float"
        )
        embeddings = [d.embedding for d in response.data]
        collection.add(
            documents=batch,
            embeddings=embeddings,
            ids=[f"doc_{start_index + i + j}" for j in range(len(batch))]
        )
        success += len(batch)
        print(f"✅ Batch {i//batch_size + 1} done — {len(batch)} vectors added")
    except Exception as e:
        print(f"❌ Batch {i//batch_size + 1} failed: {e}")

print(f"\n🎉 Done! Replaced '{args.old_file}' with '{args.new_file}' ({success} vectors added)")
print(f"📦 Total vectors in collection: {collection.count()}")