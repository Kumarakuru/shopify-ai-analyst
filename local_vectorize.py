import pandas as pd
from openai import OpenAI
import chromadb
import os
from tqdm import tqdm

# ====================== ENDPOINT ======================
EMBED_URL = "https://rnk392h3d7rcmjm3.us-east4.gcp.endpoints.huggingface.cloud/v1"
client = OpenAI(base_url=EMBED_URL.rstrip('/'), api_key="hf_dummy")

# ====================== CHROMA DB ======================
client_db = chromadb.PersistentClient(path="./shopify_chroma")
try:
    client_db.delete_collection("shopify_reports")
    print("🗑️  Cleared old collection")
except:
    pass

collection = client_db.get_or_create_collection(
    name="shopify_reports",
    metadata={"hnsw:space": "cosine"}
)
print("✅ ChromaDB ready")

# ====================== CSV FILES ======================
csv_files = [
    "Total sales by product-24MAR26-24MAR25.csv",
    "Total sales by product variant-24MAR26-24MAR25.csv",
    "products_export_1.csv",
    "Inventory-24MAR26-24MAR25.csv"   # <-- updated filename
]

texts = []
doc_index = 0

print("\nReading CSVs...")
for file in csv_files:
    if os.path.exists(file):
        df = pd.read_csv(file)
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {file}"):
            row_text = (
                f"source_file: {file} | "
                + " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
            )
            texts.append(row_text)
    else:
        print(f"⚠️  File not found: {file}")

print(f"\n📝 Total chunks prepared: {len(texts)}")

# ====================== VECTORIZE ======================
batch_size = 80
success = 0

for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    try:
        response = client.embeddings.create(
            input=batch,
            model="Qwen3-Embedding-4B",
            encoding_format="float"
        )
        embeddings = [data.embedding for data in response.data]
        collection.add(
            documents=batch,
            embeddings=embeddings,
            ids=[f"doc_{i+j}" for j in range(len(batch))]
        )
        success += len(batch)
        print(f"✅ Batch {i//batch_size + 1} done — {len(batch)} vectors added")
    except Exception as e:
        print(f"❌ Batch {i//batch_size + 1} failed: {e}")

print(f"\n🎉 Reindex complete! Total stored: {success} vectors")
print("📦 Upload the 'shopify_chroma' folder to your GitHub repo.")