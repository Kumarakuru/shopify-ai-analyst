import pandas as pd
from openai import OpenAI
import chromadb
import os
from tqdm import tqdm

# ====================== YOUR EXACT URLs ======================
EMBED_URL = "https://rnk392h3d7rcmjm3.us-east4.gcp.endpoints.huggingface.cloud/v1"

client = OpenAI(base_url=EMBED_URL.rstrip('/'), api_key="hf_dummy")   # dummy works for most endpoints

# ====================== CHROMA DB (2560 dimension) ======================
client_db = chromadb.PersistentClient(path="./shopify_chroma")
try:
    client_db.delete_collection("shopify_reports")
except:
    pass

collection = client_db.get_or_create_collection(
    name="shopify_reports",
    metadata={"hnsw:space": "cosine"}
)

print("✅ ChromaDB ready (dimension 2560 for Qwen3-Embedding-4B)")

# ====================== LOAD YOUR CSVs ======================
csv_files = [
    "Total sales by product-24MAR26-24MAR25.csv",
    "Total sales by product variant-24MAR26-24MAR25.csv",
    "products_export_1.csv",
    "Inventory-24MAR26-24MAR25-01.csv"
    # Add any other CSV files you have here
]

texts = []

print("Reading CSVs...")
for file in csv_files:
    if os.path.exists(file):
        df = pd.read_csv(file)
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {file}"):
            row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
            texts.append(row_text)
    else:
        print(f"⚠️ File not found: {file}")

print(f"Total chunks prepared: {len(texts)}")

# ====================== VECTORIZE (using your HF embedding endpoint) ======================
batch_size = 80   # safe for your T4 endpoint
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
        print(f"✅ Batch {i//batch_size + 1} done - {len(batch)} vectors added")
    except Exception as e:
        print(f"❌ Batch failed: {e}")

print(f"\n🎉 Vectorization finished! Total stored: {success} vectors")
print("Folder 'shopify_chroma' is now saved in the current directory.")
print("Next step: Zip this folder and upload it to your GitHub repo.")