import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
import os

st.title("🛍️ Shopify AI Analyst + PO Generator (Free on GitHub)")

# === Secrets (add in .streamlit/secrets.toml or GitHub Secrets later) ===
SHOPIFY_ACCESS_TOKEN = st.secrets.get("SHOPIFY_TOKEN", "your_token_here")
GROQ_API_KEY = st.secrets.get("GROQ_KEY", "gsk_xxx")   # free at groq.com

client = Groq(api_key=GROQ_API_KEY)
embedder = SentenceTransformer("Snowflake/snowflake-arctic-embed-m")  # Arctic Embed — SOTA & tiny

# Persistent Chroma (rebuilds every time — fine for <5000 rows)
@st.cache_resource
def get_chroma():
    return chromadb.PersistentClient(path="./chroma_db")

collection = get_chroma().get_or_create_collection("shopify")

# ====================== UPLOAD / VECTORIZE SECTION ======================
st.header("1. Download these 4 Shopify reports (CSV)")

st.markdown("""
**Go to Shopify Admin → Analytics → Reports** and export these:
1. **Products** → "All products" or "Total sales by product"
2. **Sales by product variant** (best sellers data)
3. **Inventory** (from Products → Inventory → Export)
4. **Orders** (last 30/90 days) or "Total sales by order"

**Alternative (easier & auto):** Use GraphQL bulk export later.

Drag the 4 CSVs here:
""")

uploaded = st.file_uploader("Upload your Shopify CSVs (multiple ok)", accept_multiple_files=True, type="csv")

if st.button("🚀 Vectorize with Arctic Embed + Save") and uploaded:
    texts = []
    for file in uploaded:
        df = pd.read_csv(file)
        for _, row in df.iterrows():
            text = f"Product: {row.get('Title') or row.get('Product')}\n"
            text += f"Sales: {row.get('Net quantity') or row.get('Sales')}\n"
            text += f"Inventory: {row.get('Available') or row.get('On hand')}\n"
            text += f"Vendor: {row.get('Vendor')}\n"
            text += f"Description: {row.get('Body') or ''}\n---"
            texts.append(text)
    
    embeddings = embedder.encode(texts)
    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=[str(i) for i in range(len(texts))]
    )
    st.success(f"✅ Vectorized {len(texts)} chunks with Snowflake Arctic Embed!")

# ====================== QUERY SECTION ======================
query = st.text_input("Ask anything", "Show best sellers last 30 days, weak areas (low stock/high return), and prepare PO for top 3 fast movers")

if st.button("Ask AI"):
    # Retrieve from vector DB
    results = collection.query(query_texts=[query], n_results=10)
    context = "\n\n".join(results["documents"][0])
    
    prompt = f"""You are expert Shopify store manager.
    Context (vectorized reports):\n{context}\n
    User: {query}
    Answer naturally + give clear tables + ready-to-copy PO if asked."""
    
    chat = client.chat.completions.create(
        model="llama3-70b-8192",  # or "qwen-qwq-32b" if available — feels like Qwen3 boosted by RAG
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    answer = chat.choices[0].message.content
    st.markdown("### 📊 Answer + PO")
    st.write(answer)
    st.download_button("📥 Download PO as .txt", answer, "purchase_order.txt")

st.caption("💡 This runs 100% free on Streamlit Cloud + Groq free tier + real Arctic Embed")
