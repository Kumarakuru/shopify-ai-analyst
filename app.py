import streamlit as st
import pandas as pd
from openai import OpenAI
import chromadb

st.set_page_config(page_title="Shopify AI Analyst", layout="wide")
st.title("🛍️ Shopify AI Analyst — HF llama.cpp")

# ====================== YOUR EXACT ENDPOINTS ======================
st.subheader("Your Endpoints")

col1, col2 = st.columns(2)
with col1:
    embed_url = st.text_input(
        "Embedding Endpoint",
        "https://rnk392h3d7rcmjm3.us-east4.gcp.endpoints.huggingface.cloud/v1",
        disabled=True
    )
with col2:
    gen_url = st.text_input(
        "Generation Endpoint",
        "https://pn6ric9mq7jcq9oi.us-east-1.aws.endpoints.huggingface.cloud/v1",
        disabled=True
    )

dummy_key = "hf_dummy"
embed_client = OpenAI(base_url=embed_url.rstrip('/'), api_key=dummy_key)
gen_client   = OpenAI(base_url=gen_url.rstrip('/'), api_key=dummy_key)

# ====================== CHROMA DB ======================
@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path="./shopify_chroma")
    try:
        client.delete_collection("shopify_reports")
    except:
        pass
    return client.get_or_create_collection(
        name="shopify_reports",
        metadata={"hnsw:space": "cosine"}
    )

collection = get_collection()

stored_count = collection.count()
st.info(f"**Current Status:** {stored_count} vectors stored")

if stored_count == 0:
    st.warning("⚠️ No data stored yet. Upload CSVs and click Vectorize.")
else:
    st.success(f"✅ {stored_count} vectors ready. You can ask questions.")

# ====================== 1. VECTORIZE ======================
st.header("1. 📁 Upload & Vectorize Shopify Reports")

uploaded = st.file_uploader("Drop Shopify CSVs here", type="csv", accept_multiple_files=True)

if st.button("🚀 Vectorize Reports") and uploaded:
    with st.spinner("Vectorizing with Qwen3-Embedding-4B..."):
        texts = []
        for file in uploaded:
            try:
                df = pd.read_csv(file)
                for _, row in df.iterrows():
                    row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                    texts.append(row_text)
            except:
                pass

        if not texts:
            st.error("No data found in the files.")
            st.stop()

        batch_size = 100
        success_count = 0

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                # FIXED: Added model parameter
                response = embed_client.embeddings.create(
                    input=batch,
                    model="Qwen3-Embedding-4B",      # This was missing
                    encoding_format="float"
                )
                embeddings = [data.embedding for data in response.data]

                collection.add(
                    documents=batch,
                    embeddings=embeddings,
                    ids=[f"doc_{i+j}" for j in range(len(batch))]
                )
                success_count += len(batch)
                st.success(f"✅ Batch {i//batch_size + 1} done ({len(batch)} rows)")
            except Exception as e:
                st.error(f"Vectorize failed: {str(e)[:300]}")

        if success_count > 0:
            st.success(f"🎉 Successfully vectorized and stored {success_count}
