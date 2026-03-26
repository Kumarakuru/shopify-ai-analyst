import streamlit as st
import pandas as pd
from openai import OpenAI
import chromadb

st.set_page_config(page_title="Shopify AI Analyst", layout="wide")
st.title("🛍️ Shopify AI Analyst — HF llama.cpp (Qwen3 Embedding + Qwen2.5 VL)")

# ====================== ENDPOINTS ======================
st.subheader("Hugging Face Endpoints")

col1, col2 = st.columns(2)
with col1:
    embed_url = st.text_input(
        "Embedding Endpoint (Qwen3-Embedding-4B)",
        "https://rnk392h3d7rcmjm3.us-east4.gcp.endpoints.huggingface.cloud/v1",
        help="Your qwen3-embedding-4b endpoint"
    )
with col2:
    gen_url = st.text_input(
        "Generation Endpoint (Qwen2.5-VL-7B-Instruct)",
        "https://YOUR-QWEN2.5-VL-ENDPOINT.us-east-1.aws.endpoints.huggingface.cloud/v1",
        help="Paste the full URL of qwen2-5-vl-7b-instruct-gguf-mat + /v1 at the end"
    )

embed_client = OpenAI(base_url=embed_url.rstrip('/'), api_key="hf_any")
gen_client   = OpenAI(base_url=gen_url.rstrip('/'), api_key="hf_any")

# Chroma DB (persistent so data stays between runs on Streamlit Cloud)
@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path="./shopify_chroma")
    return client.get_or_create_collection("shopify_reports")

collection = get_collection()

# ====================== 1. VECTORIZE ======================
st.header("1. 📁 Upload & Vectorize Shopify Reports")
st.info("Upload your Product, Sales by product, Inventory, and Orders CSVs (you can upload multiple times — it appends)")

uploaded = st.file_uploader("Drop Shopify CSVs here", type="csv", accept_multiple_files=True)

if st.button("🚀 Vectorize with Qwen3-Embedding") and uploaded:
    with st.spinner("Processing CSVs and sending to HF embedding endpoint..."):
        texts = []
        for file in uploaded:
            try:
                df = pd.read_csv(file)
                for _, row in df.iterrows():
                    row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                    texts.append(row_text)
            except Exception as e:
                st.warning(f"Could not read {file.name}: {e}")

        if not texts:
            st.error("No valid data found in uploaded files.")
            st.stop()

        # Smaller batch to avoid timeout / memory issues on HF
        batch_size = 200
        success_count = 0

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i
