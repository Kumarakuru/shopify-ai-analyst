import streamlit as st
import pandas as pd
from openai import OpenAI
import chromadb

st.set_page_config(page_title="Shopify AI Analyst", layout="wide")
st.title("🛍️ Shopify AI Analyst (Local Vectorized + HF Chat)")

# ====================== YOUR EXACT GENERATION URL ======================
GEN_URL = "https://pn6ric9mq7jcq9oi.us-east-1.aws.endpoints.huggingface.cloud/v1"

gen_client = OpenAI(base_url=GEN_URL.rstrip('/'), api_key="hf_dummy")

# ====================== LOAD PERSISTENT CHROMA FROM GITHUB ======================
@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path="./shopify_chroma")
    return client.get_or_create_collection("shopify_reports")

collection = get_collection()

st.info(f"**Loaded {collection.count()} vectors** from GitHub (pre-vectorized on your PC)")

# ====================== ASK QUESTIONS ======================
st.header("Ask About Your Store")

preset = st.selectbox("Quick Questions", [
    "Show current sales overview (top 5 best sellers + total revenue)",
    "What are the weak areas / points to improve (low stock, slow movers)",
    "Prepare a PO for next 60 days — suggest reorder quantities for fast movers with low stock",
    "Give me a full store health report with action items",
    "Custom query..."
])

if preset == "Custom query...":
    query = st.text_input("Your question")
else:
    query = preset

if st.button("🚀 Get Answer + PO"):
    if collection.count() == 0:
        st.error("No vectors found. Make sure shopify_chroma folder is in the GitHub repo.")
    else:
        with st.spinner("Retrieving context + generating answer..."):
            results = collection.query(query_texts=[query], n_results=15)
            context = "\n\n".join(results["documents"][0])

            prompt = f"""You are an expert Shopify store manager.

Context from pre-vectorized reports:
{context}

Question: {query}

Answer clearly with:
- Summary
- Key insights (best sellers, weak areas)
- Ready-to-copy Purchase Order (PO) suggestions

Use markdown tables when helpful."""

            resp = gen_client.chat.completions.create(
                model="qwen2-5-vl-7b-instruct-gguf-mat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2048
            )
            answer = resp.choices[0].message.content

            st.markdown("### 📊 Answer")
            st.markdown(answer)

            st.download_button("📥 Download Report + PO", answer, "Shopify_AI_Report.txt")

st.caption("Vectorization done locally once • Data loaded from GitHub • Only chat runs on HF")