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
            batch = texts[i:i+batch_size]
            try:
                emb_response = embed_client.embeddings.create(
                    input=batch,
                    model="Qwen3-Embedding-4B",   # try with model name
                    encoding_format="float"
                )
                embeddings = [data.embedding for data in emb_response.data]

                collection.add(
                    documents=batch,
                    embeddings=embeddings,
                    ids=[f"doc_{i+j}" for j in range(len(batch))]
                )
                success_count += len(batch)
                st.success(f"✅ Vectorized batch {i//batch_size + 1} ({len(batch)} rows)")

            except Exception as e:
                st.error(f"❌ Embedding failed for batch {i//batch_size + 1}: {str(e)[:300]}...")
                st.info("Tip: Confirm LLAMA_ARG_EMBEDDINGS=true and LLAMA_ARG_POOLING=last in your embedding endpoint Settings → Environment")

        if success_count > 0:
            st.success(f"🎉 Total vectorized: {success_count} rows from your reports!")

# ====================== 2. ASK AI ======================
st.header("2. 💬 Ask About Your Store")

preset_options = [
    "Show current sales overview (top 5 best sellers + total revenue)",
    "What are the weak areas / points to improve (low stock, slow movers)",
    "Prepare a PO for next 60 days — suggest reorder quantities for fast movers with low stock",
    "Give me a full store health report with action items",
    "Custom query..."
]

preset = st.selectbox("Quick questions", preset_options)

if preset == "Custom query...":
    query = st.text_input("Type your own question", "What should I focus on this month?")
else:
    query = preset

if st.button("🚀 Get Answer from Qwen2.5-VL (Generation)"):
    if collection.count() == 0:
        st.warning("Please vectorize some reports first!")
    else:
        with st.spinner("Retrieving relevant data + generating answer..."):
            results = collection.query(query_texts=[query], n_results=12)
            context = "\n\n".join(results["documents"][0])

            prompt = f"""You are an expert Shopify store manager.

Context from merchant's reports:
{context}

Question: {query}

Provide a clear, actionable response with:
- Key numbers / summary
- Insights (best sellers, weak areas)
- Ready-to-use PO suggestions (product names, suggested qty, reason)
Use markdown tables where helpful."""

            try:
                answer = gen_client.chat.completions.create(
                    model="qwen2-5-vl-7b-instruct-gguf-mat",   # adjust if your model alias is different
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=2048
                ).choices[0].message.content

                st.markdown("### 📊 AI Answer + PO")
                st.markdown(answer)

                st.download_button(
                    label="📥 Download as TXT",
                    data=answer,
                    file_name="Shopify_AI_Report.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Generation failed: {str(e)}")

st.caption("💡 Deployed on Streamlit Cloud • Data stays in Chroma DB • Pause your HF endpoints when not using to save cost")
