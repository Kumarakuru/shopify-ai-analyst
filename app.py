import streamlit as st
import pandas as pd
from openai import OpenAI
import chromadb

st.set_page_config(page_title="Shopify AI Analyst", layout="wide")
st.title("🛍️ Shopify AI Analyst — HF llama.cpp")

# ====================== HF TOKEN ======================
st.sidebar.header("Authentication")
hf_token = st.sidebar.text_input(
    "Hugging Face Token (hf_...)",
    type="password",
    help="Create at https://huggingface.co/settings/tokens (Read permission)"
)

if not hf_token:
    st.warning("Please enter your Hugging Face token in the sidebar to continue.")
    st.stop()

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

embed_client = OpenAI(base_url=embed_url.rstrip('/'), api_key=hf_token)
gen_client   = OpenAI(base_url=gen_url.rstrip('/'), api_key=hf_token)

# ====================== CHROMA DB - FIXED DIMENSION ======================
@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path="./shopify_chroma")
    
    # Delete old collection if dimension mismatch exists
    try:
        existing = client.get_collection("shopify_reports")
        if existing:
            client.delete_collection("shopify_reports")
            st.sidebar.success("Old collection deleted due to dimension mismatch")
    except:
        pass  # No old collection or already deleted

    # Create new collection with correct dimension for Qwen3-Embedding-4B (2560)
    return client.get_or_create_collection(
        name="shopify_reports",
        metadata={"hnsw:space": "cosine"}
    )

collection = get_collection()

st.sidebar.metric("Stored Vectors", collection.count())

# ====================== 1. VECTORIZE ======================
st.header("1. 📁 Upload & Vectorize Shopify Reports")
st.info("Upload your CSVs. The data will be saved persistently after successful vectorization.")

uploaded = st.file_uploader("Drop Shopify CSV files here", type="csv", accept_multiple_files=True)

if st.button("🚀 Vectorize Reports") and uploaded:
    with st.spinner("Vectorizing with Qwen3-Embedding (dimension 2560)..."):
        texts = []
        for file in uploaded:
            try:
                df = pd.read_csv(file)
                for _, row in df.iterrows():
                    row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                    texts.append(row_text)
            except Exception as e:
                st.warning(f"Could not read {file.name}")

        if not texts:
            st.error("No data found in files.")
            st.stop()

        batch_size = 100   # Safe batch size for T4
        success_count = 0

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = embed_client.embeddings.create(
                    input=batch,
                    encoding_format="float"
                )
                embeddings = [data.embedding for data in response.data]

                collection.add(
                    documents=batch,
                    embeddings=embeddings,
                    ids=[f"doc_{i+j}" for j in range(len(batch))]
                )
                success_count += len(batch)
                st.success(f"✅ Batch {i//batch_size + 1} completed ({len(batch)} rows)")
            except Exception as e:
                st.error(f"Batch {i//batch_size + 1} failed: {str(e)[:250]}")

        if success_count > 0:
            st.success(f"🎉 Successfully stored {success_count} vectors persistently!")
            st.rerun()

# ====================== 2. ASK QUESTIONS ======================
st.header("2. 💬 Ask About Your Store")

preset = st.selectbox("Quick Questions", [
    "Show current sales overview (top 5 best sellers + total revenue)",
    "What are the weak areas / points to improve (low stock, slow movers)",
    "Prepare a PO for next 60 days — suggest reorder quantities for fast movers with low stock",
    "Give me a full store health report with action items",
    "Custom query..."
])

if preset == "Custom query...":
    query = st.text_input("Your question", "How is my store performing this month?")
else:
    query = preset

if st.button("🚀 Get Answer + PO"):
    if collection.count() == 0:
        st.warning("No data stored yet. Please vectorize your reports first.")
    else:
        with st.spinner("Searching stored data and generating answer..."):
            results = collection.query(query_texts=[query], n_results=12)
            context = "\n\n".join(results["documents"][0])

            prompt = f"""You are an expert Shopify store manager.

Context from reports:
{context}

Question: {query}

Answer with:
- Summary
- Key Insights (best/worst areas)
- Ready-to-copy Purchase Order suggestions

Use tables when helpful."""

            try:
                resp = gen_client.chat.completions.create(
                    model="qwen2-5-vl-7b-instruct-gguf-mat",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=2048
                )
                answer = resp.choices[0].message.content

                st.markdown("### 📊 Answer")
                st.markdown(answer)

                st.download_button("📥 Download Report", answer, "Shopify_Report_PO.txt")
            except Exception as e:
                st.error(f"Generation error: {str(e)}")

st.sidebar.caption(f"Total stored vectors: {collection.count()}")
st.caption("Data is saved persistently in ./shopify_chroma • You should not need to re-vectorize every time")
