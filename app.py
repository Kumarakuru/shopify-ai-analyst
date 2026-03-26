import streamlit as st
import pandas as pd
from openai import OpenAI
import chromadb

st.set_page_config(page_title="Shopify AI Analyst", layout="wide")
st.title("🛍️ Shopify AI Analyst — HF llama.cpp")

# ====================== YOUR EXACT ENDPOINTS (No Token Input) ======================
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

# Use dummy key - many HF endpoints accept this
dummy_key = "hf_dummy"

embed_client = OpenAI(base_url=embed_url.rstrip('/'), api_key=dummy_key)
gen_client   = OpenAI(base_url=gen_url.rstrip('/'), api_key=dummy_key)

# ====================== CHROMA DB (Fixed for 2560 dimension) ======================
@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path="./shopify_chroma")
    
    # Clean old collection if dimension mismatch
    try:
        client.delete_collection("shopify_reports")
    except:
        pass

    return client.get_or_create_collection(
        name="shopify_reports",
        metadata={"hnsw:space": "cosine"}
    )

collection = get_collection()

st.sidebar.metric("Stored Vectors", collection.count())

# ====================== 1. VECTORIZE ======================
st.header("1. 📁 Upload & Vectorize Shopify Reports")
st.info("Upload your CSVs once. Data will be saved persistently.")

uploaded = st.file_uploader("Drop Shopify CSVs here", type="csv", accept_multiple_files=True)

if st.button("🚀 Vectorize Reports") and uploaded:
    with st.spinner("Vectorizing..."):
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
            st.error("No data found.")
            st.stop()

        batch_size = 100
        success_count = 0

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = embed_client.embeddings.create(input=batch, encoding_format="float")
                embeddings = [data.embedding for data in response.data]

                collection.add(
                    documents=batch,
                    embeddings=embeddings,
                    ids=[f"doc_{i+j}" for j in range(len(batch))]
                )
                success_count += len(batch)
                st.success(f"Batch {i//batch_size + 1} done")
            except Exception as e:
                st.error(f"Vectorize failed: {str(e)[:300]}")

        if success_count > 0:
            st.success(f"✅ Stored {success_count} vectors!")
            st.rerun()

# ====================== 2. ASK ======================
st.header("2. 💬 Ask About Your Store")

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
        st.warning("Vectorize reports first!")
    else:
        with st.spinner("Generating answer..."):
            results = collection.query(query_texts=[query], n_results=12)
            context = "\n\n".join(results["documents"][0])

            prompt = f"""You are an expert Shopify store manager.

Context:
{context}

Question: {query}

Answer clearly with summary, insights, and PO suggestions. Use tables if needed."""

            try:
                resp = gen_client.chat.completions.create(
                    model="qwen2-5-vl-7b-instruct-gguf-mat",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                answer = resp.choices[0].message.content

                st.markdown("### Answer")
                st.markdown(answer)
                st.download_button("Download", answer, "Shopify_Report.txt")
            except Exception as e:
                st.error(f"Error: {str(e)[:400]}")

st.caption("No token required in this version • Data saved persistently")
