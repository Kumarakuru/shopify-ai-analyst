import streamlit as st
from openai import OpenAI
import chromadb

st.set_page_config(page_title="Shopify AI Analyst", layout="wide")
st.title("🛍️ Shopify AI Analyst (Pre-vectorized on PC)")

# ====================== YOUR EXACT ENDPOINTS ======================
st.subheader("HF Endpoints")

col1, col2 = st.columns(2)
with col1:
    embed_url = st.text_input(
        "Embedding Endpoint (not used here)",
        "https://rnk392h3d7rcmjm3.us-east4.gcp.endpoints.huggingface.cloud/v1",
        disabled=True
    )
with col2:
    gen_url = st.text_input(
        "Generation Endpoint",
        "https://pn6ric9mq7jcq9oi.us-east-1.aws.endpoints.huggingface.cloud/v1",
        disabled=True
    )

gen_client = OpenAI(base_url=gen_url.rstrip('/'), api_key="hf_dummy")

# ====================== LOAD CHROMA WITH DIMENSION FIX ======================
@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path="./shopify_chroma")
    
    # Force delete old collection if dimension mismatch exists
    try:
        old_collection = client.get_collection("shopify_reports")
        client.delete_collection("shopify_reports")
        st.sidebar.success("Old mismatched collection deleted (dimension fixed)")
    except:
        pass  # No collection or already deleted

    # Create fresh collection with correct 2560 dimension for Qwen3-Embedding-4B
    return client.get_or_create_collection(
        name="shopify_reports",
        metadata={"hnsw:space": "cosine"}
    )

collection = get_collection()

stored_count = collection.count()
st.info(f"**Loaded {stored_count} vectors** from GitHub (pre-vectorized)")

if stored_count == 0:
    st.warning("⚠️ No vectors found. Make sure the 'shopify_chroma' folder is uploaded to your GitHub repo root.")
    st.stop()

# ====================== ASK QUESTIONS ======================
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
    with st.spinner("Retrieving context + generating answer..."):
        results = collection.query(query_texts=[query], n_results=12)
        context = "\n\n".join(results["documents"][0])

        prompt = f"""You are an expert Shopify store manager.

Context from pre-vectorized store reports:
{context}

Question: {query}

Answer clearly with:
- Summary of key numbers
- Insights (best sellers, weak areas)
- Ready-to-copy Purchase Order (PO) suggestions

Use markdown tables when helpful."""

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

            st.download_button("📥 Download Report + PO", answer, "Shopify_Report_PO.txt")
        except Exception as e:
            st.error(f"Generation error: {str(e)[:400]}")

st.caption("Vectors loaded from GitHub • Only chat completion runs on HF")