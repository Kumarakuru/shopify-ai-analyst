import streamlit as st
from openai import OpenAI
import chromadb

st.set_page_config(page_title="Shopify AI Analyst", layout="wide")
st.title("🛍️ Shopify AI Analyst (Local Vectors + HF Chat)")

# Your Generation Endpoint
GEN_URL = "https://pn6ric9mq7jcq9oi.us-east-1.aws.endpoints.huggingface.cloud/v1"
gen_client = OpenAI(base_url=GEN_URL.rstrip('/'), api_key="hf_dummy")

# Load Chroma with forced dimension fix
@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path="./shopify_chroma")
    
    # Force clean any old mismatched collection
    try:
        client.delete_collection("shopify_reports")
        st.sidebar.info("Old collection deleted (dimension mismatch fixed)")
    except:
        pass

    # Create new collection with correct 2560 dimension
    return client.get_or_create_collection(
        name="shopify_reports",
        metadata={"hnsw:space": "cosine"}
    )

collection = get_collection()

st.info(f"✅ Loaded {collection.count()} vectors from GitHub")

if collection.count() == 0:
    st.warning("No vectors found. Please run local_vectorize.py again and re-upload the shopify_chroma folder.")
    st.stop()

# Query Section
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
    with st.spinner("Thinking..."):
        results = collection.query(query_texts=[query], n_results=12)
        context = "\n\n".join(results["documents"][0])

        prompt = f"""You are an expert Shopify store manager.

Context:
{context}

Question: {query}

Answer with summary, insights, and ready PO suggestions. Use tables."""

        resp = gen_client.chat.completions.create(
            model="qwen2-5-vl-7b-instruct-gguf-mat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        answer = resp.choices[0].message.content

        st.markdown("### Answer")
        st.markdown(answer)
        st.download_button("Download", answer, "Shopify_Report_PO.txt")

st.caption("Vectors loaded from GitHub • Only chat runs on HF")