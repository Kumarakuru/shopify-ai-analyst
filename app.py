import streamlit as st
from openai import OpenAI
import chromadb
import os

st.set_page_config(page_title="Shopify AI Analyst", layout="wide")
st.title("🛍️ Shopify AI Analyst (Pre-vectorized Data)")

# ====================== GENERATION ENDPOINT ======================
GEN_URL = "https://pn6ric9mq7jcq9oi.us-east-1.aws.endpoints.huggingface.cloud/v1"
gen_client = OpenAI(base_url=GEN_URL.rstrip('/'), api_key="hf_dummy")

# ====================== LOAD CHROMA READ-ONLY ======================
@st.cache_resource
def get_collection():
    chroma_path = "./shopify_chroma"
    if not os.path.exists(chroma_path):
        st.error("shopify_chroma folder not found in the repo.")
        st.stop()
    
    client = chromadb.PersistentClient(path=chroma_path)
    try:
        # Try to get existing collection without writing
        collection = client.get_collection("shopify_reports")
        return collection
    except Exception as e:
        st.error(f"Failed to load collection: {e}")
        st.error("The shopify_chroma folder may be corrupted or in wrong format.")
        st.stop()

collection = get_collection()

st.success(f"✅ Successfully loaded {collection.count()} vectors from GitHub!")

# ====================== QUERY ======================
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
        st.error("Collection is empty.")
    else:
        with st.spinner("Retrieving context and generating answer..."):
            results = collection.query(query_texts=[query], n_results=12)
            context = "\n\n".join(results["documents"][0])

            prompt = f"""You are an expert Shopify store manager.

Context from pre-vectorized reports:
{context}

Question: {query}

Provide a clear answer with summary, insights, and ready-to-copy PO suggestions. Use tables when helpful."""

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
                st.error(f"Generation failed: {str(e)}")

st.caption("Pre-vectorized data loaded from GitHub • Read-only mode")