import streamlit as st
from openai import OpenAI
import chromadb

st.set_page_config(page_title="Shopify AI Analyst", layout="wide")
st.title("🛍️ Shopify AI Analyst (Pre-vectorized Data)")

# ====================== YOUR GENERATION ENDPOINT ======================
GEN_URL = "https://pn6ric9mq7jcq9oi.us-east-1.aws.endpoints.huggingface.cloud/v1"
gen_client = OpenAI(base_url=GEN_URL.rstrip('/'), api_key="hf_dummy")

# ====================== LOAD CHROMA READ-ONLY ======================
@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path="./shopify_chroma")
    try:
        # Try to get existing collection without writing
        collection = client.get_collection("shopify_reports")
        st.sidebar.success("✅ Loaded existing collection from GitHub")
        return collection
    except Exception as e:
        st.error(f"Could not load collection: {e}")
        st.error("Make sure the 'shopify_chroma' folder is correctly uploaded to the root of your GitHub repo.")
        st.stop()

collection = get_collection()

st.info(f"✅ Loaded {collection.count()} vectors from GitHub (pre-vectorized locally)")

if collection.count() == 0:
    st.warning("No vectors found in the uploaded folder.")
    st.stop()

# ====================== QUERY SECTION ======================
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
    with st.spinner("Retrieving context and generating answer..."):
        results = collection.query(query_texts=[query], n_results=12)
        context = "\n\n".join(results["documents"][0])

        prompt = f"""You are an expert Shopify store manager.

Context from your store reports:
{context}

Question: {query}

Answer with:
- Summary
- Key insights (best sellers, weak areas)
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
            st.download_button("📥 Download Report + PO", answer, "Shopify_Report_PO.txt")
        except Exception as e:
            st.error(f"Generation failed: {str(e)}")

st.caption("Vectors loaded from GitHub • Read-only mode • Only chat runs on HF")