import streamlit as st
from openai import OpenAI
import chromadb

st.set_page_config(page_title="Shopify AI Analyst", layout="wide")
st.title("🛍️ Shopify AI Analyst")

# ====================== GENERATION ENDPOINT ======================
GEN_URL = "https://pn6ric9mq7jcq9oi.us-east-1.aws.endpoints.huggingface.cloud/v1"
gen_client = OpenAI(base_url=GEN_URL.rstrip('/'), api_key="hf_dummy")

# ====================== LOAD CHROMA FROM VOLUME ======================
@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path="/app/shopify_chroma")
    
    try:
        collection = client.get_collection("shopify_reports")
        st.sidebar.success(f"✅ Loaded existing collection with {collection.count()} vectors")
        return collection
    except Exception:
        # Collection doesn't exist yet - create it
        st.sidebar.warning("Creating new collection...")
        collection = client.get_or_create_collection(
            name="shopify_reports",
            metadata={"hnsw:space": "cosine"}
        )
        st.sidebar.info("New empty collection created. Run local vectorize and re-upload folder if needed.")
        return collection

collection = get_collection()

st.info(f"Current vectors in database: {collection.count()}")

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
    if collection.count() == 0:
        st.error("No data in the database yet. Please run your local vectorize script and upload the shopify_chroma folder again.")
    else:
        with st.spinner("Retrieving context and generating answer..."):
            results = collection.query(query_texts=[query], n_results=12)
            context = "\n\n".join(results["documents"][0])

            prompt = f"""You are an expert Shopify store manager.

Context from your store reports:
{context}

Question: {query}

Answer with summary, insights, and ready-to-copy PO suggestions. Use tables when helpful."""

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

st.caption("Running on Railway with persistent volume • Data should persist now")