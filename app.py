import streamlit as st
from openai import OpenAI
import chromadb
import shutil
import os

st.set_page_config(page_title="Shopify AI Analyst", layout="wide")
st.title("🛍️ Shopify AI Analyst")

# ====================== GENERATION ENDPOINT ======================
GEN_URL = "https://pn6ric9mq7jcq9oi.us-east-1.aws.endpoints.huggingface.cloud/v1"
gen_client = OpenAI(base_url=GEN_URL.rstrip('/'), api_key="hf_dummy")

# ====================== LOAD CHROMA FROM VOLUME ======================
@st.cache_resource
def get_collection():
    volume_path = "/app/shopify_chroma"
    
    # Check if we are running on Railway (where this path exists)
    if os.path.exists(volume_path):
        client = chromadb.PersistentClient(path=volume_path)
    else:
        # Local fallback
        client = chromadb.PersistentClient(path="./shopify_chroma")
    
    try:
        # This will show us what collections actually exist in your uploaded folder
        collections = client.list_collections()
        if not collections:
            st.sidebar.error("No collections found in the folder.")
            return client.get_or_create_collection(name="shopify_reports")
        
        # Log the names of collections found
        col_names = [c.name for c in collections]
        st.sidebar.info(f"Found collections: {', '.join(col_names)}")
        
        # Pick the first one available or 'shopify_reports'
        target_name = "shopify_reports" if "shopify_reports" in col_names else col_names[0]
        collection = client.get_collection(target_name)
        
        st.sidebar.success(f"✅ Loaded {collection.count()} vectors from '{target_name}'")
        return collection
    except Exception as e:
        st.sidebar.error(f"Error: {e}")
        return client.get_or_create_collection(name="shopify_reports")

collection = get_collection()

# ====================== UI LOGIC ======================
if collection.count() > 0:
    st.success(f"✅ System Online | {collection.count()} Reports Indexed")
else:
    st.error("❌ System Empty | Check sidebar for collection names.")

st.header("Ask About Your Store")
preset = st.selectbox("Quick Questions", [
    "Show current sales overview (top 5 best sellers + total revenue)",
    "What are the weak areas / points to improve (low stock, slow movers)",
    "Prepare a PO for next 60 days",
    "Custom query..."
])

query = st.text_input("Your question") if preset == "Custom query..." else preset

if st.button("🚀 Get Answer"):
    if collection.count() == 0:
        st.error("No data available.")
    else:
        with st.spinner("Analyzing..."):
            results = collection.query(query_texts=[query], n_results=10)
            context = "\n\n".join(results["documents"][0])
            prompt = f"Context:\n{context}\n\nQuestion: {query}"
            
            resp = gen_client.chat.completions.create(
                model="qwen2-5-vl-7b-instruct-gguf-mat",
                messages=[{"role": "user", "content": prompt}]
            )
            st.markdown(resp.choices[0].message.content)

st.caption("Railway Deployment • Persistent Volume Active")