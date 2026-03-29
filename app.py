import streamlit as st
from openai import OpenAI
import chromadb
import os

# Set page config at the very top
st.set_page_config(page_title="Shopify AI Analyst", layout="wide")

st.title("🛍️ Shopify AI Analyst")

# ====================== ENDPOINT CONFIGURATION ======================
EMBED_URL = "https://rnk392h3d7rcmjm3.us-east4.gcp.endpoints.huggingface.cloud/v1"
GEN_URL = "https://pn6ric9mq7jcq9oi.us-east-1.aws.endpoints.huggingface.cloud/v1"

embed_client = OpenAI(base_url=EMBED_URL.rstrip('/'), api_key="hf_dummy")
gen_client = OpenAI(base_url=GEN_URL.rstrip('/'), api_key="hf_dummy")

# ====================== LOAD CHROMA DATA ======================
@st.cache_resource
def get_collection():
    path = "./shopify_chroma"
    if not os.path.exists(path):
        st.error(f"Directory not found: {path}")
        return None
        
    client = chromadb.PersistentClient(path=path)
    
    try:
        cols = client.list_collections()
        if not cols:
            st.sidebar.warning("No collections found in the folder.")
            return None
            
        col_names = [c.name for c in cols]
        target = "shopify_reports" if "shopify_reports" in col_names else col_names[0]
        
        collection = client.get_collection(name=target)
        st.sidebar.success(f"✅ Loaded {collection.count()} vectors")
        return collection
    except Exception as e:
        st.sidebar.error(f"Database Error: {str(e)}")
        return None

# Attempt to load collection
collection = get_collection()

# ====================== UI & QUERY LOGIC ======================
if collection is not None:
    st.header("Ask About Your Store")
    
    preset = st.selectbox("Quick Questions", [
        "Show current sales overview (top 5 best sellers + total revenue)",
        "What are the weak areas / points to improve (low stock, slow movers)",
        "Prepare a PO for next 60 days",
        "Give me a full store health report",
        "Custom query..."
    ])

    query = st.text_input("Your question") if preset == "Custom query..." else preset

    if st.button("🚀 Analyze Data"):
        with st.spinner("Communicating with AI models..."):
            try:
                # 1. Test Embedding Endpoint Connection
                try:
                    resp_embed = embed_client.embeddings.create(
                        input=[query],
                        model="Qwen3-Embedding-4B"
                    )
                    query_vector = resp_embed.data[0].embedding
                except Exception as e:
                    st.error(f"🛑 Embedding API Error (HF 503): The embedding server is currently offline or busy. Details: {str(e)}")
                    st.stop()

                # 2. Query Chroma
                results = collection.query(
                    query_embeddings=[query_vector], 
                    n_results=10
                )
                context = "\n\n".join(results["documents"][0])

                # 3. Generate Answer
                try:
                    prompt = f"You are an expert Shopify Analyst.\n\nContext:\n{context}\n\nQuestion: {query}"
                    resp_gen = gen_client.chat.completions.create(
                        model="qwen2-5-vl-7b-instruct-gguf-mat",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3
                    )
                    st.markdown("### 📊 Analysis Report")
                    st.markdown(resp_gen.choices[0].message.content)
                except Exception as e:
                    st.error(f"🛑 Generation API Error: The reasoning server failed to respond. Details: {str(e)}")
                
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
else:
    st.info("Please ensure the 'shopify_chroma' folder is present in your GitHub repository.")
    if st.button("Retry Connection"):
        st.rerun()

st.caption("Running on Railway • Data loaded from GitHub Repository")