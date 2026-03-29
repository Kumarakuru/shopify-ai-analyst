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
    # Railway Volume Mount Path
    volume_path = "/app/shopify_chroma"
    # The path where GitHub files land (relative to the script)
    github_deploy_path = os.path.join(os.getcwd(), "shopify_chroma")
    
    st.sidebar.info(f"Checking volume at: {volume_path}")

    # 1. Check if the volume is empty
    volume_exists = os.path.exists(volume_path)
    volume_files = os.listdir(volume_path) if volume_exists else []
    
    if not volume_exists or len(volume_files) == 0:
        st.sidebar.warning("Volume appears empty.")
        
        # 2. Try to find the files from the GitHub deployment to "seed" the volume
        if os.path.exists(github_deploy_path) and github_deploy_path != volume_path:
            st.sidebar.info("Found data in GitHub deployment. Copying to persistent volume...")
            try:
                shutil.copytree(github_deploy_path, volume_path, dirs_exist_ok=True)
                st.sidebar.success("✅ Volume initialized from GitHub data.")
            except Exception as e:
                st.sidebar.error(f"Copy failed: {e}")
        else:
            st.sidebar.error("No source data found in GitHub folder to copy.")

    # 3. Connect to Chroma using the Volume Path
    client = chromadb.PersistentClient(path=volume_path)
    
    try:
        collection = client.get_collection("shopify_reports")
        count = collection.count()
        if count > 0:
            st.sidebar.success(f"🎊 Database Ready: {count} vectors loaded.")
        else:
            st.sidebar.warning("⚠️ Database connected but contains 0 vectors.")
        return collection
    except Exception as e:
        st.sidebar.error(f"Chroma Error: {str(e)}")
        # Fallback: Create if doesn't exist
        collection = client.get_or_create_collection(
            name="shopify_reports",
            metadata={"hnsw:space": "cosine"}
        )
        return collection

# Run the loader
collection = get_collection()

# Main Status Indicator
if collection.count() > 0:
    st.success(f"✅ System Online | {collection.count()} Reports Indexed")
else:
    st.error("❌ System Empty | Please upload your shopify_chroma folder to the Railway Volume.")

# ====================== QUERY SECTION ======================
st.header("Ask About Your Store")

preset = st.selectbox("Quick Questions", [
    "Show current sales overview (top 5 best sellers + total revenue)",
    "What are the weak areas / points to improve (low stock, slow movers)",
    "Prepare a PO for next 60 days — suggest reorder quantities for fast movers with low stock",
    "Give me a full store health report with action items",
    "Custom query..."
])

query = st.text_input("Your question") if preset == "Custom query..." else preset

if st.button("🚀 Get Answer + PO"):
    if collection.count() == 0:
        st.error("No data found. Ensure 'shopify_chroma' folder is present in the Railway volume.")
    else:
        with st.spinner("Analyzing store data..."):
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

st.caption("Running on Railway with persistent volume • Data persists across restarts")